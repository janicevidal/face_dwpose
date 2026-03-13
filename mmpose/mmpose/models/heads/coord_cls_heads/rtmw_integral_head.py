# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy, keypoint_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors, flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMWIntegralHead(BaseHead):
    """Top-down head introduced in RTMPose-Wholebody (2023).

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        with_rle: bool = False,
        beta: float = 1.0,
        loss_cfg: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.with_rle = with_rle
        self.beta = beta
        
        self.loss_fun = loss
        self.loss_cfg = loss_cfg 

        self.multi_losses = nn.ModuleDict()
        if self.loss_cfg is not None:  
            for item_loc in loss_cfg:
                loss_name = item_loc.name
                
                for item_loss in item_loc.methods:
                    self.multi_losses[loss_name] = MODELS.build(item_loss)
        else:
            self.loss_module = MODELS.build(loss)
            
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        ps = 2
        self.ps = nn.PixelShuffle(ps)
        self.conv_dec = ConvModule(
            in_channels // ps**2,
            in_channels // 4,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))

        self.final_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))
        self.final_layer2 = ConvModule(
            in_channels // ps + in_channels // 4,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU'))

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'] // 2, bias=False))

        self.mlp2 = nn.Sequential(
            ScaleNorm(flatten_dims * ps**2),
            nn.Linear(
                flatten_dims * ps**2, gau_cfg['hidden_dims'] // 2, bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
        
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)
        
        if with_rle:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_channels, out_channels * 2)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        # enc_b  n / 2, h, w
        # enc_t  n,     h, w
        enc_b, enc_t = feats

        feats_t = self.final_layer(enc_t)
        feats_t = torch.flatten(feats_t, 2)
        feats_t = self.mlp(feats_t)

        dec_t = self.ps(enc_t)
        dec_t = self.conv_dec(dec_t)
        enc_b = torch.cat([dec_t, enc_b], dim=1)

        feats_b = self.final_layer2(enc_b)
        feats_b = torch.flatten(feats_b, 2)
        feats_b = self.mlp2(feats_b)

        feats = torch.cat([feats_t, feats_b], dim=2)

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        
        pred_x_sfm = (self.beta * pred_x).softmax(dim=-1)
        pred_y_sfm = (self.beta * pred_y).softmax(dim=-1)
        
        simc_pred_x = torch.sum(pred_x_sfm * self.linspace_x, dim=-1).unsqueeze(-1)
        simc_pred_y = torch.sum(pred_y_sfm * self.linspace_y, dim=-1).unsqueeze(-1)
        
        simc_pred = torch.cat([simc_pred_x, simc_pred_y], dim=-1)
        
        if self.with_rle: # rle            
            global_feature = self.avg(enc_t).reshape(-1, self.in_channels)
            sigma = self.fc(global_feature).reshape(-1, self.out_channels, 2) 
            
            if torch.onnx.is_in_onnx_export():
                return simc_pred
            else:
                return simc_pred, sigma, pred_x, pred_y
            
        return simc_pred, torch.zeros(1, 1), pred_x, pred_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords, _batch_sigmas, _, _ = self.forward(_feats)

            _batch_coords_flip, _batch_sigmas_flip, _, _ = self.forward(
                _feats_flip)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords, batch_sigmas, _, _ = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        preds = self.decode(batch_coords)

        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        coords, sigmas, pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
                         dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
                         dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)
        
        # calculate losses
        losses = dict()
        if self.loss_fun is not None:        
            loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)
            losses.update(loss_kpt=loss)
        
        if self.loss_cfg is not None:
            gt = torch.cat([
                d.gt_instance_labels.keypoint_labels for d in batch_data_samples
            ],
                            dim=0)
            
            all_keys = self.multi_losses.keys()
    
            if 'loss_l1' in all_keys:
                loss_name = 'loss_l1'
                losses[loss_name] = self.multi_losses[loss_name](coords, gt, keypoint_weights)
            
            if 'loss_adl' in all_keys:
                loss_name = 'loss_adl'
                losses[loss_name] = self.multi_losses[loss_name](coords, gt, keypoint_weights)
            
            if 'loss_rle' in all_keys:
                loss_name = 'loss_rle'
                losses[loss_name] = self.multi_losses[loss_name](coords, sigmas, gt, keypoint_weights)
            
            if 'loss_simc' in all_keys:
                loss_name = 'loss_simc'
                losses[loss_name] = self.multi_losses[loss_name](pred_simcc, gt_simcc, keypoint_weights)

        # calculate accuracy
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred=to_numpy(coords),
            gt=to_numpy(gt),
            mask=to_numpy(keypoint_weights) > 0,
            thr=0.05,
            norm_factor=np.ones((coords.size(0), 2), dtype=np.float32))

        acc_pose = torch.tensor(avg_acc, device=gt.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
