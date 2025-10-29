# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmcv.cnn import Scale
from mmpose.models.backbones.blocks import OREPA_1x1
from mmpose.evaluation.functional import simcc_pck_accuracy, keypoint_pck_accuracy
from mmpose.models.utils.rtmcc_block import ScaleNorm
from mmpose.models.utils.tta import flip_coordinates
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class LiteMLEHead(BaseHead):
    """
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
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
        out_sigma (bool): Predict the sigma (the viriance of the joint
            location) together with the joint location. Default: False
    """    
    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        hidden_dims:int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        loss: ConfigType = dict(type='MLECCLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        with_debias: bool = False,
        norm_debias: bool = False,
        scale_norm: bool = False,
        rep_conv1x1: bool = False,
        beta: float = 1.0,
        loss_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.with_debias = with_debias
        self.norm_debias = norm_debias
        self.scale_norm = scale_norm
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
        self.flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        
        if rep_conv1x1:
            self.final_layer = OREPA_1x1(in_channels, out_channels, kernel_size=1, deploy=False)
        else:
            self.final_layer = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=final_layer_kernel_size,
                stride=1,
                padding=final_layer_kernel_size // 2)

        if scale_norm:
            self.norm = ScaleNorm(self.flatten_dims)

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        
        if hidden_dims == self.flatten_dims:
            self.mlp = nn.Identity()
        else:
            self.mlp = nn.Linear(self.flatten_dims, hidden_dims, bias=False)
        
        self.cls_x = nn.Linear(hidden_dims , W, bias=False)
        self.cls_y = nn.Linear(hidden_dims , H, bias=False)
        
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, W) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigma_fc = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Sigmoid(),
            Scale(0.1))

    def _linear_expectation(self, heatmaps: Tensor,
                            linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""

        B, N, hidden = heatmaps.shape 
        heatmaps_temp = heatmaps.mul(linspace)

        expectation = torch.sum(heatmaps_temp, dim=2, keepdim=True)
        
        return expectation
    
    
    def _flat_softmax(self, featmaps: Tensor) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""
        
        featmaps = featmaps.clamp(min=-5e4, max=5e4)
        heatmaps = F.softmax(featmaps, dim=2)

        return heatmaps
        
        
    def generate_target_heatmap(self, kpt_targets, sigmas):
        """Generate target heatmaps for keypoints.

        This function calculates x and y bins. It then computes distances from keypoint targets to these
        bins and normalizes these distances based on the sigmas.
        Finally, it uses these distances to generate heatmaps for x and y
        coordinates under assumption of laplacian error.

        Args:
            kpt_targets (Tensor): Keypoint targets tensor.
            sigmas (Tensor): Learned deviation of grids.

        Returns:
            tuple: A tuple containing the x and y heatmaps.
        """

        # calculate the error of each bin from the GT keypoint coordinates
        dist_x = torch.abs(kpt_targets.narrow(2, 0, 1) - self.linspace_x)
        dist_y = torch.abs(kpt_targets.narrow(2, 1, 1) - self.linspace_y)

        # normalize
        sigmas_  = sigmas.clip(min=1e-3).unsqueeze(2)

        dist_x = dist_x / sigmas_
        dist_y = dist_y / sigmas_

        hm_x = torch.exp(-dist_x / 2) / sigmas_
        hm_y = torch.exp(-dist_y / 2) / sigmas_

        return hm_x, hm_y 
    
       
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
        x = feats[-1]
        feats = self.final_layer(x)  # -> B, K, H, W
        
        feats = feats.reshape(feats.size(0), feats.size(1), self.flatten_dims)
        
        if self.scale_norm:
            feats = self.norm(feats)
        
        feats = self.mlp(feats)  # -> B, K, hidden
        
        # if self.scale_norm:
        #     feats = self.norm(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)
        
        pred_x_sfm = self._flat_softmax(pred_x * self.beta)
        pred_y_sfm = self._flat_softmax(pred_y * self.beta)
            
        global_feature = self.avg(x).reshape(-1, self.in_channels)
        sigmas = self.sigma_fc(global_feature)
        
        simc_pred_x = self._linear_expectation(pred_x_sfm, self.linspace_x)
        simc_pred_y = self._linear_expectation(pred_y_sfm, self.linspace_y)
        
        # https://zhuanlan.zhihu.com/p/563022818
        if self.with_debias:
            if self.norm_debias:
                C_x = pred_x_sfm.exp().sum(dim=2).unsqueeze(-1)
                C_y = pred_x_sfm.exp().sum(dim=2).unsqueeze(-1)
            else:
                C_x = pred_x.exp().sum(dim=2).unsqueeze(-1)
                C_y = pred_y.exp().sum(dim=2).unsqueeze(-1)
                
            simc_pred_x = C_x / (C_x - 1) * (simc_pred_x - 1 / (2 * C_x))
            simc_pred_y = C_y / (C_y - 1) * (simc_pred_y - 1 / (2 * C_y))
        
        if torch.onnx.is_in_onnx_export():
            simc_pred = torch.cat([simc_pred_x, simc_pred_y], dim=-1)
            return simc_pred
        else:
            simc_pred = torch.cat([simc_pred_x, simc_pred_y], dim=-1)
            
            return simc_pred, sigmas, pred_x_sfm, pred_y_sfm


    def predict(self,
                feats: Tuple[Tensor],
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']
            _feats, _feats_flip = feats

            _batch_coords, _,  _, _ = self.forward(_feats)

            _batch_coords_flip, _, _, _ = self.forward(
                _feats_flip)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords, _, _, _ = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        batch_coords1 = batch_coords[:, :, :, :2]
        preds = self.decode(batch_coords1)

        return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        
        coords, sigmas, pred_x, pred_y = self.forward(feats)
        
        gt = torch.cat([
                d.gt_instance_labels.keypoint_labels for d in batch_data_samples
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
        
        target_hms = self.generate_target_heatmap(gt, sigmas)
        
        sigma = sigmas.unsqueeze(-1).repeat(1, 1, 2)

        # calculate losses
        losses = dict()
        
        if self.loss_cfg is not None:

            
            all_keys = self.multi_losses.keys()
    
            if 'loss_rle' in all_keys:
                loss_name = 'loss_rle'
                losses[loss_name] = self.multi_losses[loss_name](coords, sigma, gt, keypoint_weights)
            
            if 'loss_adl' in all_keys:
                loss_name = 'loss_adl'
                losses[loss_name] = self.multi_losses[loss_name](coords, gt)
            
            if 'loss_mle' in all_keys:
                loss_name = 'loss_mle'
                losses[loss_name] = self.multi_losses[loss_name](pred_simcc, target_hms, keypoint_weights)

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
