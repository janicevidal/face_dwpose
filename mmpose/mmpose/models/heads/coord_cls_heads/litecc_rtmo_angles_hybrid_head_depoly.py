# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmcv.cnn import Scale
from mmengine.structures import InstanceData
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
class MultiTaskHybridLiteMLEHead(BaseHead):
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
        in_channels: int,
        out_channels: int,
        hidden_dims:int,
        euler_in_channels: int,
        euler_bin: Tuple[int, int],
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        euler_bin_interval: Tuple[int, int, int, int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        loss: ConfigType = dict(type='MLECCLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        with_sihn: bool = False,
        with_sihn_relu: bool = False,
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
        self.euler_in_channels = euler_in_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio
        self.with_sihn = with_sihn
        self.with_sihn_relu = with_sihn_relu
        self.with_debias = with_debias
        self.norm_debias = norm_debias
        self.scale_norm = scale_norm
        self.beta = beta
        self.euler_bin = euler_bin
        self.euler_bin_interval = euler_bin_interval

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
        
        self.merged_linear = nn.Linear(hidden_dims, W + H, bias=False)
        
        # mnn convert fc to conv
        # self.cls_x_conv_ = nn.Conv2d(hidden_dims, W, kernel_size=1, bias=False)
        # self.cls_y_conv_ = nn.Conv2d(hidden_dims, H, kernel_size=1, bias=False)
        # self.cls_x_conv_.weight.data = self.cls_x.weight.data.view(W, hidden_dims, 1, 1)
        # self.cls_y_conv_.weight.data = self.cls_y.weight.data.view(H, hidden_dims, 1, 1)
        # self.cls_x_conv_.requires_grad_(False)
        # self.cls_y_conv_.requires_grad_(False)
        
        self.linspace_x = torch.arange(0.0, 1.0 * W, 1).reshape(1, 1, W) / W
        self.linspace_y = torch.arange(0.0, 1.0 * H, 1).reshape(1, 1, H) / H

        self.linspace_x = nn.Parameter(self.linspace_x, requires_grad=False)
        self.linspace_y = nn.Parameter(self.linspace_y, requires_grad=False)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigma_fc = nn.Sequential(
            nn.Linear(in_channels, out_channels), nn.Sigmoid(),
            Scale(0.1))
        
        # euler MLP
        self.yaw_fc = nn.Linear(euler_in_channels, euler_bin[0]*2 // euler_bin_interval[0])
        self.pitch_fc = nn.Linear(euler_in_channels, euler_bin[1]*2 // euler_bin_interval[0])
        
        # mnn convert fc to conv
        # self.yaw_conv_ = nn.Conv2d(euler_in_channels, euler_bin[0]*2 // euler_bin_interval[0], kernel_size=1, stride=1)
        # self.pitch_conv_ = nn.Conv2d(euler_in_channels, euler_bin[1]*2 // euler_bin_interval[0], kernel_size=1, stride=1)
        # self.yaw_conv_.weight.data = self.yaw_fc.weight.data.view(euler_bin[0]*2 // euler_bin_interval[0], euler_in_channels, 1, 1)
        # self.yaw_conv_.bias.data = self.yaw_fc.bias.data
        # self.pitch_conv_.weight.data = self.pitch_fc.weight.data.view(euler_bin[1]*2 // euler_bin_interval[0], euler_in_channels, 1, 1)
        # self.pitch_conv_.bias.data = self.pitch_fc.bias.data
        # self.yaw_conv_.requires_grad_(False)
        # self.pitch_conv_.requires_grad_(False)
        
        self.linspace_yaw = torch.arange(-euler_bin[0] * 1.0, euler_bin[0]* 1.0, euler_bin_interval[0]).reshape(1, euler_bin[0]*2 // euler_bin_interval[0])
        self.linspace_pitch = torch.arange(-euler_bin[1]* 1.0, euler_bin[1]* 1.0, euler_bin_interval[0]).reshape(1, euler_bin[1]*2 // euler_bin_interval[0])
        self.linspace_yaw = nn.Parameter(self.linspace_yaw, requires_grad=False)
        self.linspace_pitch = nn.Parameter(self.linspace_pitch, requires_grad=False)
        
        self.yaw_fc_1 = nn.Linear(euler_in_channels, euler_bin[0]*2 // euler_bin_interval[1])
        self.pitch_fc_1 = nn.Linear(euler_in_channels, euler_bin[1]*2 // euler_bin_interval[1])
        
        self.yaw_fc_2 = nn.Linear(euler_in_channels, euler_bin[0]*2 // euler_bin_interval[2])
        self.pitch_fc_2 = nn.Linear(euler_in_channels, euler_bin[1]*2 // euler_bin_interval[2])
        
        self.yaw_fc_3 = nn.Linear(euler_in_channels, euler_bin[0]*2 // euler_bin_interval[3])
        self.pitch_fc_3 = nn.Linear(euler_in_channels, euler_bin[1]*2 // euler_bin_interval[3])
        
        self.yaw_fc_4 = nn.Linear(euler_in_channels, 2)
        self.pitch_fc_4 = nn.Linear(euler_in_channels, 2)

    def _linear_expectation(self, heatmaps: Tensor,
                            linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""

        B, N, hidden = heatmaps.shape 
        heatmaps_temp = heatmaps.mul(linspace)

        expectation = torch.sum(heatmaps_temp, dim=2, keepdim=True)
        
        return expectation
    
    def _linear_expectation_euler(self, eulers: Tensor,
                            linspace: Tensor) -> Tensor:
        """Calculate linear expectation."""
        
        eulers = F.softmax(eulers, dim=1)
 
        eulers_temp = eulers.mul(linspace)

        expectation = torch.sum(eulers_temp, dim=1, keepdim=True)
        
        return expectation
    
    def _flat_softmax_sihn(self, featmaps: Tensor, eps=1e-5) -> Tensor:
        """Use Softmax to normalize the featmaps in depthwise."""
        self.eps = eps
        
        min_feat, _ = torch.min(featmaps, dim=2, keepdim=True)
        featmaps_input = featmaps - min_feat
        heatmaps = featmaps_input / (featmaps_input.sum(dim=2, keepdim=True)).clamp(min=self.eps)
        
        return heatmaps
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
    
       
    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        euler_feature = self.avg(feats[0]).reshape(-1, self.euler_in_channels)
        
        # multitask: euler angles
        # if torch.onnx.is_in_onnx_export():        
        #     euler_feature_conv = self.avg(feats[0])
        #     yaw = self.yaw_conv_(euler_feature_conv).reshape(1, self.yaw_fc.out_features)
        #     pitch = self.pitch_conv_(euler_feature_conv).reshape(1, self.pitch_fc.out_features)
        # else:
        #     yaw = self.yaw_fc(euler_feature)  # -> B, yaw_bins
        #     pitch = self.pitch_fc(euler_feature)  # -> B, pitch_bins
        
        yaw = self.yaw_fc(euler_feature)  # -> B, yaw_bins
        pitch = self.pitch_fc(euler_feature)  # -> B, pitch_bins
            
        pred_yaw = self._linear_expectation_euler(yaw, self.linspace_yaw)
        pred_pitch = self._linear_expectation_euler(pitch, self.linspace_pitch)
        
        # multitask: keypoint coordinates
        x = feats[-1]
        feats_ = self.final_layer(x)  # -> B, K, H, W
        
        feats_ = feats_.reshape(feats_.size(0), feats_.size(1), self.flatten_dims)
        
        if self.scale_norm:
            feats_ = self.norm(feats_)
        
        feats_ = self.mlp(feats_)  # -> B, K, hidden
        
        batch_size, seq_len, _ = feats_.shape
        
        weight_x = self.cls_x.weight
        weight_y = self.cls_y.weight
        weight_x_T = weight_x.T
        weight_y_T = weight_y.T
        merged_weight_T = torch.cat([weight_x_T, weight_y_T], dim=1)
        merged_weight = merged_weight_T.T
            
        self.merged_linear.weight.data  = nn.Parameter(merged_weight, requires_grad=False)
        self.merged_linear.requires_grad_(False)
        
        merged = self.merged_linear(feats_)
        merged_4d = merged.reshape(batch_size, seq_len, 2, 144)
        
        permuted = merged_4d.permute(0, 2, 1, 3)
        
        pred = permuted.reshape(batch_size, seq_len * 2, 144)
        
        if self.with_sihn:
            pred_sfm = self._flat_softmax_sihn(pred * self.beta)
        elif self.with_sihn_relu:
            # pred_sfm = F.relu(pred * self.beta)
            pred_sfm = F.relu(pred)
            pred_sfm = pred_sfm / (pred_sfm.sum(dim=2, keepdim=True)).clamp(min=1e-5)
        else:
            pred_sfm = self._flat_softmax(pred * self.beta)
        
        simc_pred = self._linear_expectation(pred_sfm, self.linspace_x)
        
        # simc_pred = torch.cat([simc_pred[:, :seq_len, :], simc_pred[:, seq_len:, :]], dim=-1) 
        angle_pred = torch.cat([pred_yaw, pred_pitch], dim=-1)
        
        return simc_pred, angle_pred


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

            _batch_coords, _,  _, _, _, _,  _, _, _, _,  _, _, _, _, pred_yaw, pred_pitch = self.forward(_feats)

            _batch_coords_flip, _,  _, _, _, _,  _, _, _, _, _, _, _, _, _, _ = self.forward(
                _feats_flip)
            _batch_coords_flip = flip_coordinates(
                _batch_coords_flip,
                flip_indices=flip_indices,
                shift_coords=test_cfg.get('shift_coords', True),
                input_size=input_size)

            batch_coords = (_batch_coords + _batch_coords_flip) * 0.5
        else:
            batch_coords, _,  _, _, _, _,  _, _, _, _, _, _, _, _, pred_yaw, pred_pitch = self.forward(feats)  # (B, K, D)

        batch_coords.unsqueeze_(dim=1)  # (B, N, K, D)
        batch_coords1 = batch_coords[:, :, :, :2]
        preds = self.decode(batch_coords1, pred_yaw, pred_pitch)

        return preds
    
    def decode(self, batch_outputs: Tensor, batch_yaws: Tensor,batch_pitchs: Tensor) -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
                a data batch

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
            decoded pose information of the instances of one data sample.
        """

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args, )
            return func(*args)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode)

        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_yaw_np = to_numpy(batch_yaws, unzip=True)
            batch_pitch_np = to_numpy(batch_pitchs, unzip=True)
            
            batch_keypoints = []
            batch_scores = []
            batch_euler_yaws = []
            batch_euler_pitchs = []
            
            for (outputs, yaws, pitchs) in zip(batch_output_np, batch_yaw_np, batch_pitch_np):
                keypoints, scores = _pack_and_call(outputs, self.decoder.decode)
                batch_keypoints.append(keypoints)
                batch_scores.append(scores)
                batch_euler_yaws.append(yaws)
                batch_euler_pitchs.append(pitchs)

        preds = [
            InstanceData(keypoints=keypoints, keypoint_scores=scores, euler_yaws=euler_yaws, euler_pitchs=euler_pitchs)
            for keypoints, scores, euler_yaws, euler_pitchs  in zip(batch_keypoints, batch_scores, batch_euler_yaws, batch_euler_pitchs)
        ]

        return preds


    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        
        coords, sigmas, pred_x, pred_y, yaw, pitch, yaw_1, pitch_1, yaw_2, pitch_2, yaw_3, pitch_3, yaw_4, pitch_4, pred_yaw, pred_pitch = self.forward(feats)
        
        gt_euler_labels = torch.cat([
                d.gt_instance_labels.euler_labels for d in batch_data_samples
            ],
                            dim=0)
        
        gt_euler_angles = torch.cat([
                d.gt_instance_labels.euler_angles for d in batch_data_samples
            ],
                            dim=0)
                
        
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
            
            if 'loss_angle_ce' in all_keys:
                loss_name = 'loss_angle_ce'
                losses['loss_ce_yaw'] = self.multi_losses[loss_name](yaw, gt_euler_labels[:, :, 0][:, 0]) \
                                            + 0.8 * self.multi_losses[loss_name](yaw_1, gt_euler_labels[:, :, 1][:, 0]) \
                                                + 0.5* self.multi_losses[loss_name](yaw_2, gt_euler_labels[:, :, 2][:, 0]) \
                                                    + 0.2* self.multi_losses[loss_name](yaw_3, gt_euler_labels[:, :, 3][:, 0])\
                                                        + 0.2* self.multi_losses[loss_name](yaw_4, gt_euler_labels[:, :, 4][:, 0])    
                losses['loss_ce_pitch'] = self.multi_losses[loss_name](pitch, gt_euler_labels[:, :, 0][:, -1]) \
                                            + 0.8 * self.multi_losses[loss_name](pitch_1, gt_euler_labels[:, :, 1][:, -1]) \
                                                + 0.5* self.multi_losses[loss_name](pitch_2, gt_euler_labels[:, :, 2][:, -1]) \
                                                    + 0.2 * self.multi_losses[loss_name](pitch_3, gt_euler_labels[:, :, 3][:, -1])\
                                                        + 0.2* self.multi_losses[loss_name](pitch_4, gt_euler_labels[:, :, 4][:, -1])
            
            if 'loss_angle_mse' in all_keys:
                loss_name = 'loss_angle_mse'
                losses['loss_mse_yaw'] = self.multi_losses[loss_name](pred_yaw, gt_euler_angles[:, 0:1])
                losses['loss_mse_pitch'] = self.multi_losses[loss_name](pred_pitch, gt_euler_angles[:, -1:])

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
