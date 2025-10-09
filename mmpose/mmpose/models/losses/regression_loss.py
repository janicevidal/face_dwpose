# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import dsntnn
import numpy as np

from mmpose.registry import MODELS
from ..utils.realnvp import RealNVP
from .classification_loss import KLDiscretLoss


@MODELS.register_module()
class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_distribution='laplace'):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_distribution = q_distribution

        self.flow_model = RealNVP()

    def forward(self, pred, sigma, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (Tensor[N, K, D]): Output regression.
            sigma (Tensor[N, K, D]): Output sigma.
            target (Tensor[N, K, D]): Target regression.
            target_weight (Tensor[N, K, D]):
                Weights across different joint types.
        """
        sigma = sigma.sigmoid()

        error = (pred - target) / (sigma + 1e-9)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


@MODELS.register_module()
class SmoothL1Loss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """

        if self.use_target_weight:
            assert target_weight is not None
            assert output.ndim >= target_weight.ndim

            for i in range(output.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class SoftWeightSmoothL1Loss(nn.Module):
    """Smooth L1 loss with soft weight for regression.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        supervise_empty (bool): Whether to supervise the output with zero
            weight.
        beta (float):  Specifies the threshold at which to change between
            L1 and L2 loss.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 use_target_weight=False,
                 supervise_empty=True,
                 beta=1.0,
                 loss_weight=1.):
        super().__init__()

        reduction = 'none' if use_target_weight else 'mean'
        self.criterion = partial(
            self.smooth_l1_loss, reduction=reduction, beta=beta)

        self.supervise_empty = supervise_empty
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    @staticmethod
    def smooth_l1_loss(input, target, reduction='none', beta=1.0):
        """Re-implement torch.nn.functional.smooth_l1_loss with beta to support
        pytorch <= 1.6."""
        delta = input - target
        mask = delta.abs() < beta
        delta[mask] = (delta[mask]).pow(2) / (2 * beta)
        delta[~mask] = delta[~mask].abs() - beta / 2

        if reduction == 'mean':
            return delta.mean()
        elif reduction == 'sum':
            return delta.sum()
        elif reduction == 'none':
            return delta
        else:
            raise ValueError(f'reduction must be \'mean\', \'sum\' or '
                             f'\'none\', but got \'{reduction}\'')

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            assert output.ndim >= target_weight.ndim

            for i in range(output.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)

            loss = self.criterion(output, target) * target_weight
            if self.supervise_empty:
                loss = loss.mean()
            else:
                num_elements = torch.nonzero(target_weight > 0).size()[0]
                loss = loss.sum() / max(num_elements, 1.0)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class WingLoss(nn.Module):
    """Wing Loss. paper ref: 'Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks' Feng et al. CVPR'2018.

    Args:
        omega (float): Also referred to as width.
        epsilon (float): Also referred to as curvature.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 omega=10.0,
                 epsilon=2.0,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        # constant that smoothly links the piecewise-defined linear
        # and nonlinear parts
        self.C = self.omega * (1.0 - math.log(1.0 + self.omega / self.epsilon))

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta / self.epsilon), delta - self.C)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class SoftWingLoss(nn.Module):
    """Soft Wing Loss 'Structure-Coherent Deep Feature Learning for Robust Face
    Alignment' Lin et al. TIP'2021.

    loss =
        1. |x|                           , if |x| < omega1
        2. omega2*ln(1+|x|/epsilon) + B, if |x| >= omega1

    Args:
        omega1 (float): The first threshold.
        omega2 (float): The second threshold.
        epsilon (float): Also referred to as curvature.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 omega1=2.0,
                 omega2=20.0,
                 epsilon=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.omega1 = omega1
        self.omega2 = omega2
        self.epsilon = epsilon
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        # constant that smoothly links the piecewise-defined linear
        # and nonlinear parts
        self.B = self.omega1 - self.omega2 * math.log(1.0 + self.omega1 /
                                                      self.epsilon)

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K
            dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
        """
        delta = (target - pred).abs()
        losses = torch.where(
            delta < self.omega1, delta,
            self.omega2 * torch.log(1.0 + delta / self.epsilon) + self.B)
        return torch.mean(torch.sum(losses, dim=[1, 2]), dim=0)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K
            dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class MPJPELoss(nn.Module):
    """MPJPE (Mean Per Joint Position Error) loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N,K,D]):
                Weights across different joint types.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = torch.mean(
                torch.norm((output - target) * target_weight, dim=-1))
        else:
            loss = torch.mean(torch.norm(output - target, dim=-1))

        return loss * self.loss_weight


@MODELS.register_module()
class L1Loss(nn.Module):
    """L1Loss loss ."""

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.l1_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class MSELoss(nn.Module):
    """MSE loss for coordinate regression."""

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.criterion = F.mse_loss
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 2]): Output regression.
            target (torch.Tensor[N, K, 2]): Target regression.
            target_weight (torch.Tensor[N, K, 2]):
                Weights across different joint types.
        """

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target * target_weight)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class BoneLoss(nn.Module):
    """Bone length loss.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        use_target_weight (bool): Option to use weighted bone loss.
            Different bone types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, joint_parents, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.joint_parents = joint_parents
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

        self.non_root_indices = []
        for i in range(len(self.joint_parents)):
            if i != self.joint_parents[i]:
                self.non_root_indices.append(i)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K-1]):
                Weights across different bone types.
        """
        output_bone = torch.norm(
            output - output[:, self.joint_parents, :],
            dim=-1)[:, self.non_root_indices]
        target_bone = torch.norm(
            target - target[:, self.joint_parents, :],
            dim=-1)[:, self.non_root_indices]
        if self.use_target_weight:
            assert target_weight is not None
            loss = torch.mean(
                torch.abs((output_bone * target_weight).mean(dim=0) -
                          (target_bone * target_weight).mean(dim=0)))
        else:
            loss = torch.mean(
                torch.abs(output_bone.mean(dim=0) - target_bone.mean(dim=0)))

        return loss * self.loss_weight


@MODELS.register_module()
class SemiSupervisionLoss(nn.Module):
    """Semi-supervision loss for unlabeled data. It is composed of projection
    loss and bone loss.

    Paper ref: `3D human pose estimation in video with temporal convolutions
    and semi-supervised training` Dario Pavllo et al. CVPR'2019.

    Args:
        joint_parents (list): Indices of each joint's parent joint.
        projection_loss_weight (float): Weight for projection loss.
        bone_loss_weight (float): Weight for bone loss.
        warmup_iterations (int): Number of warmup iterations. In the first
            `warmup_iterations` iterations, the model is trained only on
            labeled data, and semi-supervision loss will be 0.
            This is a workaround since currently we cannot access
            epoch number in loss functions. Note that the iteration number in
            an epoch can be changed due to different GPU numbers in multi-GPU
            settings. So please set this parameter carefully.
            warmup_iterations = dataset_size // samples_per_gpu // gpu_num
            * warmup_epochs
    """

    def __init__(self,
                 joint_parents,
                 projection_loss_weight=1.,
                 bone_loss_weight=1.,
                 warmup_iterations=0):
        super().__init__()
        self.criterion_projection = MPJPELoss(
            loss_weight=projection_loss_weight)
        self.criterion_bone = BoneLoss(
            joint_parents, loss_weight=bone_loss_weight)
        self.warmup_iterations = warmup_iterations
        self.num_iterations = 0

    @staticmethod
    def project_joints(x, intrinsics):
        """Project 3D joint coordinates to 2D image plane using camera
        intrinsic parameters.

        Args:
            x (torch.Tensor[N, K, 3]): 3D joint coordinates.
            intrinsics (torch.Tensor[N, 4] | torch.Tensor[N, 9]): Camera
                intrinsics: f (2), c (2), k (3), p (2).
        """
        while intrinsics.dim() < x.dim():
            intrinsics.unsqueeze_(1)
        f = intrinsics[..., :2]
        c = intrinsics[..., 2:4]
        _x = torch.clamp(x[:, :, :2] / x[:, :, 2:], -1, 1)
        if intrinsics.shape[-1] == 9:
            k = intrinsics[..., 4:7]
            p = intrinsics[..., 7:9]

            r2 = torch.sum(_x[:, :, :2]**2, dim=-1, keepdim=True)
            radial = 1 + torch.sum(
                k * torch.cat((r2, r2**2, r2**3), dim=-1),
                dim=-1,
                keepdim=True)
            tan = torch.sum(p * _x, dim=-1, keepdim=True)
            _x = _x * (radial + tan) + p * r2
        _x = f * _x + c
        return _x

    def forward(self, output, target):
        losses = dict()

        self.num_iterations += 1
        if self.num_iterations <= self.warmup_iterations:
            return losses

        labeled_pose = output['labeled_pose']
        unlabeled_pose = output['unlabeled_pose']
        unlabeled_traj = output['unlabeled_traj']
        unlabeled_target_2d = target['unlabeled_target_2d']
        intrinsics = target['intrinsics']

        # projection loss
        unlabeled_output = unlabeled_pose + unlabeled_traj
        unlabeled_output_2d = self.project_joints(unlabeled_output, intrinsics)
        loss_proj = self.criterion_projection(unlabeled_output_2d,
                                              unlabeled_target_2d, None)
        losses['proj_loss'] = loss_proj

        # bone loss
        loss_bone = self.criterion_bone(unlabeled_pose, labeled_pose, None)
        losses['bone_loss'] = loss_bone

        return losses


@MODELS.register_module()
class DSNTLoss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, sigma = 1, mse_weight=1,js_weight = 1, is_dsnt = True):
        super().__init__()
        
        self.mse = dsntnn.euclidean_losses
        self.js = dsntnn.js_reg_losses
        self.avg = dsntnn.average_loss

        self.use_target_weight = use_target_weight
        self.mse_weight = mse_weight
        self.js_weight = js_weight
        self.sigma = sigma

    def forward(self, output, target, heatmap, target_weight=None): # 预测坐标，gt坐标，预测热图， (n,num_joints,h,w)
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weight is not None
            print(output.size())
            print(target_weight.size())
            print(target.size())
            print(heatmap.size())
            mse_loss = self.mse(output * target_weight,
                                  target * target_weight)
        else:
            mse_loss = self.mse(output, target)
        js_loss = self.js(heatmap, target, sigma_t=self.sigma)

        loss = self.avg(self.mse_weight*mse_loss + self.js_weight*js_loss)

        return loss
    
    
@MODELS.register_module()
class SimCC_DSNTRLE_Loss(nn.Module):
    """
        rle + dsnt + simcc
    """

    def __init__(self, dsnt_param, rle_param, simc_param, dsnt_weight, rle_weight, simc_weight):
        super().__init__()
        
        dsnt_use_target_weight = getattr(dsnt_param, 'use_target_weight', True)
        sigma = getattr(dsnt_param, 'sigma', 0.25)
        mse_weight = getattr(dsnt_param, 'mse_weight' ,1)
        js_weight = getattr(dsnt_param, 'js_weight' ,1)
        self.dsnt = DSNTLoss(dsnt_use_target_weight, sigma, mse_weight, js_weight)

        rle_use_target_weight = getattr(rle_param, 'use_target_weight', True)
        size_average = getattr(rle_param, 'size_average', True)
        residual = getattr(rle_param, 'residual', True)
        self.rle = RLELoss(rle_use_target_weight, size_average, residual)
        
        simc_use_target_weight = getattr(simc_param, 'use_target_weight', True)
        beta = getattr(simc_param, 'beta', 1.0)
        label_softmax = getattr(simc_param, 'label_softmax', False)
        self.simc = KLDiscretLoss(beta=beta, label_softmax=label_softmax, use_target_weight=simc_use_target_weight)
    
        self.dw = dsnt_weight
        self.re = rle_weight
        self.sw = simc_weight

    def forward(self, pred_simcc, coord, heatmap, gt_simcc, gt, target_weight):
      
        dsnt_loss = self.dsnt(coord[...,:2], gt, heatmap, target_weight)
        rle_loss = self.rle(coord, gt, target_weight)
        cimc_loss = self.simc(pred_simcc, gt_simcc, target_weight)

        loss = self.dw * dsnt_loss + rle_loss * self.re + cimc_loss * self.sw

        return loss


@MODELS.register_module()
class AnisotropicDirectionLoss(nn.Module):
    def __init__(self, scale=0.01, loss_lambda=2.0, lambda_mode=1, loss_weight=1):
        super(AnisotropicDirectionLoss, self).__init__()
        self.max_node_number = 1000
        self.scale = scale
        self.loss_lambda = loss_lambda
        self.lambda_mode = lambda_mode
        self.edge_info = (
                (False, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)), # FaceContour
                (True, (37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56)), # 左眼眉毛
                (True, (57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76)), # 右眼眉毛
                (True, (77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)), # LeftEyebrow
                (True, (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124)), # RightEyebrow
                (True, (125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137)), # Nose
                (False, (138, 139, 140)), # NoseLine
                (True, (141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176)), # OuterLip
                (True, (177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200)), # InnerLip
                # (False, (201)), # 左眼瞳孔
                # (False, (202)), # 右眼瞳孔
                (True, (203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218)), # LeftEye
                (True, (219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234)), # RightEye
            )
        self.neighbors = self._get_neighbors(self.edge_info)
        self.bins = list()
        self.max_bins = 1000
        self.loss_weight = loss_weight

    def __repr__(self):
        return "AnisotropicDirectionLoss()"

    def _get_neighbors(self, edge_info):
        neighbors = np.arange(self.max_node_number)[:,np.newaxis].repeat(3, axis=1)
        for is_closed, indices in edge_info:
            n = len(indices)
            for i in range(n):
                cur_id = indices[i]
                pre_id = indices[(i-1)%n]
                nex_id = indices[(i+1)%n]
                if not is_closed:
                    if i == 0:
                        pre_id = nex_id
                    elif i == n-1:
                        nex_id = pre_id
                neighbors[cur_id][0] = cur_id
                neighbors[cur_id][1] = pre_id
                neighbors[cur_id][2] = nex_id
        return neighbors

    def _inverse_vector(self, vector):
        """
        input: b x n x 2
        output: b x n x 2
        """
        inversed_vector = torch.stack((-vector[:,:,1], vector[:,:,0]), dim=-1)
        return inversed_vector

    def _get_normals_from_neighbors(self, landmarks):
        # input: b x n x 2
        # output: # b x n x 2
        point_num = landmarks.shape[1]
        itself = self.neighbors[0:point_num, 0]
        previous_neighbors = self.neighbors[0:point_num, 1]
        next_neighbors = self.neighbors[0:point_num, 2]
    
        # condition 1
        bi_normal_vector = F.normalize(landmarks[:, previous_neighbors] - landmarks[:, itself], p=2, dim=-1) + \
                           F.normalize(landmarks[:, next_neighbors] - landmarks[:, itself], p=2, dim=-1)
        # condition 2
        previous_tangent_vector = landmarks[:, previous_neighbors] - landmarks[:, itself]
        next_tangent_vector = landmarks[:, next_neighbors] - landmarks[:, itself]
    
        normal_vector = torch.where(previous_tangent_vector == next_tangent_vector, self._inverse_vector(previous_tangent_vector), bi_normal_vector)
    
        normal_vector = F.normalize(normal_vector, p=2, dim=-1)
        return normal_vector

    def _get_loss_lambda(self, pv_gt, normal_force, tangent_force, normal_vector, tangent_vector, lambda_mode=2):
        # fix
        if lambda_mode == 1:
            # 1
            loss_lambda = self.loss_lambda
        # dynamic
        elif lambda_mode == 2:
            loss_lambda = torch.clamp(tangent_force.pow(2) / torch.clamp(normal_force.pow(2), min=1e-6), min=1.0, max=9.0)
            # b x n
            loss_lambda = loss_lambda.detach()
        # statistic
        elif lambda_mode == 4:
            cur_loss_lambda = tangent_force.pow(2) / torch.clamp(normal_force.pow(2), min=1e-6) # b x n
            self.bins.extend(cur_loss_lambda.tolist()) # (1000 x b) x n
            while len(self.bins) > self.max_bins:
                del self.bins[0]
            loss_lambda = torch.tensor(self.bins).to(pv_gt) # (1000 x b) x n
            loss_lambda = loss_lambda.mean(dim=0, keepdim=True) # 1 x n
            loss_lambda = torch.clamp(loss_lambda, min=1.0, max=9.0)
            # 1 x n
            loss_lambda = loss_lambda.detach()
        # statistic
        elif lambda_mode == 5:
            self.bins.extend(pv_gt.tolist()) # (1000 x b) x n x 2
            while len(self.bins) > self.max_bins:
                del self.bins[0]
            direction = torch.tensor(self.bins).to(pv_gt) # (1000 x b) x n x 2
            dx = direction[:, :, 0] # (1000 x b) x n
            dy = direction[:, :, 1] # (1000 x b) x n
            dx = dx * dy.sign() # (1000 x b) x n
            dy = dy.abs() # (1000 x b) x n
            dx = dx.sum([0]) # n
            dy = dy.sum([0]) # n
            tangent_vector = torch.stack([dx, dy], dim=-1) # n x 2
            tangent_vector = F.normalize(tangent_vector, p=2, dim=-1) # n x 2
            normal_vector = torch.stack((-tangent_vector[:,1], tangent_vector[:,0]), dim=-1) # n x 2

            normal_std2 = torch.mul(direction, normal_vector.unsqueeze(0)).sum(dim=-1).pow(2).sum(dim=0) # n
            tangent_std2 = torch.mul(direction, tangent_vector.unsqueeze(0)).sum(dim=-1).pow(2).sum(dim=0) # n

            loss_lambda = torch.clamp(tangent_std2 / torch.clamp(normal_std2, min=1e-6), min=1.0, max=9.0).unsqueeze(0) # 1 x n
            # 1 x n
            loss_lambda = loss_lambda.detach()
        else:
            assert False
        return loss_lambda


    def forward(self, coord, gt):
        # [0, 1] to [-1, 1]
        groundtruth = gt * 2 - 1
        output = coord * 2 - 1

        normal_vector = self._get_normals_from_neighbors(groundtruth) # b x n x 2, [-1, 1]
        tangent_vector = self._inverse_vector(normal_vector) # b x n x 2, [-1, 1]

        pv_gt = output - groundtruth # b x n x 2, [-1, 1]

        normal_force = torch.mul(pv_gt, normal_vector).sum(dim=-1, keepdim=False) # b x n
        tangent_force = torch.mul(pv_gt, tangent_vector).sum(dim=-1, keepdim=False) # b x n
        
        loss_lambda = self._get_loss_lambda(pv_gt.detach(), normal_force.detach(), tangent_force.detach(), normal_vector.detach(), tangent_vector.detach(), lambda_mode=self.lambda_mode)

        alpha = 2 * loss_lambda / (loss_lambda + 1.0)
        belta = 2 * 1 / (loss_lambda + 1.0)
        delta_2_asy = alpha * normal_force.pow(2) + belta * tangent_force.pow(2) # b x n

        delta_2_sy = pv_gt.pow(2).sum(dim=-1, keepdim=False) # b x n

        delta_2 = torch.where(normal_vector.norm(p=2, dim=-1) < 0.5, delta_2_sy, delta_2_asy)

        delta = delta_2.clamp(min=1e-128).sqrt() # delta_2.sqrt()
        loss = torch.where(delta < self.scale, 0.5 / self.scale * delta_2, delta - 0.5 * self.scale)

        return self.loss_weight * loss.mean()
    

@MODELS.register_module()
class AnisotropicRLELoss(RLELoss):
    """Anisotropic RLE Loss.
    
    结合 RLELoss 的残差对数似然估计和 AnisotropicDirectionLoss 的各向异性方向约束

    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_distribution='laplace',
                 loss_lambda=2.0,
                 lambda_mode=1):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_distribution = q_distribution

        self.flow_model = RealNVP()
        
        # Anisotropic components
        self.edge_info = (
                (False, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)), # FaceContour
                (True, (37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56)), # 左眼眉毛
                (True, (57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76)), # 右眼眉毛
                (True, (77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)), # LeftEyebrow
                (True, (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124)), # RightEyebrow
                (True, (125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137)), # Nose
                (False, (138, 139, 140)), # NoseLine
                (True, (141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176)), # OuterLip
                (True, (177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200)), # InnerLip
                # (False, (201)), # 左眼瞳孔
                # (False, (202)), # 右眼瞳孔
                (True, (203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218)), # LeftEye
                (True, (219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234)), # RightEye
            )
        self.max_node_number = 1000
        self.loss_lambda = loss_lambda
        self.lambda_mode = lambda_mode
        self.neighbors = self._get_neighbors(self.edge_info)
        self.bins = []
        self.max_bins = 1000
        
    def _get_neighbors(self, edge_info):
        """Get neighbor indices for each landmark point."""
        neighbors = np.arange(self.max_node_number)[:,np.newaxis].repeat(3, axis=1)
        for is_closed, indices in edge_info:
            n = len(indices)
            for i in range(n):
                cur_id = indices[i]
                pre_id = indices[(i-1)%n]
                nex_id = indices[(i+1)%n]
                if not is_closed:
                    if i == 0:
                        pre_id = nex_id
                    elif i == n-1:
                        nex_id = pre_id
                neighbors[cur_id][0] = cur_id
                neighbors[cur_id][1] = pre_id
                neighbors[cur_id][2] = nex_id
        return neighbors

    def _inverse_vector(self, vector):
        """Inverse vector by rotating 90 degrees."""
        inversed_vector = torch.stack((-vector[:,:,1], vector[:,:,0]), dim=-1)
        return inversed_vector

    def _get_normals_from_neighbors(self, landmarks):
        """Calculate normal vectors from landmark neighbors."""
        point_num = landmarks.shape[1]
        itself = self.neighbors[0:point_num, 0]
        previous_neighbors = self.neighbors[0:point_num, 1]
        next_neighbors = self.neighbors[0:point_num, 2]
    
        # Calculate bi-normal vector
        bi_normal_vector = F.normalize(landmarks[:, previous_neighbors] - landmarks[:, itself], p=2, dim=-1) + \
                           F.normalize(landmarks[:, next_neighbors] - landmarks[:, itself], p=2, dim=-1)
        
        # Calculate tangent vectors
        previous_tangent_vector = landmarks[:, previous_neighbors] - landmarks[:, itself]
        next_tangent_vector = landmarks[:, next_neighbors] - landmarks[:, itself]
    
        # Handle edge cases
        normal_vector = torch.where(previous_tangent_vector == next_tangent_vector, self._inverse_vector(previous_tangent_vector), bi_normal_vector)
    
        normal_vector = F.normalize(normal_vector, p=2, dim=-1)
        return normal_vector
    
    def _get_loss_lambda(self, pv_gt, normal_force, tangent_force, normal_vector, tangent_vector, lambda_mode=2):
        # fix
        if lambda_mode == 1:
            # 1
            loss_lambda = self.loss_lambda
        # dynamic
        elif lambda_mode == 2:
            loss_lambda = torch.clamp(tangent_force.pow(2) / torch.clamp(normal_force.pow(2), min=1e-6), min=1.0, max=9.0)
            # b x n
            loss_lambda = loss_lambda.detach()
        # statistic
        elif lambda_mode == 4:
            cur_loss_lambda = tangent_force.pow(2) / torch.clamp(normal_force.pow(2), min=1e-6) # b x n
            self.bins.extend(cur_loss_lambda.tolist()) # (1000 x b) x n
            while len(self.bins) > self.max_bins:
                del self.bins[0]
            loss_lambda = torch.tensor(self.bins).to(pv_gt) # (1000 x b) x n
            loss_lambda = loss_lambda.mean(dim=0, keepdim=True) # 1 x n
            loss_lambda = torch.clamp(loss_lambda, min=1.0, max=9.0)
            # 1 x n
            loss_lambda = loss_lambda.detach()
        # statistic
        elif lambda_mode == 5:
            self.bins.extend(pv_gt.tolist()) # (1000 x b) x n x 2
            while len(self.bins) > self.max_bins:
                del self.bins[0]
            direction = torch.tensor(self.bins).to(pv_gt) # (1000 x b) x n x 2
            dx = direction[:, :, 0] # (1000 x b) x n
            dy = direction[:, :, 1] # (1000 x b) x n
            dx = dx * dy.sign() # (1000 x b) x n
            dy = dy.abs() # (1000 x b) x n
            dx = dx.sum([0]) # n
            dy = dy.sum([0]) # n
            tangent_vector = torch.stack([dx, dy], dim=-1) # n x 2
            tangent_vector = F.normalize(tangent_vector, p=2, dim=-1) # n x 2
            normal_vector = torch.stack((-tangent_vector[:,1], tangent_vector[:,0]), dim=-1) # n x 2

            normal_std2 = torch.mul(direction, normal_vector.unsqueeze(0)).sum(dim=-1).pow(2).sum(dim=0) # n
            tangent_std2 = torch.mul(direction, tangent_vector.unsqueeze(0)).sum(dim=-1).pow(2).sum(dim=0) # n

            loss_lambda = torch.clamp(tangent_std2 / torch.clamp(normal_std2, min=1e-6), min=1.0, max=9.0).unsqueeze(0) # 1 x n
            # 1 x n
            loss_lambda = loss_lambda.detach()
        else:
            assert False
        return loss_lambda
    
    def _check_tensor(self, name, tensor):
        """检查张量的数值稳定性"""
        self.debug_mode= True
        if self.debug_mode:
            if torch.isnan(tensor).any():
                print(f"NaN detected in {name}")
                raise ValueError(f"{name} contains NaN")
            if torch.isinf(tensor).any():
                print(f"Inf detected in {name}: min={tensor.min()}, max={tensor.max()}")
                # 可以选择clamp或者直接报错
            if tensor.abs().max() > 1e6:
                print(f"Large values in {name}: max={tensor.abs().max()}")
    
    def forward(self, pred, sigma, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (Tensor[N, K, D]): Output regression.
            sigma (Tensor[N, K, D]): Output sigma.
            target (Tensor[N, K, D]): Target regression.
            target_weight (Tensor[N, K, D]):
                Weights across different joint types.
        """
        sigma = sigma.sigmoid()
        
        groundtruth = target * 2 - 1
        output = pred * 2 - 1

        normal_vector = self._get_normals_from_neighbors(groundtruth) # b x n x 2, [-1, 1]
        tangent_vector = self._inverse_vector(normal_vector) # b x n x 2, [-1, 1]

        pv_gt = output - groundtruth # b x n x 2, [-1, 1]

        normal_force = torch.mul(pv_gt, normal_vector).sum(dim=-1, keepdim=True) # b x n x 1
        tangent_force = torch.mul(pv_gt, tangent_vector).sum(dim=-1, keepdim=True) # b x n x 1

        loss_lambda = self._get_loss_lambda(pv_gt.detach(), normal_force.detach(), tangent_force.detach(), normal_vector.detach(), tangent_vector.detach(), lambda_mode=self.lambda_mode)

        alpha = 2 * loss_lambda / (loss_lambda + 1.0)
        belta = 2 * 1 / (loss_lambda + 1.0)
        
        dist = torch.cat((alpha * normal_force, belta * tangent_force), dim=-1)
        
        # if normal_vector.norm(p=2, dim=-1) < 0.5:
        #     delta = dist
        # else:
        #     delta = pv_gt

        # 用变换后的误差向量代替原始误差向量
        delta = dist / (sigma + 1e-9)
        
        error = pv_gt / (sigma + 1e-9)
        
        # 后续与RLELoss相同
        log_phi = self.flow_model.log_prob(delta.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1], 2)
        nf_loss = log_sigma - log_phi
        
        # self._check_tensor("pred", pred)
        # self._check_tensor("sigma", sigma) 
        # self._check_tensor("target", target)
        # self._check_tensor("error_anisotropic", error_anisotropic)
        # self._check_tensor("error", error)
        # self._check_tensor("log_phi", log_phi)

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()
    

@MODELS.register_module()
class IHLLoss(nn.Module):
    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 sigmoid=False,
                 q_distribution='laplace'):
        super(IHLLoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.sigmoid = sigmoid
        self.q_distribution = q_distribution

        self.flow_model = RealNVP()

    def forward(self, pred, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (Tensor[N, K, D]): Output regression.
            sigma (Tensor[N, K, D]): Output sigma.
            target (Tensor[N, K, D]): Target regression.
            target_weight (Tensor[N, K, D]):
                Weights across different joint types.
        """
        miu = pred[:, :, :2]
        sigma = pred[:, :, 2:4]

        sigma = torch.sqrt(abs(sigma))
        
        if self.sigmoid:
            sigma = sigma.sigmoid()
            
        error = (miu - target) / (sigma + 1e-9)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight.unsqueeze(-1)

        if self.size_average:
            loss /= len(loss)

        return loss.sum()    
