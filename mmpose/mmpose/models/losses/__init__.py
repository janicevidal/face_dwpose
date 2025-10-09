# Copyright (c) OpenMMLab. All rights reserved.
from .ae_loss import AssociativeEmbeddingLoss
from .classification_loss import BCELoss, JSDiscretLoss, KLDiscretLoss, SinkhornDistance
from .heatmap_loss import (AdaptiveWingLoss, KeypointMSELoss,
                           KeypointOHKMMSELoss)
from .loss_wrappers import CombinedLoss, MultipleLossWrapper
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss, DSNTLoss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss, SimCC_DSNTRLE_Loss, 
                              AnisotropicDirectionLoss, AnisotropicRLELoss, IHLLoss)
from .kd import KDLoss
from .fea import FeaLoss

__all__ = [
    'KeypointMSELoss', 'KeypointOHKMMSELoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'KLDiscretLoss', 'MultipleLossWrapper', 'JSDiscretLoss', 'CombinedLoss', 'SinkhornDistance',
    'AssociativeEmbeddingLoss', 'SoftWeightSmoothL1Loss', 'KDLoss', 'FeaLoss',
    'SimCC_DSNTRLE_Loss', 'DSNTLoss', 'AnisotropicDirectionLoss', 'AnisotropicRLELoss', 'IHLLoss'
]
