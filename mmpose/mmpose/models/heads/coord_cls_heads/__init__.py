# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .simcc_head import SimCCHead
from .litecc_head import LiteCCHead
from .litecc_integral_head import LiteCCIntegralHead
from .litecc_ihl_head import LiteImplicitHead
from .litecc_rtmo_head import LiteMLEHead
from .litecc_rtmo_angles_head import MultiTaskLiteMLEHead
from .litecc_rtmo_angles_hybrid_head import MultiTaskHybridLiteMLEHead

__all__ = ['SimCCHead', 'RTMCCHead', 'LiteCCHead', 'LiteCCIntegralHead', 'LiteImplicitHead', 'LiteMLEHead', 'MultiTaskLiteMLEHead', 'MultiTaskHybridLiteMLEHead']
