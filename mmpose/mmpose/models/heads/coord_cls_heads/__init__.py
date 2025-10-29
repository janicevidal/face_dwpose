# Copyright (c) OpenMMLab. All rights reserved.
from .rtmcc_head import RTMCCHead
from .simcc_head import SimCCHead
from .litecc_head import LiteCCHead
from .litecc_integral_head import LiteCCIntegralHead
from .litecc_ihl_head import LiteImplicitHead
from .litecc_rtmo_head import LiteMLEHead

__all__ = ['SimCCHead', 'RTMCCHead', 'LiteCCHead', 'LiteCCIntegralHead', 'LiteImplicitHead', 'LiteMLEHead']
