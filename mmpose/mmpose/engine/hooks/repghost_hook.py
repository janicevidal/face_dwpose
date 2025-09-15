# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks.hook import Hook

from mmpose.models.backbones.repghost import repghost_model_convert
from mmpose.registry import HOOKS


@HOOKS.register_module()
class RepGhostHook(Hook):

    def before_test_epoch(self, runner) -> None:
        runner.model = repghost_model_convert(runner.model)
