# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization By Chengpeng Chen, Zichao Guo, Haien Zeng, Pengfei Xiong, and Jian Dong.
https://arxiv.org/abs/2211.06088
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_backbone import BaseBackbone

from mmcv.cnn import ConvModule
from mmpose.registry import MODELS

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.ReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class RepGhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True, reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        se_ratio=0.0,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st repghost bottleneck
        x1 = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)


# @MODELS.register_module()
# class RepGhostNet(BaseBackbone):
#     def __init__(
#         self,
#         cfgs,
#         width=1.0,
#         dropout=0.2,
#         shortcut=True,
#         reparam=True,
#         reparam_bn=True,
#         reparam_identity=False,
#         deploy=False,
#         init_cfg=[
#             dict(type='Kaiming', layer=['Conv2d']),
#             dict(
#                 type='Constant',
#                 val=1,
#                 layer=['_BatchNorm', 'GroupNorm'])
#         ], 
#     ):
#         super().__init__(init_cfg=init_cfg)
#         # setting of inverted residual blocks
#         self.cfgs = cfgs
#         self.dropout = dropout

#         # building first layer
#         output_channel = _make_divisible(16 * width, 4)
#         self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(output_channel)
#         self.act1 = nn.ReLU(inplace=True)
#         input_channel = output_channel

#         # building inverted residual blocks
#         stages = []
#         block = RepGhostBottleneck
#         for cfg in self.cfgs:
#             layers = []
#             for k, exp_size, c, se_ratio, s in cfg:
#                 output_channel = _make_divisible(c * width, 4)
#                 hidden_channel = _make_divisible(exp_size * width, 4)
#                 layers.append(
#                     block(
#                         input_channel,
#                         hidden_channel,
#                         output_channel,
#                         k,
#                         s,
#                         se_ratio=se_ratio,
#                         shortcut=shortcut,
#                         reparam=reparam,
#                         reparam_bn=reparam_bn,
#                         reparam_identity=reparam_identity,
#                         deploy=deploy
#                     ),
#                 )
#                 input_channel = output_channel
#             stages.append(nn.Sequential(*layers))

#         self.blocks = nn.Sequential(*stages)

#     def forward(self, x):
#         x = self.conv_stem(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.blocks(x)
#         return [x]

#     def convert_to_deploy(self):
#         repghost_model_convert(self, do_copy=False)


class Injection(nn.Module):
    """the Aggregate Attention Module.

    Args:
        in_channels (tuple): the input channel.
        out_channels (int): the output channel.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
    """

    def __init__(
            self,
            in_channels=(256, 384),
            out_channels=256,
            act_cfg=dict(type='Sigmoid'),
            norm_cfg=dict(type='BN'),
    ):
        super().__init__()
        self.embedding_s4 = ConvModule(
                        in_channels=in_channels[0],
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                    )
        self.embedding_s5 = ConvModule(
                        in_channels=in_channels[1],
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                    )
        self.act_embedding = ConvModule(
                        in_channels=in_channels[1],
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
    def forward(self, inputs):
        # x_x16, x_x32
        low_feat1 = self.embedding_s4(inputs[0])

        high_feat_act = F.interpolate(
            self.act_embedding(inputs[1]),
            size=low_feat1.shape[2:],
            mode='bilinear',
        )
        high_feat = F.interpolate(
            self.embedding_s5(inputs[1]),
            size=low_feat1.shape[2:],
            mode='bilinear')

        res = high_feat_act * low_feat1 + high_feat

        return res

    
@MODELS.register_module()
class RepGhostNet(BaseBackbone):
    def __init__(
        self,
        cfgs,
        out_indices,
        width=1.0,
        dropout=0.0,
        out_channels=256,
        out_feat_chs=None,
        shortcut=True,
        reparam=True,
        reparam_bn=True,
        reparam_identity=False,
        deploy=False,
        init_cfg=[
            dict(type='Kaiming', layer=['Conv2d']),
            dict(
                type='Constant',
                val=1,
                layer=['_BatchNorm', 'GroupNorm'])
        ], 
    ):
        super().__init__(init_cfg=init_cfg)
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.out_indices = out_indices

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        self.layers = []
        block = RepGhostBottleneck
        for i, cfg in enumerate(cfgs):
            layers = []
            # layer_name = 'layer{}'.format(i + 1)
            # layer_name = 'blocks{}'.format(i)
            layer_name = 'blocks{}'.format(i + 1)
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio=se_ratio,
                        shortcut=shortcut,
                        reparam=reparam,
                        reparam_bn=reparam_bn,
                        reparam_identity=reparam_identity,
                        deploy=deploy
                    ),
                )
                input_channel = output_channel
            
            self.add_module(layer_name, nn.Sequential(*layers))
            self.layers.append(layer_name)
            
        self.inj_module = Injection(in_channels=out_feat_chs, out_channels=out_channels)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        output = self.inj_module(outs)
        
        return [output]

    def convert_to_deploy(self):
        repghost_model_convert(self, do_copy=False)


def repghost_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    """
    taken from from https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


def repghostnet(enable_se=True, **kwargs):
    """
    Constructs a RepGhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 8, 16, 0, 1]],
        # stage2
        [[3, 24, 24, 0, 2]],
        [[3, 36, 24, 0, 1]],
        # stage3
        [[5, 36, 40, 0.25 if enable_se else 0, 2]],
        [[5, 60, 40, 0.25 if enable_se else 0, 1]],
        # stage4
        [[3, 120, 80, 0, 2]],
        [
            [3, 100, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 240, 112, 0.25 if enable_se else 0, 1],
            [3, 336, 112, 0.25 if enable_se else 0, 1],
        ],
        # stage5
        [[5, 336, 160, 0.25 if enable_se else 0, 2]],
        [
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
        ],
    ]

    return RepGhostNet(cfgs, **kwargs)