# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import build_plugin_layer
from ..builder import BACKBONES
from ..utils import CSPLayer


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@BACKBONES.register_module()
class CSPDarknetCH_MultiStream(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architechture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 in_channels=3,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 plugins=None,
                 stream=3,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position= ['after_stage1', 'after_stage2', 'after_stage3', 'after_stage4']
            assert all(p['position'] in allowed_position for p in plugins)
        self.stream = stream
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        self.plugins = plugins
        self.with_plugins = plugins is not None
        if self.with_plugins:
            # collect plugins for every stage
            self.after_stage1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_stage1'
            ]
            self.after_stage2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_stage2'
            ]
            self.after_stage3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_stage3'
            ]
            self.after_stage4_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_stage4'
            ]

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        # 3个focus模块
        self.stem_s1 = Focus(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.stem_s2 = Focus(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if stream == 3:
            self.stem_s3 = Focus(
            in_channels,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.layers = []
        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            layer = []
            for s in range(stream):
                
                in_channels = int(in_channels * widen_factor)
                out_channels = int(out_channels * widen_factor)
                num_blocks = max(round(num_blocks * deepen_factor), 1)
                stage = []
                conv_layer = conv(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(conv_layer)
                if use_spp:
                    spp = SPPBottleneck(
                        out_channels,
                        out_channels,
                        kernel_sizes=spp_kernal_sizes,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
                    stage.append(spp)
                csp_layer = CSPLayer(
                    out_channels,
                    out_channels,
                    num_blocks=num_blocks,
                    add_identity=add_identity,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(csp_layer)
                self.add_module(f'stage{i + 1}_s{s+1}', nn.Sequential(*stage))
                layer.append(f'stage{i + 1}_s{s+1}')
            self.layers.append(layer)
        
        if self.with_plugins:
            self.after_stage1_plugins_names = self.make_block_plugins(self.after_stage1_plugins, '_plugin_stage1')
            self.after_stage2_plugins_names = self.make_block_plugins(self.after_stage2_plugins, '_plugin_stage2')
            self.after_stage3_plugins_names = self.make_block_plugins(self.after_stage3_plugins, '_plugin_stage3')
            self.after_stage4_plugins_names = self.make_block_plugins(self.after_stage4_plugins, '_plugin_stage4')
            self.plugin_names = [self.after_stage1_plugins_names, self.after_stage2_plugins_names, self.after_stage3_plugins_names, self.after_stage4_plugins_names]

            for plgs in self.plugin_names:                
                if plgs[0] and plgs[1]:
                    _stage = int(plgs[0][0][-1])
                    self.add_module('b_norm_rgb'+plgs[0][0][-14:],\
                                        nn.BatchNorm2d(arch_setting[_stage-1][1], momentum=0.03, eps=0.001))
                    self.add_module('b_norm_tir'+plgs[1][0][-14:],\
                                        nn.BatchNorm2d(arch_setting[_stage-1][1], momentum=0.03, eps=0.001))

    print('a')

    def make_block_plugins(self,plugins, postfix):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        if not plugins:
            return  [[],[]]
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                # in_channels=in_channels,
                postfix=plugin.pop('postfix', postfix))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append([name])
        return plugin_names

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknetCH_MultiStream, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
    
    def forward_plugin(self, x1, x2, plugin_names):
        for name in plugin_names:
            x1, x2 = getattr(self, name)(x1, x2)
        return x1, x2

    # def forward(self, x):
    #     outs = []
    #     for i, layer_name in enumerate(self.layers):

    #         layer = getattr(self, layer_name)
            
    #         x = layer(x)
    #         if self.with_plugins:
    #             x = self.forward_plugin(x, self.plugin_names[i])
    #         if i in self.out_indices:
    #             outs.append(x)
    #     return tuple(outs)

    def forward(self, x):
        outs = []
        unique_outs = []
        # 先计算stem
        x1 = getattr(self, 'stem_s1')(x[:, :3, :, :])   #rgb
        x2 = getattr(self, 'stem_s2')(x[:, 3:, :, :])   #lwir
        if self.stream==3:
            x3 = getattr(self, 'stem_s3')(x)
        
        # 计算各stage的值，并在其中计算plugin
        for i, layer_name in enumerate(self.layers):
            x1 = getattr(self, layer_name[0])(x1)   #rgb
            x2 = getattr(self, layer_name[1])(x2)   #lwir
            if len(layer_name) == 3:
                x3 = getattr(self, layer_name[2])(x3)
            else:
                x3 = None
            if self.with_plugins:
                if len(layer_name) == 3:
                    if hasattr(self, f'b_norm_rgb_plugin_stage{i+1}'):
                        u_x1 = x1
                        x1, x3 = self.forward_plugin(x1, x3, self.plugin_names[i][0])
                        x1 += u_x1
                        assert int(self.plugin_names[i][0][0][-1]) == i + 1
                        x1 = getattr(self, f'b_norm_rgb_plugin_stage{i+1}')(x1)

                        u_x2 = x2
                        x2, x3 = self.forward_plugin(x2, x3, self.plugin_names[i][1])
                        x2 += u_x2
                        x2 = getattr(self, f'b_norm_tir_plugin_stage{i+1}')(x2)
                    else:
                        x1, x3 = self.forward_plugin(x1, x3, self.plugin_names[i][0])
                        x2, x3 = self.forward_plugin(x2, x3, self.plugin_names[i][1])
                else:               
                    if hasattr(self, f'b_norm_rgb_plugin_stage{i+1}'):
                        u_x1, u_x2 = x1, x2
                        x1, x2 = self.forward_plugin(x1, x2, self.plugin_names[i])
                        x1 += u_x1
                        assert int(self.plugin_names[i][0][0][-1]) == i + 1
                        x1 = getattr(self, f'b_norm_rgb_plugin_stage{i+1}')(x1)
                        x2 += u_x2
                        x2 = getattr(self, f'b_norm_tir_plugin_stage{i+1}')(x2)
                    else:
                        x1, x2 = self.forward_plugin(x1, x2, self.plugin_names[i])
                    x3 = None
            if i+1 in self.out_indices:
                outs.append([x1, x2, x3])
                unique_outs.append([u_x1, u_x2])
        return tuple(outs), tuple(unique_outs)
