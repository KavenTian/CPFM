import logging, re
import torch
import torch.nn as nn

from mmcv.cnn import constant_init, kaiming_init, normal_init, build_plugin_layer
from ..builder import BACKBONES


def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation)

def make_vgg_layer(inplanes,
                   planes,
                   num_blocks,
                   dilation=1,
                   with_bn=False,
                   ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


@BACKBONES.register_module()
class VGG_Mul(nn.Module):
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 with_bn=False,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 bn_eval=True,
                 bn_frozen=False,
                 ceil_mode=False,
                 with_last_pool=True,
                 plugins=None,
                 stream=2,
                 init_cfg=None):
        super(VGG_Mul, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages

        self.plugins = plugins
        self.with_plugins = plugins is not None
        if plugins is not None:
            allowed_position= ['after_stage1', 'after_stage2', 'after_stage3', 'after_stage4']
            assert all(p['position'] in allowed_position for p in plugins)

        if self.with_plugins:
            # collect plugins for every stage
            self.after_stage0_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_stage0'
            ]
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

        self.stream = stream
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.init_cfg = init_cfg
        self.module_name = []
        for s in range(self.stream):
            out_channels = []
            self.inplanes = 3
            start_idx = 0
            vgg_layers = []
            self.range_sub_modules = []
            for i, num_blocks in enumerate(self.stage_blocks):
                num_modules = num_blocks * (2 + with_bn) + 1
                end_idx = start_idx + num_modules
                dilation = dilations[i]
                # planes = 64 * 2**i if i < 4 else 512
                planes = 64 * 2**i
                out_channels.append(planes)
                vgg_layer = make_vgg_layer(
                    self.inplanes,
                    planes,
                    num_blocks,
                    dilation=dilation,
                    with_bn=with_bn,
                    ceil_mode=ceil_mode)
                vgg_layers.extend(vgg_layer)
                self.inplanes = planes
                self.range_sub_modules.append([start_idx, end_idx])
                start_idx = end_idx
            if not with_last_pool:
                vgg_layers.pop(-1)
                self.range_sub_modules[-1][1] -= 1
            self.module_name.append(f'features_s{s+1}')
            self.add_module(self.module_name[-1], nn.Sequential(*vgg_layers))

        if self.with_plugins:
            self.after_stage0_plugins_names = self.make_block_plugins(self.after_stage0_plugins, '_plugin_stage0')
            self.after_stage1_plugins_names = self.make_block_plugins(self.after_stage1_plugins, '_plugin_stage1')
            self.after_stage2_plugins_names = self.make_block_plugins(self.after_stage2_plugins, '_plugin_stage2')
            self.after_stage3_plugins_names = self.make_block_plugins(self.after_stage3_plugins, '_plugin_stage3')
            self.after_stage4_plugins_names = self.make_block_plugins(self.after_stage4_plugins, '_plugin_stage4')
            self.plugin_names = [self.after_stage0_plugins_names,
                                self.after_stage1_plugins_names,
                                self.after_stage2_plugins_names,
                                self.after_stage3_plugins_names,
                                self.after_stage4_plugins_names]

            for plgs in self.plugin_names:
                if not plgs or not plgs[0]:
                    continue             
                _stage = int(plgs[0][0][-1])
                self.add_module('b_norm_rgb'+plgs[0][0][-14:],\
                                    nn.BatchNorm2d(out_channels[_stage], momentum=0.03, eps=0.001))
                self.add_module('b_norm_tir'+plgs[-1][0][-14:],\
                                    nn.BatchNorm2d(out_channels[_stage], momentum=0.03, eps=0.001))

        # self.init_from_original_weights()

    def init_from_original_weights(self):
        original_model = torch.load(self.init_cfg['checkpoint'], map_location=torch.device('cpu'))['state_dict']
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if 'plugin' in name:
                continue

            if 'features' in name:
                replace_str = re.compile('_s\d{1,}')
                origin_name = 'backbone.' + replace_str.sub('', name)
                state_dict[name] = original_model[origin_name]
      
        self.load_state_dict(state_dict)
        print('init weight done') 

    def make_block_plugins(self, plugins, postfix):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        if not plugins:
            return  [[],[]] if self.stream == 3 else []
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
    
    
    
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_plugin(self, x1, x2, plugin_names):
        for name in plugin_names:
            x1, x2 = getattr(self, name)(x1, x2)
        return x1, x2
    
    def forward(self, x):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        outs = []
        unique_outs = []
        vgg_layers_rgb = getattr(self, self.module_name[0])
        vgg_layers_tir = getattr(self, self.module_name[1])
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers_rgb[j]
                x1 = vgg_layer(x1)
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers_tir[j]
                x2 = vgg_layer(x2)
            
            if self.with_plugins:
                if hasattr(self, f'b_norm_rgb_plugin_stage{i}'):
                    u_x1, u_x2 = x1, x2
                    x1, x2 = self.forward_plugin(x1, x2, self.plugin_names[i][0])
                    x1 += u_x1
                    assert int(self.plugin_names[i][0][0][-1]) == i
                    x1 = getattr(self, f'b_norm_rgb_plugin_stage{i}')(x1)
                    x2 += u_x2
                    x2 = getattr(self, f'b_norm_tir_plugin_stage{i}')(x2)
                else:
                    x1, x2 = self.forward_plugin(x1, x2, self.plugin_names[i])
            
            if i in self.out_indices:
                outs.append([x1, x2, None])
                unique_outs.append([u_x1, u_x2])

        return tuple(outs), tuple(unique_outs)

    def train(self, mode=True):
        super(VGG_Mul, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        for s in [0, 1]:
            vgg_layers = getattr(self, self.module_name[s])
            if mode and self.frozen_stages >= 0:
                for i in range(self.frozen_stages):
                    for j in range(*self.range_sub_modules[i]):
                        mod = vgg_layers[j]
                        mod.eval()
                        for param in mod.parameters():
                            param.requires_grad = False

