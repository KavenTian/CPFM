import warnings

import torch.nn as nn

from mmcv.cnn import build_conv_layer,build_norm_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES

class Conv(BaseModule):
    def __init__(self,
                 conv_cfg,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 act='relu',
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'silu':
            self.act = nn.SiLU()
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels, postfix=1)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=dilation,
            bias=False
        )
        self.add_module(self.norm_name, norm)
    
    @property
    def norm(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    def forward(self, x):
        out = self.conv(x)
        out = self._modules[self.norm_name](out)
        out = self.act(out)
        return out


class Linear(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act='relu',
                 drop_out_rate=0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.Linear = nn.Linear(in_channels, out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act =='silu':
            self.act = nn.SiLU()
        self.dropout = nn.Dropout(drop_out_rate)
    
    def forward(self, x):
        out = self.Linear(x)
        out = self.act(out)
        out = self.dropout(out)
        
        return out


@BACKBONES.register_module()
class Illum_Aware_Module(BaseModule):
    '''illumination aware module.
    input_shape:(height, width) of module.Resize input image to (batch, 3, height, width).
    norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    '''
    def __init__(self,
                 in_channels,
                 norm_eval=True,
                 pretrained=None,
                 zero_init=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)


        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm']
                    )
                ]
                if zero_init:
                    self.init_cfg = dict(
                        type='Constant',
                        val=0,
                    )
            else:
                raise TypeError('pretrained must be a str or None')
        self.maxpool = nn.MaxPool2d((16, 20))
        self.conv1 = Conv(None, in_channels, 64, 3, 1, init_cfg=self.init_cfg)
        self.conv2 = Conv(None, 64, 32, 3, 2, init_cfg=self.init_cfg)
        self.linear1 = Linear(in_channels=8192,out_channels=128)   #32*1/4height*1/4width，此处待在pipeline中实现RGB的Resize后再修改
        self.linear2 = Linear(in_channels=128, out_channels=64) 
        self.illu = Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)
        self.norm_eval = norm_eval


    def _freeze_stage(self):    # 如此小的网络不需要固定部分参数，一般不使用此参数
        pass
    

    def forward(self, x):
        out = self.maxpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.view(out.shape[0], -1)    #b,
        out = self.linear1(out)
        
        out = self.linear2(out)
        out = self.illu(out)
        out = self.softmax(out)
        return out

    def train(self, mode=True):
        if mode and self.norm_eval:
            # trick: eval have effect on BatchNorm only
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
