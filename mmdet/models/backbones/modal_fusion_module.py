from typing import Type
import warnings
from mmcv.cnn.bricks import conv
from ..builder import build_loss

import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer,build_norm_layer,build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils.correlation import corrcoef

class BN_Conv(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation=1,
                 act='relu',
                 norm_cfg=dict(type='GN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
        if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm']
                    )
                ]
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels, postfix=1)
        self.conv = build_conv_layer(
            None,
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
        return getattr(self, self.norm_name)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

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
            self.act = nn.ReLU(inplace=True)
        elif act == 'silu':
            self.act = nn.SiLU(inplace=True)
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
        return getattr(self, self.norm_name)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)

        return out


class SelfAttention(BaseModule):
    def __init__(self,
                 d_model, 
                 d_k,
                 d_v,
                 h,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 init_cfg=None
                  ):
        super().__init__(init_cfg=init_cfg)
        assert d_k % h ==0
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.que_proj = nn.Linear(d_model, h * self.d_k)
        self.key_proj = nn.Linear(d_model, h * self.d_k)
        self.val_proj = nn.Linear(d_model, h * self.d_v)
        self.out_proj = nn.Linear(h * self.d_v, d_model)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weight=None):
        
        b_s, nq = x.shape[:2]   #TODO:nq值与nk值
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous() #(b_s, h, nq, d_k) K
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous() #(b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous() #(b_s, h, nk, d_v)
        

        att = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attention_weight is not None:
            att = att * attention_weight
        if attention_mask is not None:
            att = att.mask_fill(attention_mask, -np.inf)
        
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)    # (b_s, nq, h * d_v)
        out = self.resid_drop(self.out_proj(out))   # (b_s, nq, d_model)
        return out


class TransformerBlock(BaseModule):

    def __init__(self,
                d_model,
                d_k,
                d_v,
                h,
                block_exp,
                attn_pdrop,
                resid_pdrop,
                init_cfg=None 
                ):
        super().__init__(init_cfg=init_cfg)
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class CFT_block(BaseModule):
    def __init__(self,
                d_model,
                h=2,
                block_exp=4,
                n_layer=6,
                vert_anchors=8,
                horz_anchors=10,
                embed_pdrop=0.1,
                attn_pdrop=0.1,
                resid_pdrop=0.1,
                init_cfg=None,
                ):
        super().__init__(init_cfg=init_cfg)

        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model
        self.n_embed = d_model
        self.pos_emb = nn.Parameter(torch.zeros(1, 3 * vert_anchors * horz_anchors , self.n_embed))
        self.trans_blocks = nn.Sequential(*[TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                    for layer in range(n_layer)
                                    ])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embed)

        # regularization
        self.drop = nn.Dropout(attn_pdrop)

        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weight
        self.init_weights()

    def init_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        


    def forward(self, x):
        
        rgb_fea = x[0]
        ir_fea = x[1]
        pub_fea = x[2]
        
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation

        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)
        pub_fea = self.avgpool(pub_fea)

        rgb_fea_flat = rgb_fea.view(bs, c, -1)
        ir_fea_flat = ir_fea.view(bs, c, -1)
        pub_fea_flat = pub_fea.view(bs, c, -1)

        token_embedings = torch.cat([rgb_fea_flat, ir_fea_flat, pub_fea_flat], dim=2)   #concat

        token_embedings = token_embedings.permute(0, 2, 1).contiguous() # dim:(B, 3*H*W, C)

        x = self.drop(self.pos_emb + token_embedings)
        x = self.trans_blocks(x) #dim:(B, 3*H*W, C)

        x = self.ln_f(x)
        # x = x.view(bs, 3, self.vert_anchors, self.horz_anchors, self.n_embed)
        # x = x.permute(0, 1, 4, 2, 3)    #dim:(B, 3, C, H, W)

        # rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embed, self.vert_anchors, self.horz_anchors)
        # ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embed, self.vert_anchors, self.horz_anchors)
        # pub_fea_out = x[:, 2, :, :, :].contiguous().view(bs, self.n_embed, self.vert_anchors, self.horz_anchors)

        # rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        # ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        # pub_fea_out = F.interpolate(pub_fea_out, size=([h, w]), mode='bilinear')

        # return rgb_fea_out, ir_fea_out, pub_fea_out
        
        # transformer作为模态融合模块放在FPN中间
        x = x.view(bs, self.vert_anchors, self.horz_anchors, self.n_embed * 3).contiguous()
        x = x.permute(0,  3, 1, 2)  #dim:(B, 3C, H, W)

        x = F.interpolate(x, size=([h, w]), mode='bilinear', align_corners=False)
        return x
    
    def train(self, mode=True):
        if mode and self.norm_eval:
            # trick: eval have effect on BatchNorm only
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()



@BACKBONES.register_module()
class ModalFusionWithTransformer(BaseModule):
    '''
    transformer fusion module
    '''

    def __init__(self, 
                in_channels,
                out_channels,
                norm_eval=True,
                zero_init=False,
                init_cfg=None):
        super().__init__(init_cfg=init_cfg)
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
                    conv_init_config = dict(
                        type='Constant',
                        val=0,
                    )
        self.module_lists = nn.ModuleList()
        for in_channel,out_channel in zip(in_channels, out_channels):
            transformer = CFT_block(in_channel)     
            self.module_lists.append(nn.Sequential(
                transformer,
                 Conv(None, in_channel * 3, out_channel, 3, 1, init_cfg=self.init_cfg)
                
            ))
        self.norm_eval = norm_eval
        
    
    @property
    def norm(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm_name)

    def forward(self, x_rgb, x_pub, x_lwir):
        
        x = [self.module_lists[i]([rgb, lwir, pub]) for i, (rgb, pub, lwir) in enumerate(zip(x_rgb, x_pub, x_lwir))]
        
        # out = [self.bn_conv[i](w) for i, w in enumerate(x)]
        
        return x

    def train(self, mode=True):
        if mode and self.norm_eval:
            # trick: eval have effect on BatchNorm only
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class SE_bolock(BaseModule):
    def __init__(self, in_channel, reduction=16, init_cfg=None):
        super().__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze
        y = self.fc(y).view(b, c, 1, 1) # excitation
        return x * y.expand_as(x)


@BACKBONES.register_module()
class ModalFusion(BaseModule):
    '''
    concat fusion module
    '''

    def __init__(self, 
                in_channels,
                out_channels,
                streams=['rgb', 'lwir', 'pub'],
                norm_eval=True,
                use_corrloss=False,
                with_se=False,
                init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.init_cfg = init_cfg
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm']
                    )
            ]
        self.module_lists = nn.ModuleList()
        n_modal = len(streams)
        for in_channel,out_channel in zip(in_channels, out_channels):
            bn_conv = Conv(None, in_channel*n_modal, out_channel, 3, 1, init_cfg=self.init_cfg)    
            self.module_lists.append(bn_conv)
        self.norm_eval = norm_eval
        self.streams = streams
        self.n_branch = len(in_channels)
        self.with_se = with_se
        if with_se:
            self.se = nn.ModuleList()
            for in_channel,out_channel in zip(in_channels, out_channels):    
                self.se.append(SE_bolock(in_channel=in_channel*len(streams)))
        self.use_corrloss = use_corrloss
        if use_corrloss:
            self.corrloss = nn.ModuleList()
            for i in range(len(in_channels)):
                self.corrloss.append(build_loss(
                    dict(type='MSELoss', loss_weight=1.0)
                ))

    @property
    def norm(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm_name)

    def forward(self, x_rgb, x_pub, x_lwir):
        out = []
        if self.use_corrloss:
            if self.train():
                loss_corr = []
        for i in range(self.n_branch):
            if len(self.streams) == 3:
                if self.use_corrloss:
                    if self.train():
                        loss_corr.append(self._corrloss([x_rgb[i], x_pub[i], x_lwir[i]], i))
                    
                x = torch.cat([x_rgb[i], x_lwir[i], x_pub[i]], axis=1)
            elif set(['rgb', 'lwir', 'pub']) - set(self.streams) == set(['pub']):
                x = torch.cat([x_rgb[i], x_lwir[i]], axis=1)
            elif set(['rgb', 'lwir', 'pub']) - set(self.streams) == set(['pub', 'lwir']):
                x = x_rgb[i]
            elif set(['rgb', 'lwir', 'pub']) - set(self.streams) == set(['pub', 'rgb']):
                x = x_lwir[i]
            elif set(['rgb', 'lwir', 'pub']) - set(self.streams) == set(['lwir', 'rgb']):
                x = x_pub[i]
            out.append(self.module_lists[i](x)) 
        if self.use_corrloss:
            if self.train():
                sum(loss_corr).backward(retain_graph=True)

        # x = [torch.cat([rgb, lwir, pub], axis=1) for rgb, pub, lwir in zip(x_rgb, x_pub, x_lwir)]
        
        # out = [self.module_lists[i](w) for i, w in enumerate(x)]
        return out

    def _corrloss(self, feat, i):
        b = feat[0].shape[0]
        feat = [feat[k].flatten(start_dim=1).unsqueeze(1) for k in range(len(self.streams))]
        feat = torch.cat(feat, axis=1)
        # 构造target
        target = torch.eye(len(self.streams), dtype=torch.float).repeat(b,1,1).cuda()
        feat_corr = torch.zeros_like(target).cuda()
        for j in range(b):
            feat_corr[j,:,:] = corrcoef(feat[j])
        return self.corrloss[i](feat_corr, target)


    def train(self, mode=True):
        super(ModalFusion, self).train(mode)
        if mode and self.norm_eval:
            # trick: eval have effect on BatchNorm only
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
