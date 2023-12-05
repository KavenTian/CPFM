# kinds of modal fusion module
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import PLUGIN_LAYERS, build_activation_layer
from ..backbones.pvt import AbsolutePositionEmbedding

"""
modal fusion blocks:
    - CoCrossAttention: two cross attention layer
    - 
"""

@PLUGIN_LAYERS.register_module()
class CoCrossAttention(nn.Module):
    """
    two cross attention layer
    - reimplement from ViLBERT: Pretraining Task-Agnostic Visiolinguistic
      Representations for Vision-and-Language Tasks [NIPS2020]
    pos_shape: pos_embed shape [h, w]
    pos_dim: pos_embed dim ndim
    """
    def __init__(self, pos_shape, pos_dim, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, 
    act_cfg=dict(type='ReLU'),
    ):
        super(CoCrossAttention, self).__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.norm11 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        
        # FeedForward
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout) #FFD中的dropout
        self.linear12 = nn.Linear(dim_feedforward, d_model)
        self.dropout12 = nn.Dropout(dropout)
        self.norm12 = nn.LayerNorm(d_model)
        

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)
        self.dropout22 = nn.Dropout(dropout)
        self.norm22 = nn.LayerNorm(d_model)
        

        self.act1 = build_activation_layer(act_cfg)
        self.act2 = build_activation_layer(act_cfg)
        # init co attention weight 
        self.init_weights()
        # pos embed
        self.pos_embed = AbsolutePositionEmbedding(pos_shape=pos_shape, pos_dim=pos_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2):
        n, c, h, w = src1.shape
        src1 = src1.flatten(2).permute(0, 2, 1)
        src2 = src2.flatten(2).permute(0, 2, 1)
        pos_embed = self.pos_embed(x = src1, hw_shape=[h, w])
        # attention of src1
        src1_att = self.multihead_attn1(
            query=self.with_pos_embed(src1, pos_embed),
            key=self.with_pos_embed(src2, pos_embed),
            value=src2
        )[0]
        src1 = src1 + self.dropout11(src1_att)
        src1 = self.norm11(src1)
        src12 = self.linear12(self.dropout1(self.act1(self.linear11(src1))))
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        
        # attention of src2
        src2_att = self.multihead_attn2(
            query=self.with_pos_embed(src2, pos_embed),
            key=self.with_pos_embed(src1, pos_embed),
            value=src1
        )[0]
        src2 = src2 + self.dropout21(src2_att)
        src2 = self.norm21(src2)
        src22 = self.linear22(self.dropout2(self.act2(self.linear21(src2))))
        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)

        # 变形回原来的值 b, hw, c -> b ,c, h, w
        src1 = src1.transpose(1,2).view(n, c, h, w).contiguous()
        src2 = src2.transpose(1,2).view(n, c, h, w).contiguous()
        return src1, src2

    def forward_pre(self, src1, src2):
        n, c, h, w = src1.shape
        src1 = src1.flatten(2).permute(0, 2, 1)
        src2 = src2.flatten(2).permute(0, 2, 1)
        pos_embed = self.pos_embed(x = src1, hw_shape=[h, w])
        # attention of src1
        src1 = self.norm11(src1)
        src1_att = self.multihead_attn1(
            query=self.with_pos_embed(src1, pos_embed),
            key=self.with_pos_embed(src2, pos_embed),
            value=src2
        )[0]
        src1 = src1 + self.dropout11(src1_att)
        
        src12 = self.linear12(self.dropout1(self.act1(self.linear11(self.norm12(src1)))))
        src1 = src1 + self.dropout12(src12)
        
        # attention of src2
        src2 = self.norm21(src2)
        src2_att = self.multihead_attn2(
            query=self.with_pos_embed(src2, pos_embed),
            key=self.with_pos_embed(src1, pos_embed),
            value=src1
        )[0]
        src2 = src2 + self.dropout21(src2_att)
        
        src22 = self.linear22(self.dropout2(self.act2(self.linear21(self.norm22(src2)))))
        src2 = src2 + self.dropout22(src22)

        # 变形回原来的值 b, hw, c -> b ,c, h, w
        src1 = src1.transpose(1,2).view(n, c, h, w).contiguous()
        src2 = src2.transpose(1,2).view(n, c, h, w).contiguous()
        return src1, src2

    def forward(self, src1, src2):
        return self.forward_post(src1, src2)
        # return self.forward_pre(src1, src2)


@PLUGIN_LAYERS.register_module()
class CoCrossAttentionCopy(nn.Module):
    """
    three cross attention layer copy from CoCrossAttention
    - reimplement from ViLBERT: Pretraining Task-Agnostic Visiolinguistic
      Representations for Vision-and-Language Tasks [NIPS2020]
    pos_shape: pos_embed shape [h, w]
    pos_dim: pos_embed dim ndim
    """
    def __init__(self, pos_shape, pos_dim, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, 
    act_cfg=dict(type='ReLU'),
    ):
        super(CoCrossAttentionCopy, self).__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.norm11 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        
        # FeedForward
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout) #FFD中的dropout
        self.linear12 = nn.Linear(dim_feedforward, d_model)
        self.dropout12 = nn.Dropout(dropout)
        self.norm12 = nn.LayerNorm(d_model)
        

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)
        self.dropout22 = nn.Dropout(dropout)
        self.norm22 = nn.LayerNorm(d_model)
        

        self.act1 = build_activation_layer(act_cfg)
        self.act2 = build_activation_layer(act_cfg)
        # init co attention weight 
        self.init_weights()
        # pos embed
        self.pos_embed = AbsolutePositionEmbedding(pos_shape=pos_shape, pos_dim=pos_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2):
        n, c, h, w = src1.shape
        src1 = src1.flatten(2).permute(0, 2, 1)
        src2 = src2.flatten(2).permute(0, 2, 1)
        pos_embed = self.pos_embed(x = src1, hw_shape=[h, w])
        # attention of src1
        src1_att = self.multihead_attn1(
            query=self.with_pos_embed(src1, pos_embed),
            key=self.with_pos_embed(src2, pos_embed),
            value=src2
        )[0]
        src1 = src1 + self.dropout11(src1_att)
        src1 = self.norm11(src1)
        src12 = self.linear12(self.dropout1(self.act1(self.linear11(src1))))
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        
        # attention of src2
        src2_att = self.multihead_attn2(
            query=self.with_pos_embed(src2, pos_embed),
            key=self.with_pos_embed(src1, pos_embed),
            value=src1
        )[0]
        src2 = src2 + self.dropout21(src2_att)
        src2 = self.norm21(src2)
        src22 = self.linear22(self.dropout2(self.act2(self.linear21(src2))))
        src2 = src2 + self.dropout22(src22)
        src2 = self.norm22(src2)

        # 变形回原来的值 b, hw, c -> b ,c, h, w
        src1 = src1.transpose(1,2).view(n, c, h, w).contiguous()
        src2 = src2.transpose(1,2).view(n, c, h, w).contiguous()
        return src1, src2

    def forward_pre(self, src1, src2):
        n, c, h, w = src1.shape
        src1 = src1.flatten(2).permute(0, 2, 1)
        src2 = src2.flatten(2).permute(0, 2, 1)
        pos_embed = self.pos_embed(x = src1, hw_shape=[h, w])
        # attention of src1
        src1 = self.norm11(src1)
        src1_att = self.multihead_attn1(
            query=self.with_pos_embed(src1, pos_embed),
            key=self.with_pos_embed(src2, pos_embed),
            value=src2
        )[0]
        src1 = src1 + self.dropout11(src1_att)
        
        src12 = self.linear12(self.dropout1(self.act1(self.linear11(self.norm12(src1)))))
        src1 = src1 + self.dropout12(src12)
        
        # attention of src2
        src2 = self.norm21(src2)
        src2_att = self.multihead_attn2(
            query=self.with_pos_embed(src2, pos_embed),
            key=self.with_pos_embed(src1, pos_embed),
            value=src1
        )[0]
        src2 = src2 + self.dropout21(src2_att)
        
        src22 = self.linear22(self.dropout2(self.act2(self.linear21(self.norm22(src2)))))
        src2 = src2 + self.dropout22(src22)

        # 变形回原来的值 b, hw, c -> b ,c, h, w
        src1 = src1.transpose(1,2).view(n, c, h, w).contiguous()
        src2 = src2.transpose(1,2).view(n, c, h, w).contiguous()
        return src1, src2

    def forward(self, src1, src2):
        return self.forward_post(src1, src2)
        # return self.forward_pre(src1, src2)


@PLUGIN_LAYERS.register_module()
class CoAttention(nn.Module):
    """
    two cross attention layer
    - reimplement from ViLBERT: Pretraining Task-Agnostic Visiolinguistic
      Representations for Vision-and-Language Tasks [NIPS2020]
    pos_shape: pos_embed shape [h, w]
    pos_dim: pos_embed dim ndim
    """
    def __init__(self, pos_shape, pos_dim, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, 
    act_cfg=dict(type='ReLU'),
    ):
        super(CoAttention, self).__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.norm11 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        
        # FeedForward
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout) #FFD中的dropout
        self.linear12 = nn.Linear(dim_feedforward, d_model)
        self.dropout12 = nn.Dropout(dropout)
        self.norm12 = nn.LayerNorm(d_model)
        

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)
        self.dropout22 = nn.Dropout(dropout)
        self.norm22 = nn.LayerNorm(d_model)
        

        self.act1 = build_activation_layer(act_cfg)
        self.act2 = build_activation_layer(act_cfg)
        # init co attention weight 
        self.init_weights()
        # pos embed
        self.pos_embed = AbsolutePositionEmbedding(pos_shape=pos_shape, pos_dim=pos_dim)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2):
        n, c, h, w = src1.shape
        src1 = src1.flatten(2).permute(0, 2, 1)
        src2 = src2.flatten(2).permute(0, 2, 1)
        pos_embed = self.pos_embed(x = src1, hw_shape=[h, w])
        # attention of src1
        src1_att = self.multihead_attn1(
            query=self.with_pos_embed(src1, pos_embed),
            key=self.with_pos_embed(src2, pos_embed),
            value=src2
        )[0]
        _src1 = src1 + self.dropout11(src1_att)
        _src1 = self.norm11(_src1)
        src12 = self.linear12(self.dropout1(self.act1(self.linear11(_src1))))
        _src1 = _src1 + self.dropout12(src12)
        _src1 = self.norm12(_src1)
        
        # attention of src2
        src2_att = self.multihead_attn2(
            query=self.with_pos_embed(src2, pos_embed),
            key=self.with_pos_embed(src1, pos_embed),
            value=src1
        )[0]
        _src2 = src2 + self.dropout21(src2_att)
        _src2 = self.norm21(_src2)
        src22 = self.linear22(self.dropout2(self.act2(self.linear21(_src2))))
        _src2 = _src2 + self.dropout22(src22)
        _src2 = self.norm22(_src2)

        # 变形回原来的值 b, hw, c -> b ,c, h, w
        _src1 = _src1.transpose(1,2).view(n, c, h, w).contiguous()
        _src2 = _src2.transpose(1,2).view(n, c, h, w).contiguous()
        return _src1, _src2

    def forward(self, src1, src2):
        return self.forward_post(src1, src2)