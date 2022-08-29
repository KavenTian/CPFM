import math, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from cvpods.layers.border_align import BorderAlign

from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from .yolox_head import YOLOXHead



fp32_apply_cls = ("rgb_decoded_bbox",
                            "tir_decoded_bbox",
                            "union_decoded_bbox",
                            "obj_rgb",
                            "obj_tir",
                            "obj_union",
                            "cls_rgb",
                            "cls_tir")
fp32_apply_nocls = ("rgb_decoded_bbox",
                            "tir_decoded_bbox",
                            "union_decoded_bbox",
                            "obj_rgb",
                            "obj_tir",
                            "obj_union")

@HEADS.register_module()
class MultiSpeHead(YOLOXHead):
    def __init__(self,
                 use_cls_branch,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):

        super(YOLOXHead, self).__init__(init_cfg=init_cfg)
        self.use_cls_branch = use_cls_branch
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls) if use_cls_branch else None
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)

        self.use_l1 = False  # This flag will be modified by hooks.
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if not use_cls_branch:
            self.train_cfg['assigner']['cls_weight'] = 0.

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self._init_layers()
        self.feat_align = BorderAlign(pool_size=10)


    def _init_layers(self):
        # fusion
        self.rgb_unique_fusion_reg = nn.ModuleList()
        self.tir_unique_fusion_reg = nn.ModuleList()

        # branch
        self.union_multi_level_reg_convs = nn.ModuleList()
        self.rgb_multi_level_reg_convs = nn.ModuleList()
        self.tir_multi_level_reg_convs = nn.ModuleList()
        
        # head
        self.union_multi_level_conv_reg = nn.ModuleList()
        self.rgb_multi_level_conv_reg = nn.ModuleList()
        self.tir_multi_level_conv_reg = nn.ModuleList()
        
        self.union_multi_level_conv_obj = nn.ModuleList()
        self.rgb_multi_level_conv_obj = nn.ModuleList()
        self.tir_multi_level_conv_obj = nn.ModuleList()
        if self.use_cls_branch:
            self.rgb_unique_fusion_cls = nn.ModuleList()
            self.tir_unique_fusion_cls = nn.ModuleList()

            self.union_multi_level_cls_convs = nn.ModuleList()
            self.rgb_multi_level_cls_convs = nn.ModuleList()
            self.tir_multi_level_cls_convs = nn.ModuleList()

            self.rgb_multi_level_conv_cls = nn.ModuleList()
            self.tir_multi_level_conv_cls = nn.ModuleList()

        for i in range(len(self.strides)):
            self.rgb_unique_fusion_reg.append(
                self._build_fusion_convs(2**i*256+self.feat_channels, 5 * self.feat_channels, 2))
            self.tir_unique_fusion_reg.append(
                self._build_fusion_convs(2**i*256+self.feat_channels, 5 * self.feat_channels, 2))

            self.union_multi_level_reg_convs.append(self._build_stacked_convs(self.stacked_convs))
            self.rgb_multi_level_reg_convs.append(
                self._build_fusion_convs(5 * self.feat_channels, self.feat_channels, self.stacked_convs // 2))
            self.tir_multi_level_reg_convs.append(
                self._build_fusion_convs(5 * self.feat_channels, self.feat_channels, self.stacked_convs // 2))
            
            if self.use_cls_branch:
                self.rgb_unique_fusion_cls.append(
                    self._build_fusion_convs(2**i*256+self.feat_channels, 5 * self.feat_channels, 2))
                self.tir_unique_fusion_cls.append(
                    self._build_fusion_convs(2**i*256+self.feat_channels, 5 * self.feat_channels, 2))

                self.union_multi_level_cls_convs.append(self._build_stacked_convs(self.stacked_convs))
                self.rgb_multi_level_cls_convs.append(
                    self._build_fusion_convs(5 * self.feat_channels, self.feat_channels, self.stacked_convs // 2))
                self.tir_multi_level_cls_convs.append(
                    self._build_fusion_convs(5 * self.feat_channels, self.feat_channels, self.stacked_convs // 2))
            
            _, conv_reg, conv_obj = self._build_predictor()
            self.union_multi_level_conv_reg.append(conv_reg)
            self.union_multi_level_conv_obj.append(conv_obj)
            
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.rgb_multi_level_conv_reg.append(conv_reg)
            self.rgb_multi_level_conv_obj.append(conv_obj)
            if self.use_cls_branch:
                self.rgb_multi_level_conv_cls.append(conv_cls)

            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.tir_multi_level_conv_reg.append(conv_reg)
            self.tir_multi_level_conv_obj.append(conv_obj)
            if self.use_cls_branch:
                self.tir_multi_level_conv_cls.append(conv_cls)

    def _build_fusion_convs(self, in_channels, out_channels, num_stack):
        conv = ConvModule
        stacked_convs = [conv(in_channels, self.feat_channels, 1,
                              conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg,
                              bias=self.conv_bias)]
        for _ in range(num_stack - 1):

            stacked_convs.append(
                conv(self.feat_channels,
                     self.feat_channels,
                     3,
                     stride=1,
                     padding=1,
                     conv_cfg=self.conv_cfg,
                     norm_cfg=self.norm_cfg,
                     act_cfg=self.act_cfg,
                     bias=self.conv_bias)
            )
        stacked_convs.append(
                conv(self.feat_channels,
                     out_channels,
                     3,
                     stride=1,
                     padding=1,
                     conv_cfg=self.conv_cfg,
                     norm_cfg=self.norm_cfg,
                     act_cfg=self.act_cfg,
                     bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_stacked_convs(self, num_stacks):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(num_stacks):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == num_stacks - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1) if self.use_cls_branch else None
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def init_weights(self):
        super(YOLOXHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for union_obj, rgb_obj, tir_obj in zip(self.union_multi_level_conv_obj,
                                               self.rgb_multi_level_conv_obj,
                                               self.tir_multi_level_conv_obj):
            union_obj.bias.data.fill_(bias_init)
            rgb_obj.bias.data.fill_(bias_init)
            tir_obj.bias.data.fill_(bias_init)
            
        if self.use_cls_branch:
            for rgb_cls, tir_cls in zip(self.rgb_multi_level_conv_cls,
                                        self.tir_multi_level_conv_cls):
                rgb_cls.bias.data.fill_(bias_init)
                tir_cls.bias.data.fill_(bias_init)
   
    def _prior_bbox(self, size, stride, dtype, device):
        assert len(size) == 2
        feat_h, feat_w = size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_x = shift_x.to(dtype)

        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_y = shift_y.to(dtype)

        shift_xx, shift_yy = torch.meshgrid(shift_x, shift_y)
        shift_xx, shift_yy = shift_xx.T, shift_yy.T
        stride_w = shift_xx.new_full(shift_xx.shape, stride).to(dtype)
        stride_h = shift_yy.new_full(shift_yy.shape, stride).to(dtype)
        shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=0)
        
        return shifts.to(device)
    
    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[..., 2:]) + priors[..., :2]
        whs = bbox_preds[..., 2:].exp() * priors[..., 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes
    
    def border_align(self, feats, init_bbox, strides):
        # feats.shape: N,5C,H,W
        # init_bbox.shape: N,H,W,4
        # first scale the bbox
        N, Ch, H, W = feats.shape
        C = Ch // 5
        bbox = init_bbox / strides
        bbox[...,0].clamp_(min=0, max=W-1)
        bbox[...,1].clamp_(min=0, max=H-1)
        bbox[...,2].clamp_(min=0, max=W-1)
        bbox[...,3].clamp_(min=0, max=H-1)
        bbox = bbox.reshape(N, -1, 4)

        # for border align
        shot = feats[:,:C,:,:]
        align_feats = feats[:,C:,:,:]
        align_feats = self.feat_align(align_feats, bbox)
        align_feats = align_feats.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        out = torch.cat([align_feats, shot], dim=1)
        
        return out
    
    def get_refined_bbox(self, bbox_init, delta):
        wh_init = bbox_init[...,2:] - bbox_init[...,:2]
        wh_init = torch.cat([wh_init, wh_init], dim=-1)
        assert wh_init.shape == delta.shape == bbox_init.shape
        refined_bbox = delta * wh_init + bbox_init

        return refined_bbox
    
    def forward_single(self,
                        x, unique_x, strides,
                        rgb_u_fu_reg,
                        tir_u_fu_reg,
                        u_reg_convs,
                        rgb_reg_convs,
                        tir_reg_convs,
                        u_conv_reg,
                        rgb_conv_reg,
                        tir_conv_reg,
                        u_conv_obj,
                        rgb_conv_obj,
                        tir_conv_obj,
                        rgb_u_fu_cls=None,
                        tir_u_fu_cls=None,
                        u_cls_convs=None,
                        rgb_cls_convs=None,
                        tir_cls_convs=None,
                        rgb_conv_cls=None,
                        tir_conv_cls=None):
        rgb_unique, tir_unique = unique_x
        reg_feats = u_reg_convs(x)
        if self.use_cls_branch:
            cls_feats = u_cls_convs(x)

        bbox_pred_init = u_conv_reg(reg_feats)  # init_reg head c_x,c_y,w,h
        obj_init = u_conv_obj(reg_feats)        # init_obj head

        rgb_feats_reg = torch.cat((rgb_unique, reg_feats), dim=1)
        rgb_feats_reg = rgb_u_fu_reg(rgb_feats_reg)
        tir_feats_reg = torch.cat((tir_unique, reg_feats), dim=1)
        tir_feats_reg = tir_u_fu_reg(tir_feats_reg)
        if self.use_cls_branch:
            rgb_feats_cls = torch.cat((rgb_unique, cls_feats), dim=1)
            rgb_feats_cls = rgb_u_fu_cls(rgb_feats_cls)
            tir_feats_cls = torch.cat((tir_unique, cls_feats), dim=1)
            tir_feats_cls = tir_u_fu_cls(tir_feats_cls)

        featmap_sizes = bbox_pred_init.shape[2:]
        _prior_bbox = self._prior_bbox(featmap_sizes, strides, bbox_pred_init.dtype, bbox_pred_init.device)
        assert bbox_pred_init.shape[1:] == _prior_bbox.shape

        bbox_pred_off = bbox_pred_init.detach()
        # get x1y1x2y2 box
        pre_bbox_init = self._bbox_decode(_prior_bbox.permute(1, 2, 0), bbox_pred_off.permute(0, 2, 3, 1))
        
        rgb_feats_reg = self.border_align(rgb_feats_reg, pre_bbox_init, strides)
        tir_feats_reg = self.border_align(tir_feats_reg, pre_bbox_init, strides)
        if self.use_cls_branch:
            rgb_feats_cls = self.border_align(rgb_feats_cls, pre_bbox_init, strides)
            tir_feats_cls = self.border_align(tir_feats_cls, pre_bbox_init, strides)

        rgb_feats_reg = rgb_reg_convs(rgb_feats_reg)
        tir_feats_reg = tir_reg_convs(tir_feats_reg)
        if self.use_cls_branch:
            rgb_feats_cls = rgb_cls_convs(rgb_feats_cls)
            tir_feats_cls = tir_cls_convs(tir_feats_cls)
        
        # delta_bbox coded as [(x1-x1_init)/w_init, (y1-y1_init)/h_init,
        #                      (x2-x2_init)/w_init, (y2-y2_init)/h_init]
        delta_rgb_bbox = rgb_conv_reg(rgb_feats_reg)    
        obj_rgb = rgb_conv_obj(rgb_feats_reg)
        delta_tir_bbox = tir_conv_reg(tir_feats_reg)
        obj_tir = tir_conv_obj(tir_feats_reg)
        if self.use_cls_branch:
            cls_rgb = rgb_conv_cls(rgb_feats_cls)
            cls_tir = tir_conv_cls(tir_feats_cls)
        
        assert delta_rgb_bbox.shape == pre_bbox_init.permute(0,3,1,2).shape == delta_tir_bbox.shape
      
        return  delta_rgb_bbox, delta_tir_bbox, bbox_pred_init, \
                obj_rgb, obj_tir, obj_init, \
                cls_rgb if self.use_cls_branch else None, \
                cls_tir if self.use_cls_branch else None

    def forward(self, feats, unique_feats):
        if self.use_cls_branch:
            return multi_apply(self.forward_single, feats, unique_feats,
                            self.strides,
                            self.rgb_unique_fusion_reg,
                            self.tir_unique_fusion_reg,
                            self.union_multi_level_reg_convs,
                            self.rgb_multi_level_reg_convs,
                            self.tir_multi_level_reg_convs,
                            self.union_multi_level_conv_reg,
                            self.rgb_multi_level_conv_reg,
                            self.tir_multi_level_conv_reg,
                            self.union_multi_level_conv_obj,
                            self.rgb_multi_level_conv_obj,
                            self.tir_multi_level_conv_obj,
                            self.rgb_unique_fusion_cls,
                            self.tir_unique_fusion_cls,
                            self.union_multi_level_cls_convs,
                            self.rgb_multi_level_cls_convs,
                            self.tir_multi_level_cls_convs,
                            self.rgb_multi_level_conv_cls,
                            self.tir_multi_level_conv_cls)
        else:
            return multi_apply(self.forward_single, feats, unique_feats,
                            self.strides,
                            self.rgb_unique_fusion_reg,
                            self.tir_unique_fusion_reg,
                            self.union_multi_level_reg_convs,
                            self.rgb_multi_level_reg_convs,
                            self.tir_multi_level_reg_convs,
                            self.union_multi_level_conv_reg,
                            self.rgb_multi_level_conv_reg,
                            self.tir_multi_level_conv_reg,
                            self.union_multi_level_conv_obj,
                            self.rgb_multi_level_conv_obj,
                            self.tir_multi_level_conv_obj)

    
    def forward_train(self, x, img_metas, people_num, **supervision):
        
        feats, unique_feats = x
        outs = self(feats, unique_feats)
        assert len(outs) == 8      

        losses = self.loss(*outs, img_metas, people_num, **supervision)
        
        return losses

    @force_fp32(apply_to=fp32_apply_cls)
    def loss(self,
             rgb_bbox_delta,
             tir_bbox_delta,
             union_bbox_pred,
             obj_rgb,
             obj_tir,
             obj_union,
             cls_rgb,
             cls_tir,
             img_metas,
             people_num,
             **supervision):
        if not self.use_cls_branch:
            assert not cls_rgb[0] and not cls_tir[0]
        
        gt_bboxes_rgb = supervision['gt_bboxes_rgb']
        gt_bboxes_tir = supervision['gt_bboxes_tir']
        gt_bboxes_union = supervision['gt_bboxes_union']
        local_person_ids_rgb = supervision['local_person_ids_rgb']
        local_person_ids_tir = supervision['local_person_ids_tir']
        local_person_ids_union = supervision['local_person_ids_union']
        if self.use_cls_branch:
            gt_labels_rgb = supervision['gt_labels_rgb']
            gt_labels_tir = supervision['gt_labels_tir']
            gt_labels_union = supervision['gt_labels_union']
        
        N = len(img_metas)
        featmap_sizes = [bbox.shape[2:] for bbox in rgb_bbox_delta]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=obj_rgb[0].dtype,
            device=obj_rgb[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)

        flatten_rgb_delta_multi_stage = \
            [rgb_delta.permute(0,2,3,1).reshape(N,-1,4) for rgb_delta in rgb_bbox_delta]
        flatten_tir_delta_multi_stage = \
            [tir_delta.permute(0,2,3,1).reshape(N,-1,4) for tir_delta in tir_bbox_delta]
        flatten_union_bbox_pred_multi_stage = \
            [union_bbox.permute(0,2,3,1).reshape(N,-1,4) for union_bbox in union_bbox_pred]
        flatten_rgb_delta = torch.cat(flatten_rgb_delta_multi_stage, dim=1)
        flatten_tir_delta = torch.cat(flatten_tir_delta_multi_stage, dim=1)
        flatten_union_bboxes_pred = torch.cat(flatten_union_bbox_pred_multi_stage, dim=1)
        
        flatten_union_bboxes = self._bbox_decode(flatten_priors, flatten_union_bboxes_pred)
        flatten_rgb_bboxes = self.get_refined_bbox(flatten_union_bboxes.detach(), flatten_rgb_delta)
        flatten_tir_bboxes = self.get_refined_bbox(flatten_union_bboxes.detach(), flatten_tir_delta)

        flatten_obj_rgb_multi_stage = [_obj_rgb.permute(0,2,3,1).reshape(N,-1) for _obj_rgb in obj_rgb]
        flatten_obj_tir_multi_stage = [_obj_tir.permute(0,2,3,1).reshape(N,-1) for _obj_tir in obj_tir]
        flatten_obj_union_multi_stage = [_obj_union.permute(0,2,3,1).reshape(N,-1) for _obj_union in obj_union]
        flatten_obj_rgb = torch.cat(flatten_obj_rgb_multi_stage, dim=1)
        flatten_obj_tir = torch.cat(flatten_obj_tir_multi_stage, dim=1)
        flatten_obj_union = torch.cat(flatten_obj_union_multi_stage, dim=1)

        if self.use_cls_branch:
            flatten_cls_rgb_multi_stage = [_cls_rgb.permute(0,2,3,1).reshape(N,-1) for _cls_rgb in cls_rgb]
            flatten_cls_tir_multi_stage = [_cls_tir.permute(0,2,3,1).reshape(N,-1) for _cls_tir in cls_tir]
            flatten_cls_rgb = torch.cat(flatten_cls_rgb_multi_stage, dim=1)
            flatten_cls_tir = torch.cat(flatten_cls_tir_multi_stage, dim=1)
        else:
            flatten_cls_rgb = torch.ones_like(flatten_obj_rgb)
            flatten_cls_tir = torch.ones_like(flatten_obj_tir)
        flatten_cls_union = torch.ones_like(flatten_obj_union)

        _union_target, _rgb_target, _tir_target = \
                multi_apply(self._get_target_single, 
                        flatten_cls_union,
                        flatten_obj_union.detach(),
                        flatten_priors.unsqueeze(0).repeat(N, 1, 1),
                        flatten_union_bboxes.detach(),
                        gt_bboxes_union,
                        gt_labels_union if self.use_cls_branch else None,
                        flatten_rgb_delta.detach(),
                        flatten_tir_delta.detach(),
                        local_person_ids_rgb,
                        local_person_ids_tir,
                        local_person_ids_union,
                        gt_bboxes_rgb,
                        gt_bboxes_tir)
        
        '''get union targets'''
        union_target = []
        for item_1, item_2 in zip(*_union_target):
            union_target.append(torch.cat((item_1, item_2), 0))
        
        assert len(union_target) == 6        
        union_pos_masks, union_cls_targets, union_obj_targets,\
            union_bbox_targets, union_l1_targets, union_num_fg_imgs = union_target
        
        union_num_pos = torch.tensor(
            union_num_fg_imgs.sum(0).item(),
            dtype=torch.float,
            device=flatten_obj_union.device)
        union_num_total_samples = max(reduce_mean(union_num_pos), 1.0)
        
        del _union_target, union_target, union_num_fg_imgs
        if not self.use_l1: del union_l1_targets

        '''get rgb targets'''
        rgb_target = []
        for item_1, item_2 in zip(*_rgb_target):
            rgb_target.append(torch.cat((item_1, item_2), 0))

        assert len(rgb_target) == 6        
        rgb_pos_masks, rgb_cls_targets, rgb_obj_targets,\
            rgb_bbox_targets, rgb_l1_targets, rgb_num_fg_imgs = rgb_target

        rgb_num_pos = torch.tensor(
            rgb_num_fg_imgs.sum(0).item(),
            dtype=torch.float,
            device=flatten_obj_rgb.device)
        rgb_num_total_samples = max(reduce_mean(rgb_num_pos), 1.0)

        del _rgb_target, rgb_target, rgb_num_fg_imgs
        if not self.use_l1: del rgb_l1_targets

        '''get tir targets'''
        tir_target = []
        for item_1, item_2 in zip(*_tir_target):
            tir_target.append(torch.cat((item_1, item_2), 0))

        assert len(tir_target) == 6        
        tir_pos_masks, tir_cls_targets, tir_obj_targets,\
            tir_bbox_targets, tir_l1_targets, tir_num_fg_imgs = tir_target

        tir_num_pos = torch.tensor(
            tir_num_fg_imgs.sum(0).item(),
            dtype=torch.float,
            device=flatten_obj_tir.device)
        tir_num_total_samples = max(reduce_mean(tir_num_pos), 1.0)

        del _tir_target, tir_target, tir_num_fg_imgs
        if not self.use_l1: del tir_l1_targets

        '''caculate loss'''
        loss_bbox_union = self.loss_bbox(
            flatten_union_bboxes.view(-1, 4)[union_pos_masks], union_bbox_targets)\
                / union_num_total_samples
        loss_obj_union = self.loss_obj(flatten_obj_union.view(-1, 1), union_obj_targets)\
                / union_num_total_samples

        loss_bbox_rgb = self.loss_bbox(
            flatten_rgb_bboxes.view(-1, 4)[rgb_pos_masks], rgb_bbox_targets)\
                / rgb_num_total_samples
        loss_obj_rgb = self.loss_obj(flatten_obj_rgb.view(-1, 1), rgb_obj_targets)\
                / rgb_num_total_samples

        loss_bbox_tir = self.loss_bbox(
            flatten_tir_bboxes.view(-1, 4)[tir_pos_masks], tir_bbox_targets)\
                / tir_num_total_samples
        loss_obj_tir = self.loss_obj(flatten_obj_tir.view(-1, 1), tir_obj_targets)\
                / tir_num_total_samples

        loss_dict = dict(
            loss_bbox_union=loss_bbox_union,
            loss_obj_union=loss_obj_union,
            loss_bbox_rgb=loss_bbox_rgb,
            loss_obj_rgb=loss_obj_rgb,
            loss_bbox_tir=loss_bbox_tir,
            loss_obj_tir=loss_obj_tir)

        if self.use_cls_branch:
            loss_cls_rgb = self.loss_cls(
                flatten_cls_rgb.view(-1, self.num_classes)[rgb_pos_masks], rgb_cls_targets)\
                    / rgb_num_total_samples
            loss_cls_tir = self.loss_cls(
                flatten_cls_tir.view(-1, self.num_classes)[tir_pos_masks], tir_cls_targets)\
                    / tir_num_total_samples

            loss_dict.update(loss_cls_rgb=loss_cls_rgb, loss_cls_tir=loss_cls_tir)
        
        if self.use_l1:
            loss_l1_union = self.loss_l1(
                flatten_union_bboxes_pred.view(-1, 4)[union_pos_masks],
                union_l1_targets) / union_num_total_samples
            loss_l1_rgb = self.loss_l1(
                flatten_rgb_delta.view(-1, 4)[rgb_pos_masks],
                rgb_l1_targets) / rgb_num_total_samples
            loss_l1_tir = self.loss_l1(
                flatten_tir_delta.view(-1, 4)[tir_pos_masks],
                tir_l1_targets) / tir_num_total_samples

            loss_dict.update(loss_l1_union=loss_l1_union,
                             loss_l1_rgb=loss_l1_rgb,
                             loss_l1_tir=loss_l1_tir)
        
        return loss_dict
        
        
    @torch.no_grad()
    def _get_target_single(self,
                            cls_preds,
                            objectness,
                            priors,
                            decoded_bboxes,
                            gt_bboxes,
                            gt_labels,
                            decoded_rgb_delta,
                            decoded_tir_delta,
                            person_id_rgb,
                            person_id_tir,
                            person_id_union,
                            gt_bboxes_rgb,
                            gt_bboxes_tir):
        num_priors = priors.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        
        if not self.use_cls_branch:
            gt_labels = torch.zeros(num_gts, dtype=torch.int64, device=gt_bboxes.device)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            _out =  (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, torch.tensor([0]))
            return _out, _out, _out
        
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.unsqueeze(1) * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        union_num_pos_per_img = pos_inds.size(0)

        if self.use_cls_branch:
            pos_ious = assign_result.max_overlaps[pos_inds]
            union_cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) \
                            * pos_ious.unsqueeze(-1)
        else:
            union_cls_target = cls_preds.new_zeros((0, self.num_classes))
        union_obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        union_obj_target[pos_inds] = 1
        union_bbox_target = sampling_result.pos_gt_bboxes

        union_l1_target = cls_preds.new_zeros((union_num_pos_per_img, 4))
        if self.use_l1:
            # code bbox for pos_anchors
            union_l1_target = self._get_l1_target(union_l1_target, union_bbox_target, priors[pos_inds])
        union_foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        union_foreground_mask[pos_inds] = 1

        union_gt_inds = sampling_result.pos_assigned_gt_inds
        union_assigned_person = person_id_union[union_gt_inds]
        
        '''get target for rgb'''
        valid_mask_rgb = self.get_valid_inds(union_assigned_person, person_id_union, person_id_rgb)
        pos_inds_rgb = pos_inds[valid_mask_rgb]
        rgb_num_pos_per_img = pos_inds_rgb.size(0)

        rgb_obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        rgb_obj_target[pos_inds_rgb] = 1
        
        rgb_assigned_person = union_assigned_person[valid_mask_rgb]
        rgb_bbox_target = self.get_bboxes_by_id(rgb_assigned_person, gt_bboxes_rgb, person_id_rgb)
        if self.use_cls_branch:
            decoded_rgb_bboxes = self.get_refined_bbox(decoded_bboxes, decoded_rgb_delta)
            pos_ious_rgb = bbox_overlaps(decoded_rgb_bboxes[pos_inds_rgb], rgb_bbox_target, is_aligned=True)
            rgb_cls_target = F.one_hot(gt_labels[rgb_assigned_person], self.num_classes) \
                            * pos_ious_rgb.unsqueeze(-1)
        else:
            rgb_cls_target = cls_preds.new_zeros((0, self.num_classes))
        
        rgb_l1_target = cls_preds.new_zeros((rgb_num_pos_per_img, 4))
        if self.use_l1:
            rgb_l1_target = self.l1_refine_target(rgb_l1_target, rgb_bbox_target, decoded_bboxes[pos_inds_rgb])
        
        rgb_foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        rgb_foreground_mask[pos_inds_rgb] = 1

        '''get target for tir'''
        valid_mask_tir = self.get_valid_inds(union_assigned_person, person_id_union, person_id_tir)
        pos_inds_tir = pos_inds[valid_mask_tir]
        tir_num_pos_per_img = pos_inds_tir.size(0)

        tir_obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        tir_obj_target[pos_inds_tir] = 1

        tir_assigned_person = union_assigned_person[valid_mask_tir]
        tir_bbox_target = self.get_bboxes_by_id(tir_assigned_person, gt_bboxes_tir, person_id_tir)
        if self.use_cls_branch:
            decoded_tir_bboxes = self.get_refined_bbox(decoded_bboxes, decoded_tir_delta)
            pos_ious_tir = bbox_overlaps(decoded_tir_bboxes[pos_inds_tir], tir_bbox_target, is_aligned=True)
            tir_cls_target = F.one_hot(gt_labels[tir_assigned_person], self.num_classes) \
                            * pos_ious_tir.unsqueeze(-1)
        else:
            tir_cls_target = cls_preds.new_zeros((0, self.num_classes))

        tir_l1_target = cls_preds.new_zeros((tir_num_pos_per_img, 4))
        if self.use_l1:
            tir_l1_target = self.l1_refine_target(tir_l1_target, tir_bbox_target, decoded_bboxes[pos_inds_tir])
        
        tir_foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        tir_foreground_mask[pos_inds_tir] = 1

        union=(union_foreground_mask, union_cls_target, union_obj_target, union_bbox_target,
                union_l1_target, torch.tensor([union_num_pos_per_img]))
        rgb=(rgb_foreground_mask, rgb_cls_target, rgb_obj_target, rgb_bbox_target,
                rgb_l1_target, torch.tensor([rgb_num_pos_per_img]))
        tir=(tir_foreground_mask, tir_cls_target, tir_obj_target, tir_bbox_target,
                tir_l1_target, torch.tensor([tir_num_pos_per_img]))

        return union, rgb, tir

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target

    def l1_refine_target(self, l1_target, gt_bboxes, priors):
        '''priors: union_decoded bboxes'''
        wh_priors = priors[:,2:] - priors[:,:2]
        wh_priors = torch.cat([wh_priors, wh_priors], dim=1)
        l1_target = (gt_bboxes - priors) / wh_priors
        return l1_target

    def get_valid_inds(self, union_assigned_person, person_id_union, _person_id):
        '''get invalid anchor for unpaired bbox in rgb/tir'''
        diff_set = set(person_id_union.tolist()) - set(_person_id.tolist())
        valid_mask = torch.ones_like(union_assigned_person, dtype=torch.bool)
        for id in diff_set:
            invalid_id = torch.tensor([id], device=union_assigned_person.device)
            valid_mask &= invalid_id != union_assigned_person                   
        return valid_mask

    def get_bboxes_by_id(self, pos_person_id, gt_bboxes, local_person_ids):
        '''get relevant bbox by filtered person_id'''
        if local_person_ids.shape[0] == 0:
            return gt_bboxes.new_full((0, 4), -1)
        max_id = local_person_ids.max()
        id_relev_bboxes = gt_bboxes.new_full((max_id + 1, 4), -1)
        id_relev_bboxes[local_person_ids, :] = gt_bboxes
        pos_bboxes = id_relev_bboxes[pos_person_id, :]
        return pos_bboxes
    
    def simple_test(self, feats, img_metas, rescale):
        outs = self(*feats)
        assert len(outs) == 8
        results_list = self.get_bboxes(*outs, img_metas=img_metas, rescale=rescale)

        return results_list

    def get_bboxes(self,
                   rgb_bbox_delta,
                   tir_bbox_delta,
                   union_bbox_pred,
                   obj_rgb,
                   obj_tir,
                   obj_union,
                   cls_rgb,
                   cls_tir,
                   img_metas,
                   rescale):
        assert len(rgb_bbox_delta) == len(tir_bbox_delta) == len(union_bbox_pred)\
            == len(obj_rgb) == len(obj_tir) == len(obj_union)
        if self.use_cls_branch:
            assert len(cls_rgb) == len(cls_tir) == len(obj_union)
        else:
            assert not cls_rgb[0] and not cls_tir[0]

        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        N = len(img_metas)
        featmap_sizes = [bbox.shape[2:] for bbox in rgb_bbox_delta]
        
        '''decode bboxes'''
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=obj_rgb[0].dtype,
            device=obj_rgb[0].device,
            with_stride=True)
        flatten_priors = torch.cat(mlvl_priors)  

        flatten_rgb_delta_multi_stage = \
            [rgb_delta.permute(0,2,3,1).reshape(N,-1,4) for rgb_delta in rgb_bbox_delta]
        flatten_tir_delta_multi_stage = \
            [tir_delta.permute(0,2,3,1).reshape(N,-1,4) for tir_delta in tir_bbox_delta]
        flatten_union_bbox_pred_multi_stage = \
            [union_bbox.permute(0,2,3,1).reshape(N,-1,4) for union_bbox in union_bbox_pred]
        flatten_rgb_delta = torch.cat(flatten_rgb_delta_multi_stage, dim=1)
        flatten_tir_delta = torch.cat(flatten_tir_delta_multi_stage, dim=1)
        flatten_union_bboxes_pred = torch.cat(flatten_union_bbox_pred_multi_stage, dim=1)

        flatten_union_bboxes = self._bbox_decode(flatten_priors, flatten_union_bboxes_pred)
        flatten_rgb_bboxes = self.get_refined_bbox(flatten_union_bboxes, flatten_rgb_delta)
        flatten_tir_bboxes = self.get_refined_bbox(flatten_union_bboxes, flatten_tir_delta)

        '''objectness'''
        flatten_obj_rgb_multi_stage = [_obj_rgb.permute(0,2,3,1).reshape(N,-1) for _obj_rgb in obj_rgb]
        flatten_obj_tir_multi_stage = [_obj_tir.permute(0,2,3,1).reshape(N,-1) for _obj_tir in obj_tir]
        flatten_obj_union_multi_stage = [_obj_union.permute(0,2,3,1).reshape(N,-1) for _obj_union in obj_union]
        flatten_obj_rgb = torch.cat(flatten_obj_rgb_multi_stage, dim=1).sigmoid()
        flatten_obj_tir = torch.cat(flatten_obj_tir_multi_stage, dim=1).sigmoid()
        flatten_obj_union = torch.cat(flatten_obj_union_multi_stage, dim=1).sigmoid()

        '''cls'''
        if self.use_cls_branch:
            flatten_cls_rgb_multi_stage = [_cls_rgb.permute(0,2,3,1).reshape(N,-1) for _cls_rgb in cls_rgb]
            flatten_cls_tir_multi_stage = [_cls_tir.permute(0,2,3,1).reshape(N,-1) for _cls_tir in cls_tir]
            flatten_cls_rgb = torch.cat(flatten_cls_rgb_multi_stage, dim=1).sigmoid()
            flatten_cls_tir = torch.cat(flatten_cls_tir_multi_stage, dim=1).sigmoid()
        else:
            flatten_cls_rgb = torch.ones_like(flatten_obj_rgb)
            flatten_cls_tir = torch.ones_like(flatten_obj_tir)

        if rescale:
            flatten_rgb_bboxes[..., :4] /= flatten_rgb_bboxes.new_tensor(scale_factors).unsqueeze(1)
            flatten_tir_bboxes[..., :4] /= flatten_tir_bboxes.new_tensor(scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(N):
            rgb_cls_scores = flatten_cls_rgb[img_id]
            tir_cls_scores = flatten_cls_tir[img_id]
            rgb_score_factor = flatten_obj_rgb[img_id]
            tir_score_factor = flatten_obj_tir[img_id]
            rgb_bboxes = flatten_rgb_bboxes[img_id]
            tir_bboxes = flatten_tir_bboxes[img_id]

            result_list.append(
                self.pair_bboxes_nms(rgb_cls_scores,
                                     tir_cls_scores,
                                     rgb_bboxes,
                                     tir_bboxes,
                                     rgb_score_factor,
                                     tir_score_factor,
                                     self.test_cfg))

        return result_list

    def pair_bboxes_nms(self,
                        rgb_cls_scores,
                        tir_cls_scores,
                        rgb_bboxes,
                        tir_bboxes,
                        rgb_score_factor,
                        tir_score_factor,
                        cfg):
        score_factor = torch.stack([rgb_score_factor, tir_score_factor], dim=-1)
        cls_scores = torch.stack([rgb_cls_scores, tir_cls_scores], dim=-1)
        
        max_obj, idx = score_factor.max(dim=1)
        idx = idx.unsqueeze(-1)
        cls_mask = cls_scores.new_full(cls_scores.shape[:2], 0, dtype=torch.bool)
        cls_mask.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
        max_obj_cls = cls_scores[cls_mask]
        valid_mask = max_obj * max_obj_cls >= cfg.score_thr

        rgb_bboxes = rgb_bboxes[valid_mask]
        tir_bboxes = tir_bboxes[valid_mask]
        
        modal_scores = score_factor[valid_mask] * cls_scores[valid_mask]
        anchor_scores = max_obj[valid_mask] * max_obj_cls[valid_mask]
        
        bbox_weight = score_factor[valid_mask]
        weight_mask = bbox_weight < cfg.score_thr
        bbox_weight[weight_mask] = 0
        bbox_weight = bbox_weight / bbox_weight.sum(1).unsqueeze(-1)
        assert len(rgb_bboxes) == len(tir_bboxes) == len(bbox_weight)
        anchor_bboxes = rgb_bboxes * bbox_weight[:,:1] + tir_bboxes * bbox_weight[:,1:]
        
        # have only one class
        labels = anchor_bboxes.new_full((len(anchor_bboxes),), 0, dtype=torch.int64)

        if anchor_scores.numel() == 0:
            return rgb_bboxes,\
                   tir_bboxes,\
                   rgb_bboxes.new_full((len(rgb_bboxes), ), 0, dtype=torch.int64),\
                   tir_bboxes.new_full((len(tir_bboxes), ), 0, dtype=torch.int64),\
                   rgb_bboxes.new_full((len(rgb_bboxes), ), 0, dtype=torch.float32),\
                   tir_bboxes.new_full((len(tir_bboxes), ), 0, dtype=torch.float32),\
                   labels,\
                   labels,\
                   anchor_scores
        else:
            _, keep = batched_nms(anchor_bboxes, anchor_scores, labels, cfg.nms)
            person_ids = torch.arange(len(keep))
            bbox_mask = ~weight_mask[keep]
            rgb_bboxes = rgb_bboxes[keep]
            tir_bboxes = tir_bboxes[keep]
            anchor_scores = anchor_scores[keep]
            modal_scores = modal_scores[keep]
            labels = labels[keep]

            assert len(anchor_scores) == len(person_ids) == len(labels)
            rgb_mask = bbox_mask[:, 0]
            tir_mask = bbox_mask[:, 1]
            assert len(rgb_bboxes) == len(rgb_mask)
            rgb_bboxes = rgb_bboxes[rgb_mask]
            tir_bboxes = tir_bboxes[tir_mask]
            rgb_ids = person_ids[rgb_mask]
            tir_ids = person_ids[tir_mask]
            rgb_scores = modal_scores[rgb_mask, 0]
            tir_scores = modal_scores[tir_mask, 1]
            rgb_labels = labels[rgb_mask]
            tir_labels = labels[tir_mask]
            assert len(rgb_labels) == len(rgb_bboxes) == len(rgb_scores) == len(rgb_ids)
            assert len(tir_labels) == len(tir_bboxes) == len(tir_scores) == len(tir_ids)
            
            return rgb_bboxes, tir_bboxes, rgb_ids, tir_ids, rgb_scores, tir_scores,\
                   rgb_labels, tir_labels, anchor_scores

