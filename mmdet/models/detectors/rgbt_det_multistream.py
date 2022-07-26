# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_loss,build_neck
from .single_stage import SingleStageDetector
from copy import deepcopy
from mmdet.core import bbox2result
import torch

import re


@DETECTORS.register_module()
class RGBT_Det_MultiStream(SingleStageDetector):
    """my own baseline"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 feature_fusion_module,
                 streams=['rgb', 'lwir', 'pub'],
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RGBT_Det_MultiStream, self).__init__(backbone, neck, bbox_head, 
                                   train_cfg, test_cfg, pretrained, init_cfg)
        
        self.backbone = build_backbone(backbone)
        
        self.feature_fusion_module = build_backbone(feature_fusion_module)

        # neck
        self.neck = build_neck(neck)

        self.streams = streams
        # init weights from original pth
        if init_cfg and init_cfg['type'] == 'Pretrained':
            self.init_from_original_weights()
        
        
    def init_from_original_weights(self):
        original_model = torch.load(self.init_cfg['checkpoint'], map_location=torch.device('cpu'))['state_dict']
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if 'head' in name[:5]:
                continue
            if 'neck' in name[:5]:
                continue
            
            if 'backbone' in name[:10]:
                if name == 'backbone.stem_s3.conv.conv.weight':
                    state_dict[name] = original_model[name.replace('_s3', '')].repeat(1,2,1,1)
                    continue
                if 'plugin' in name:
                    continue
                replace_str = re.compile('_s\d{1,}')
                origin_name = replace_str.sub('', name)
                state_dict[name] = original_model[origin_name]
      
        self.load_state_dict(state_dict)
        print('init weight done')
    
    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        
        x = self.extract_feat(img)  #8 16 32, 64
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                            gt_labels, gt_bboxes_ignore)
        return losses
    

    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        feats = self.backbone(img)  # [[x_rgb, x_lwir, x_pub],...]
        feat_rgb = [feat[0] for feat in feats]
        feat_lwir = [feat[1] for feat in feats]
        feat_pub = [feat[2] for feat in feats]
        
        feat_out = self.feature_fusion_module(feat_rgb, feat_pub, feat_lwir)
        
        if self.with_neck:
            x = self.neck(feat_out)
        return x

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    