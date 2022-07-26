# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS, build_backbone, build_loss,build_neck
from .single_stage import SingleStageDetector
from copy import deepcopy
from mmdet.core import bbox2result
import torch




@DETECTORS.register_module()
class RGBT_Det(SingleStageDetector):
    """my own baseline"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 pub_feat_module,
                 feature_fusion_module,
                 streams=['rgb', 'lwir', 'pub'],
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_ill_module=False,
                 illum_aware_module=None,
                 init_cfg=None):
        super(RGBT_Det, self).__init__(backbone, neck, bbox_head, 
                                   train_cfg, test_cfg, pretrained, init_cfg)
        # rgb backbone
        if 'rgb' in streams:
            self.backbone = build_backbone(backbone)
        # lwir backbone
        if 'lwir' in streams:
            backbone_lwir = deepcopy(backbone)
            self.backbone_lwir = build_backbone(backbone_lwir)
        # public backbone
        # TODO:backbone_pub construct
        if 'pub' in streams:
            self.backbone_pub = build_backbone(pub_feat_module)
        
        self.feature_fusion_module = build_backbone(feature_fusion_module)

        # neck
        self.neck = build_neck(neck)

        self.streams = streams
        # init weights from original pth
        if init_cfg and init_cfg['type'] == 'Pretrained':
            self.init_from_original_weights()
        # 光照感知模块
        self.use_ill_module = use_ill_module
        if self.use_ill_module:
            self.weight_rgb = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
            self.weight_lwir = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
            self.weight_pub = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
            assert illum_aware_module is not None
            self.illum_aware_module = build_backbone(illum_aware_module)
            self.illumination = None    # 占位符，保存当前光照值
            self.loss_ce = build_loss(dict(
                type='CrossEntropyLoss', loss_weight=1.0
            ))
        else:
            self.weight_rgb = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.weight_lwir = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
            self.weight_pub = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        # free unused tensor cuda memory
        # torch.cuda.empty_cac
        

    def init_from_original_weights(self):
        original_model = torch.load(self.init_cfg['checkpoint'], map_location=torch.device('cpu'))['state_dict']
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if 'head' in name:
                continue
            if 'neck' in name:
                continue
            if name in original_model:
                assert state_dict[name].shape == original_model[name].shape, "%s mismatch."%name
                state_dict[name] = original_model[name]
            elif 'backbone_lwir' in name:
                assert state_dict[name].shape == original_model[name.replace('backbone_lwir','backbone')].shape
                state_dict[name] = original_model[name.replace('backbone_lwir','backbone')]
            elif 'backbone_pub' in name:
                # 适用yolox
                if name == 'backbone_pub.stem.conv.conv.weight':
                    state_dict[name][:, :12, :, :] = original_model[name.replace('backbone_pub','backbone')]
                    state_dict[name][:, 12:, :, :] = original_model[name.replace('backbone_pub','backbone')]
                else:
                    state_dict[name] = original_model[name.replace('backbone_pub','backbone')]
            else:
                continue
                # print('%s not in original models.'%(name))
        self.load_state_dict(state_dict)
    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_illumination=None):
        
        x = self.extract_feat(img)  #8 16 32, 64
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                            gt_labels, gt_bboxes_ignore)
        if self.use_ill_module and gt_illumination is not None:
            loss_illu = self.loss_ce(self.illumination, gt_illumination)
            losses['ill_bce_loss'] = loss_illu
        return losses
    

    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if self.use_ill_module:

            self.illumination = self.illum_aware_module(img[:,:3,:,:])
            illu = self.illumination
            
            weight_rgb = (illu[:,0:1,None,None] + 0.5) * self.weight_rgb
            weight_lwir = (1.5 - illu[:, 1:,None,None]) * self.weight_lwir
        else:
            weight_rgb = self.weight_rgb
            weight_lwir = self.weight_lwir
        if 'rgb' in self.streams:
            feat_rgb = self.backbone(img[:, :3, :, :])  # ([batch,channel,height,width], ..., [batch,channel,height,width])
            feat_rgb = [torch.mul(fea, weight_rgb) for fea in feat_rgb]
        
        else:
            feat_rgb = None
        if 'lwir' in self.streams:
            feat_lwir = self.backbone_lwir(img[:, 3:, :, :])
            feat_lwir = [torch.mul(fea, weight_lwir) for fea in feat_lwir]
            
        else:
            feat_lwir = None
        
        if 'pub' in self.streams:
            feat_pub = self.backbone_pub(img)
            feat_pub = [torch.mul(fea, self.weight_pub) for fea in feat_pub]
            
        else:
            feat_pub = None
        
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
    