# Copyright (c) OpenMMLab. All rights reserved.
from tkinter import OUTSIDE
from turtle import shape
from ..builder import DETECTORS, build_backbone, build_loss,build_neck
from .single_stage import SingleStageDetector
from copy import deepcopy
from mmdet.core import bbox2result
import torch
import numpy as np
import re
import mmcv
import matplotlib.pyplot as plt
import cv2


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
        if 'plugins' in backbone:
            bbox_head.update(has_unique=True)
        else:
            bbox_head.update(has_unique=False)
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
                if state_dict[name].shape == original_model[origin_name].shape:
                    state_dict[name] = original_model[origin_name]
      
        self.load_state_dict(state_dict)
        print('init weight done')
    
    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
    

    def forward_train(self, img, img_metas, **kwargs):   
        x = self.extract_feat(img)  #8 16 32, 64
        
        losses = self.bbox_head.forward_train(x, img_metas, **kwargs)
        return losses
    

    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        feats, unique_feats = self.backbone(img)  # (([rgb_1, tir_1, pub_1],...), ([u_rgb_1, u_tir_1],...))
        feat_rgb = [feat[0] for feat in feats]
        feat_lwir = [feat[1] for feat in feats]
        feat_pub = [feat[2] for feat in feats]
        
        feat_out = self.feature_fusion_module(feat_rgb, feat_pub, feat_lwir)
        
        if self.with_neck:
            x = self.neck(feat_out)
        return x, unique_feats

    def simple_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [
            self.bbox2result(*outs, self.bbox_head.num_classes) for outs in results_list]
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
            self.bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
    
    def bbox2result(self,
                    rgb_bboxes,
                    tir_bboxes,
                    rgb_ids, 
                    tir_ids, 
                    rgb_scores, 
                    tir_scores, 
                    rgb_labels,
                    tir_labels,
                    union_bboxes,
                    anchor_scores, 
                    num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:

            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if anchor_scores.shape[0] == 0:
            return [np.zeros((0, 4), dtype=np.float32) for i in range(num_classes)],\
                   [np.zeros((0, 4), dtype=np.float32) for i in range(num_classes)],\
                   [np.zeros((0,), dtype=np.int64) for i in range(num_classes)],\
                   [np.zeros((0,), dtype=np.int64) for i in range(num_classes)],\
                   [np.zeros((0,), dtype=np.float32) for i in range(num_classes)],\
                   [np.zeros((0,), dtype=np.float32) for i in range(num_classes)],\
                   [np.zeros((0, 4), dtype=np.float32) for i in range(num_classes)],\
                   [np.zeros((0,), dtype=np.float32) for i in range(num_classes)]
        else:
            if isinstance(rgb_bboxes, torch.Tensor):
                f = lambda x: x.detach().cpu().numpy()
                rgb_bboxes, tir_bboxes, rgb_ids, tir_ids,\
                rgb_scores, tir_scores,\
                rgb_labels, tir_labels,\
                union_bboxes, anchor_scores = list(map(f, (rgb_bboxes, tir_bboxes, rgb_ids, tir_ids,\
                                                rgb_scores, tir_scores,\
                                                rgb_labels, tir_labels,\
                                                union_bboxes, anchor_scores)))

                anchor_labels = np.zeros_like(anchor_scores, dtype=rgb_labels.dtype)
                for j in range(len(rgb_ids)):
                    anchor_labels[rgb_ids[j]] = rgb_labels[j]
                for j in range(len(tir_ids)):
                    anchor_labels[tir_ids[j]] = tir_labels[j]

            return [rgb_bboxes[rgb_labels == i, :] for i in range(num_classes)],\
                   [tir_bboxes[tir_labels == i, :] for i in range(num_classes)],\
                   [rgb_ids[rgb_labels == i] for i in range(num_classes)],\
                   [tir_ids[tir_labels == i] for i in range(num_classes)],\
                   [rgb_scores[rgb_labels == i] for i in range(num_classes)],\
                   [tir_scores[tir_labels == i] for i in range(num_classes)],\
                   [union_bboxes[anchor_labels == i] for i in range(num_classes)],\
                   [anchor_scores[anchor_labels == i] for i in range(num_classes)]

    def show_result(self,
                    img:list,
                    result:tuple,
                    score_thr=0,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=16,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):

        img_rgb, img_tir = img.copy()
        
        rgb_bbox, tir_bbox = result[:2]
        union_bbox = result[-2]
        
        rgb_labels = np.concatenate([
            np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(rgb_bbox)])
        tir_labels = np.concatenate([
            np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(tir_bbox)])
        union_labels = np.concatenate([
            np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(union_bbox)])

        f = lambda x: np.concatenate(x, axis=0)
        rgb_bboxes, tir_bboxes, rgb_ids, tir_ids, rgb_scores, tir_scores, union_bboxes, anchor_scores \
            = list(map(f, result))   

        assert len(rgb_bboxes) == len(rgb_ids) == len(rgb_scores) == len(rgb_labels) and \
               len(tir_bboxes) == len(tir_ids) == len(tir_scores) == len(tir_labels) and \
               len(union_bboxes) == len(anchor_scores)
        union_ids = np.array([i for i in range(len(union_bboxes))], dtype=rgb_ids.dtype)

        # use "labels" key_word to mark person ids
        class_names = [str(i) for i in range(len(anchor_scores))]          
        
        # if out_file specified, do not show image in window

        # draw bounding boxes
        rgb_out = imshow_det_bboxes(
            img_rgb,
            rgb_bboxes,
            scores=rgb_scores,
            person_id=rgb_ids,
            labels=rgb_labels,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=(0,255,0),
            text_color=(0,255,0),
            # bbox_color=(255,69,0),
            # text_color=(255,69,0),
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            wait_time=wait_time)

        # rgb_out = imshow_det_bboxes(
        #     rgb_out,
        #     union_bboxes,
        #     scores=anchor_scores,
        #     person_id=union_ids,
        #     labels=union_labels,
        #     class_names=class_names,
        #     score_thr=score_thr,
        #     bbox_color=(255,255,0),
        #     text_color=(255,255,0),
        #     thickness=thickness,
        #     font_size=font_size,
        #     win_name=win_name,
        #     wait_time=wait_time) 

        tir_out = imshow_det_bboxes(
            img_tir,
            tir_bboxes,
            scores=tir_scores,
            person_id=tir_ids,
            labels=tir_labels,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=(0,255,0),
            text_color=(0,255,0),
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            wait_time=wait_time)

        # tir_out = imshow_det_bboxes(
        #     tir_out,
        #     union_bboxes,
        #     scores=anchor_scores,
        #     person_id=union_ids,
        #     labels=union_labels,
        #     class_names=class_names,
        #     score_thr=score_thr,
        #     bbox_color=(255,255,0),
        #     text_color=(255,255,0),
        #     thickness=thickness,
        #     font_size=font_size,
        #     win_name=win_name,
        #     wait_time=wait_time)

        img = np.concatenate((rgb_out, tir_out), axis=1)

        if not (show or out_file):
            return img

        mmcv.imwrite(rgb_out, out_file[:-4]+'_rgb'+out_file[-4:])
        mmcv.imwrite(tir_out, out_file[:-4]+'_tir'+out_file[-4:])


EPS = 1e-2
def imshow_det_bboxes(img,
                      bboxes,
                      scores,
                      person_id,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=2,
                      font_size=8,
                      win_name='',
                      show=False,
                      wait_time=0,
                      out_file=None):
    from mmdet.core.visualization.palette import get_palette, palette_val
    from mmdet.core.visualization.image import _get_adaptive_scales, draw_bboxes, draw_labels
    
    assert len(bboxes) == len(person_id) == len(scores) == len(labels)
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 4
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        person_id = person_id[inds]
        scores = scores[inds]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness

        person_id = np.zeros_like(person_id, dtype=np.int64)
        class_names = ['person']
        scores *= 100.

        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        # positions = bboxes[:, :2].astype(np.int32) - \
        #             np.repeat(np.array([[0,font_size*1.5]], dtype=np.int32), bboxes.shape[0], axis=0) *\
        #             scales.reshape(scales.shape[0], 1)
        
        draw_labels(
            ax,
            person_id,
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)


    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img