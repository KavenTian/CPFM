# Copyright (c) OpenMMLab. All rights reserved.
from ctypes import Union
import itertools
import logging
from ntpath import join
import os.path as osp
from symbol import continue_stmt
import tempfile
import time
import warnings
from collections import OrderedDict, defaultdict
import datetime, sys
import pdb, traceback
import copy, os, json, shutil
from xml.dom import NotSupportedErr

import cv2, torch
import mmcv
import numpy as np
from mmcv.utils import print_log
from scipy.fftpack import shift
from terminaltables import AsciiTable

from mmdet.core import anchor, eval_recalls
from .api_wrappers import COCO, COCOeval
from pycocotools.cocoeval import Params
from .builder import DATASETS
from .custom import CustomDataset

import scipy
from .brambox import boxes as bbb
from .brambox.boxes.statistics.mr_fppi import mr_fppi


@DATASETS.register_module()
class GneralKaist(CustomDataset):
    CLASSES = ('person',)
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 filter_unpaired_sample:str=None,  # only work in train time
                 choose_unpaired:str=None,         # only work in test time
                 file_client_args=dict(backend='disk'),
                 test_union=True,
                 test_trans_dict: dict=None,
                 test_cvc14=None):
        
        self.filter_unpaired_sample = filter_unpaired_sample 
        self.choose_unpaired = choose_unpaired
        self.test_cvc14 = test_cvc14

        super(GneralKaist, self).__init__(ann_file,
                                            pipeline,
                                            classes,
                                            data_root,
                                            img_prefix,
                                            seg_prefix,
                                            proposal_file,
                                            test_mode,
                                            filter_empty_gt,
                                            file_client_args)
        if test_mode and choose_unpaired:
            valid_inds = self.test_time_filter()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
        
        self.test_union = test_union
        try:
            self.local_rank = torch.distributed.get_rank()
            print(f'Get local rank = {self.local_rank} !')
        except:
            self.local_rank = 0
        
        if test_trans_dict: # build data gt for test
            if self.local_rank == 0:
                if os.path.exists('/workspace/annotations/shift_test/'):
                    shutil.rmtree('/workspace/annotations/shift_test/')
                    print('Remove old \'annotations/shift_test/\'')
                if os.path.exists('/workspace/annotations/shift_test_matlab/'):
                    shutil.rmtree('/workspace/annotations/shift_test_matlab/')
                    print('Remove old \'annotations/shift_test_matlab/\'')
            else:
                time.sleep(1)

            self.test_union = False
            self.width, self.height = 640, 512
            self.alpha = 2.
            self.delta_x = []
            self.delta_y = []
            self.shift_index = []
                       
            assert isinstance(test_trans_dict, dict), 'test_trans_dict must be a dict'
            self.gt_trans_type = test_trans_dict['type']
            
            if self.gt_trans_type == 'homo':
                points_pair = test_trans_dict['points_pair']
                self.build_test_gts('homo', points_pair=points_pair)
            elif self.gt_trans_type == 'shift':
                shifts = test_trans_dict['shifts']
                self.build_test_gts('shift', shifts=shifts)
            elif self.gt_trans_type == 'stay':
                self.build_test_gts('stay')
            else:
                raise TypeError

    
    def build_test_gts(self, ttype:str, **kwargs):
        '''Only make transforms in TIR modal'''
        assert ttype in ['homo', 'shift', 'stay']
        style_name = ''
        if ttype == 'homo':
            points_pair = kwargs['points_pair'] 
            srcp=np.array(points_pair[0], dtype=np.float64).reshape(-1, 1, 2)
            tgtp=np.array(points_pair[1], dtype=np.float64).reshape(-1, 1, 2)
            self.H, _ = cv2.findHomography(srcp, tgtp, 0)
            bram_tir_dir, mat_tir_dir = self.make_txts(tmp_len=20)
        elif ttype == 'stay':
            self.H = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
            bram_tir_dir, mat_tir_dir = self.make_txts(tmp_len=20)
        elif ttype == 'shift':
            shifts = kwargs['shifts']
            f = lambda x: 'P' if x >= 0 else 'N'
            style_name = f'_x_{f(shifts[0])}{shifts[0]}_y_{f(shifts[1])}{shifts[1]}'
            self.H = np.array([[1,0,shifts[0]],[0,1,shifts[1]],[0,0,1]], dtype=np.float64)
            bram_tir_dir, mat_tir_dir = self.make_txts(tmp_len=20)

        # rename tmp dir
        self.asi = sum(self.shift_index) / len(self.shift_index) # average shift index
        self.adx = sum(self.delta_x) / len(self.delta_x)
        self.ady = sum(self.delta_y) / len(self.delta_y)

        self.shift_path_name = f'{ttype}{style_name}_{self.asi:>09.6f}'.replace('.', '_').replace('-', '')
        new_bram_tir_dir = bram_tir_dir[:-20] + 'tir_' + self.shift_path_name
        new_mat_tir_dir = mat_tir_dir[:-20] + 'tir_' + self.shift_path_name
        if self.local_rank == 0:
            if not os.path.exists(new_bram_tir_dir):
                os.rename(bram_tir_dir, new_bram_tir_dir)
            else:
                shutil.rmtree(bram_tir_dir)
            if not os.path.exists(new_mat_tir_dir):
                os.rename(mat_tir_dir, new_mat_tir_dir)
            else:
                shutil.rmtree(mat_tir_dir)
            

        
    def make_txts(self, 
                  ori_gts_path:str='/workspace/id_paired_annotations/',
                  gts_path='/workspace/annotations/shift_test/',
                  matlab_gts_path='/workspace/annotations/shift_test_matlab/',
                  tmp_len=20):
        tmp_dir = ''.join(list(map(str, np.random.randint(10, size=tmp_len).tolist())))
        if self.local_rank == 0:
            os.makedirs(os.path.join(gts_path, tmp_dir))
            if not os.path.exists(os.path.join(matlab_gts_path, tmp_dir)):
                os.makedirs(os.path.join(matlab_gts_path, tmp_dir))
        
        # traversal tir txts
        for set in range(6, 12):
            s = f'set{set:0>2d}'
            videos = os.listdir(os.path.join(ori_gts_path, s))
            for v in videos:
                txts = os.listdir(os.path.join(ori_gts_path, s, v, 'lwir'))
                for t in txts:
                    txt_path = os.path.join(ori_gts_path, s, v, 'lwir', t)
                    paired_txt_path = os.path.join(ori_gts_path, s, v, 'visible', t)
                    text_transformed = self.bbox_transform(txt_path, paired_txt_path)
                    
                    if self.local_rank == 0:
                        # make brambox txts
                        new_txt_dir = os.path.join(gts_path, tmp_dir, s, v)
                        if not os.path.exists(new_txt_dir):
                            os.makedirs(new_txt_dir)
                        with open(os.path.join(new_txt_dir, t), 'w') as f:
                            f.write(text_transformed)

                        # make matlab txts
                        txt_name = '_'.join([s, v, t])
                        self.make_matlab_txts(os.path.join(matlab_gts_path, tmp_dir), txt_name, text_transformed)

        return os.path.join(gts_path, tmp_dir), \
               os.path.join(matlab_gts_path, tmp_dir)


    @staticmethod
    def make_matlab_txts(tir_dir, txt_name, context):
        '''write a matlab txt'''
        # all
        if not os.path.exists(os.path.join(tir_dir, 'test-all')):
            os.makedirs(os.path.join(tir_dir, 'test-all/annotations'))
            os.makedirs(os.path.join(tir_dir, 'test-all/annotations_KAIST_test_set'))      
        
        op = os.path.join(tir_dir, 'test-all/annotations', txt_name)
        with open(op, 'w') as f:
            f.write(context)
        op = os.path.join(tir_dir, 'test-all/annotations_KAIST_test_set', txt_name)
        with open(op, 'w') as f:
            f.write(context)

        # day
        if int(txt_name[3:5]) in [6, 7, 8]:
            if not os.path.exists(os.path.join(tir_dir, 'test-day')):
                os.makedirs(os.path.join(tir_dir, 'test-day/annotations'))
                os.makedirs(os.path.join(tir_dir, 'test-day/annotations_KAIST_test_set'))
            
            op = os.path.join(tir_dir, 'test-day/annotations', txt_name)
            with open(op, 'w') as f:
                f.write(context)
            op = os.path.join(tir_dir, 'test-day/annotations_KAIST_test_set', txt_name)
            with open(op, 'w') as f:
                f.write(context)
        # night
        else:
            if not os.path.exists(os.path.join(tir_dir, 'test-night')):
                os.makedirs(os.path.join(tir_dir, 'test-night/annotations'))
                os.makedirs(os.path.join(tir_dir, 'test-night/annotations_KAIST_test_set'))
            
            op = os.path.join(tir_dir, 'test-night/annotations', txt_name)
            with open(op, 'w') as f:
                f.write(context)
            op = os.path.join(tir_dir, 'test-night/annotations_KAIST_test_set', txt_name)
            with open(op, 'w') as f:
                f.write(context)


    def bbox_transform(self, txt_path:str, paired_txt_path:str) -> str:
        def get_paired_box(_lines:list, _person_id:int)->list:
            for _line in _lines:
                if _line[0] == '%':
                    continue
                _list = _line.split(' ')
                _id = int(_list[-1].rstrip())
                if _id != _person_id:
                    continue
                return list(map(float, _list[1:5])) # [lx, ly, w, h]
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        with open(paired_txt_path, 'r') as f:
            paired_lines = f.readlines()
        
        final_lines = []
        for i, line in enumerate(lines):
            if line[0] == '%':
                final_lines.append(line.rstrip())
                continue
            text_list = line.split(' ')
            _bbox = list(map(float, text_list[1:5])) # [lx, ly, w, h]
            
            bb_points = np.array([[[_bbox[0], _bbox[1]]], [[_bbox[0]+_bbox[2], _bbox[1]+_bbox[3]]]],
                                   dtype=np.float64)
            new_bb_points = cv2.perspectiveTransform(bb_points, self.H)\
                                .astype(np.int).reshape(-1, 2)
            # clip border
            new_bb_points[:, 0] = np.clip(new_bb_points[:, 0], 0, self.width-1)
            new_bb_points[:, 1] = np.clip(new_bb_points[:, 1], 0, self.height-1)
            bbox = new_bb_points.reshape(-1).tolist()
            if abs(bbox[0]-bbox[2])==0 or abs(bbox[1]-bbox[3])==0:
                print(f'Get ignored bbox in {txt_path}, line {i}')
                continue
            bbox = [min(bbox[0],bbox[2]), min(bbox[1],bbox[3]), \
                    abs(bbox[0]-bbox[2]), abs(bbox[1]-bbox[3])] # [lx, ly, w, h]
            text_list[1:5] = list(map(str, bbox))
            person_id = int(text_list[-1].rstrip())
            text_list.pop()
            final_lines.append(' '.join(text_list))

            # get statistics
            paired_box = get_paired_box(paired_lines, person_id)
            # w_max = max(bbox[2], paired_box[2])
            # h_max = max(bbox[3], paired_box[3])
            delta_x = abs(bbox[0] + 0.5*bbox[2] - (paired_box[0] + 0.5*paired_box[2]))  # absolut shifts
            delta_y = abs(bbox[1] + 0.5*bbox[3] - (paired_box[1] + 0.5*paired_box[3]))
            self.delta_x.append(delta_x)
            self.delta_y.append(delta_y)
            delta_w, delta_h = abs(bbox[2] - paired_box[2]), abs(bbox[3] - paired_box[3])
            shift_index = delta_x / paired_box[2] + delta_y / paired_box[3] \
                            + self.alpha * (delta_w / paired_box[2] + delta_h / paired_box[3])
            self.shift_index.append(shift_index)

        return '\n'.join(final_lines)

    def test_time_filter(self, min_size=32):
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        if self.choose_unpaired:
            ids_with_unpair = set(_['image_id'] for _ in self.coco.anns.values() \
                                        if len(_['bbox'][0])+len(_['bbox'][1]) == 4)

            choose_ids = set()
            for id in ids_with_unpair:
                if self.choose_unpaired in self.coco.imgs[id]['file_name'][0]:
                    choose_ids.add(id)
            ids_with_ann = choose_ids
        
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])

        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds
        
    
    def load_annotations(self, ann_file):
            """Load annotation from COCO style annotation file.

            Args:
                ann_file (str): Path of annotation file.

            Returns:
                list[dict]: Annotation info from COCO api.
            """

            self.coco = COCO(ann_file)
            # The order of returned `cat_ids` will not
            # change with the order of the CLASSES
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.img_ids = self.coco.get_img_ids()
            data_infos = []
            total_ann_ids = []
            for i in self.img_ids:
                info = self.coco.load_imgs([i])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
                ann_ids = self.coco.get_ann_ids(img_ids=[i])
                total_ann_ids.extend(ann_ids)
            assert len(set(total_ann_ids)) == len(
                total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
            return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())

        if self.filter_unpaired_sample:
            ids_with_unpair = set(_['image_id'] for _ in self.coco.anns.values() \
                                        if len(_['bbox'][0])+len(_['bbox'][1]) == 4)
            removed_ids = set()
            for id in ids_with_unpair:
                if self.filter_unpaired_sample in self.coco.imgs[id]['file_name'][0]:
                    removed_ids.add(id)
            
            ids_with_ann -= removed_ids

        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_rgb_bboxes, gt_tir_bboxes = [], []
        rgb_local_ids, tir_local_ids = [], []
        gt_labels = []
        rgb_inds, tir_inds = [], []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            rgb_bbox, tir_bbox = ann['bbox']
            local_person_id = ann['local_person_id']
            if rgb_bbox:
                rgb_bbox = [rgb_bbox[0], rgb_bbox[1], rgb_bbox[0]+rgb_bbox[2], rgb_bbox[1]+rgb_bbox[3]]
                gt_rgb_bboxes.append(rgb_bbox)
                rgb_local_ids.append(local_person_id)
                rgb_inds.append(i)
            
            if tir_bbox:
                tir_bbox = [tir_bbox[0], tir_bbox[1], tir_bbox[0]+tir_bbox[2], tir_bbox[1]+tir_bbox[3]]           
                gt_tir_bboxes.append(tir_bbox)
                tir_local_ids.append(local_person_id)
                tir_inds.append(i)
            
            gt_labels.append(self.cat2label[ann['category_id']])    # coco cat_id -> cat_inds(from 0)

        if gt_rgb_bboxes:
            gt_rgb_bboxes = np.array(gt_rgb_bboxes, dtype=np.float32)
            rgb_local_ids = np.array(rgb_local_ids, dtype=np.int64)
            gt_rgb_labels = np.array(gt_labels, dtype=np.int64)[rgb_inds]
        else:
            gt_rgb_bboxes = np.zeros((0, 4), dtype=np.float32)
            rgb_local_ids = np.array([], dtype=np.int64)
            gt_rgb_labels = np.array([], dtype=np.int64)

        if gt_tir_bboxes:
            gt_tir_bboxes = np.array(gt_tir_bboxes, dtype=np.float32)
            tir_local_ids = np.array(tir_local_ids, dtype=np.int64)
            gt_tir_labels = np.array(gt_labels, dtype=np.int64)[tir_inds]
        else:
            gt_tir_bboxes = np.zeros((0, 4), dtype=np.float32)
            tir_local_ids = np.array([], dtype=np.int64)
            gt_tir_labels = np.array([], dtype=np.int64)

        ann = dict(
            bboxes=[gt_rgb_bboxes, gt_tir_bboxes],
            labels=[gt_rgb_labels, gt_tir_labels],
            local_person_ids=[rgb_local_ids, tir_local_ids]
            )
        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def get_union_bbox(self, bbox1:np.array, bbox2:np.array) -> list:
        '''
        Inputs: bbox:np.array = [x1, y1, x2, y2]
        Return: union:list = [x1 ,y1, w, h]
        '''
        def f(bbox:np.array) -> list:
            if bbox.shape[0] == 0 or bbox.shape[0] == 4:
                return bbox.tolist()
            elif bbox.shape[0] == 1 and bbox.shape[1] == 4:
                return bbox[0].tolist()
            else:
                assert False, 'bbox dim error, id error!'
        
        bbox1, bbox2 = map(f, [bbox1, bbox2])
        
        union = []
        if not bbox1:
            union = [bbox2[0], bbox2[1], bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]]
        elif not bbox2:
            union = [bbox1[0], bbox1[1], bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]]
        else:
            x1 = min(bbox1[0], bbox2[0])
            y1 = min(bbox1[1], bbox2[1])
            x2 = max(bbox1[2], bbox2[2])
            y2 = max(bbox1[3], bbox2[3])
            union = [x1, y1, x2 - x1, y2 - y1]
        return union

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        assert len(results) == len(self)

        rgb_json_results = []
        tir_json_results = []
        union_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            # anchor nms
            # rgb_bboxes, tir_bboxes, rgb_ids, tir_ids, rgb_scores, tir_scores, union_bboxes, anchor_scores\
            #      = results[idx]
            #pair nms
            rgb_bboxes, tir_bboxes, rgb_ids, tir_ids, rgb_scores, tir_scores, _, anchor_scores\
                 = results[idx]
            # rgb
            for label in range(len(rgb_bboxes)): # in different class
                bboxes = rgb_bboxes[label]
                ids = rgb_ids[label]
                scores = rgb_scores[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(scores[i])
                    data['category_id'] = self.cat_ids[label]
                    data['person_id'] = int(ids[i])
                    rgb_json_results.append(data)

            # tir
            for label in range(len(tir_bboxes)):
                bboxes = tir_bboxes[label]
                ids = tir_ids[label]
                scores = tir_scores[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(scores[i])
                    data['category_id'] = self.cat_ids[label]
                    data['person_id'] = int(ids[i])
                    tir_json_results.append(data)

            # union
            # pair nms
            for label in range(len(anchor_scores)):
                anch_scores = anchor_scores[label]
                rgb_bb = rgb_bboxes[label]
                tir_bb = tir_bboxes[label]
                r_ids = rgb_ids[label]
                t_ids = tir_ids[label]
                r_scores = rgb_scores[label]
                t_scores = tir_scores[label]
                _union_res = []

                u_ids = np.unique(np.concatenate([r_ids, t_ids]))
                assert len(u_ids) == len(anch_scores)
                for id in u_ids:
                    _union_res.append(
                        self.get_union_bbox(rgb_bb[r_ids == id], tir_bb[t_ids == id]))

                for i in range(len(_union_res)):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = _union_res[i]
                    data['score'] = float(anch_scores[i])
                    data['category_id'] = self.cat_ids[label]
                    data['person_id'] = int(u_ids[i])
                    union_json_results.append(data)
            
            # anchor nms
            # id = 0
            # for label in range(len(union_bboxes)):
            #     bboxes = union_bboxes[label]
            #     scores = anchor_scores[label]
            #     for i in range(bboxes.shape[0]):
            #         data = dict()
            #         data['image_id'] = img_id
            #         data['bbox'] = self.xyxy2xywh(bboxes[i])
            #         data['score'] = float(scores[i])
            #         data['category_id'] = self.cat_ids[label]
            #         data['person_id'] = id
            #         union_json_results.append(data)
            #         id += 1

        return rgb_json_results, tir_json_results, union_json_results
    
    def results2json(self, results, outfile_prefix) -> tuple:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        rgb_result_files = dict()
        tir_result_files = dict()
        union_result_files = dict()

        rgb_json_results, tir_json_results, union_json_results = self._det2json(results)
        rgb_result_files['bbox'] = f'{outfile_prefix}/rgb_bbox.json'
        tir_result_files['bbox'] = f'{outfile_prefix}/tir_bbox.json'
        union_result_files['bbox'] = f'{outfile_prefix}/union_bbox.json'

        mmcv.dump(rgb_json_results, rgb_result_files['bbox'])
        mmcv.dump(tir_json_results, tir_result_files['bbox'])
        mmcv.dump(union_json_results, union_result_files['bbox'])

        return rgb_result_files, tir_result_files, union_result_files
    
    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        
        rgb_result_files,  tir_result_files, union_result_files = self.results2json(results, jsonfile_prefix)

        return rgb_result_files, tir_result_files, union_result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None
                 ):
        if self.choose_unpaired:
            return
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox',]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        
        # jsonfile_prefix = '/workspace/work_dirs/yolox_kaist_3stream_2nc_coattention/brambox_res'
        # get det in each modal
        if jsonfile_prefix and hasattr(self, "shift_path_name"):
            jsonfile_prefix += f'_{self.shift_path_name}'
            if not os.path.exists(jsonfile_prefix) and self.local_rank == 0:
                os.mkdir(jsonfile_prefix)

        rgb_result_files, tir_result_files, union_result_files, tmp_dir = \
                                        self.format_results(results, jsonfile_prefix)
        
        if self.test_cvc14:
            if tmp_dir is not None:
                tmp_dir.cleanup()
            return dict()
        
        # IF NOT TEST LAMR
        # return dict()

        self.mr_rgb, self.mr_tir, self.mr_union = self.evaluate_lamr(rgb_result_files['bbox'],
                                                        tir_result_files['bbox'],
                                                        union_result_files['bbox'],
                                                        jsonfile_prefix)
        
        msg = '\nHere LAMR percision not aligned with official matlab release!\n'
        msg += f'Union-LAMR All = {100 * self.mr_union["all"]:>5.2f}, '\
            + f'Union-LAMR Day = {100 * self.mr_union["day"]:>5.2f}, '\
            + f'Union-LAMR Night = {100 * self.mr_union["night"]:>5.2f}\n'\
            + f'RGB-LAMR All   = {100 * self.mr_rgb["all"]:>5.2f}, '\
            + f'RGB-LAMR Day   = {100 * self.mr_rgb["day"]:>5.2f}, '\
            + f'RGB-LAMR Night   = {100 * self.mr_rgb["night"]:>5.2f}\n' \
            + f'TIR-LAMR All   = {100 * self.mr_tir["all"]:>5.2f}, '\
            + f'TIR-LAMR Day   = {100 * self.mr_tir["day"]:>5.2f}, '\
            + f'TIR-LAMR Night   = {100 * self.mr_tir["night"]:>5.2f}'
        print_log(msg, logger=logger)
      
        if tmp_dir is not None:
            tmp_dir.cleanup()

        eval_result = OrderedDict()
        for modal in ['union', 'rgb', 'tir',]:
            res = getattr(self, 'mr_' + modal)
            eval_result[f'{modal}_mr_all'] = 100 * res['all']
            eval_result[f'{modal}_mr_day'] = 100 * res['day']
            eval_result[f'{modal}_mr_night'] = 100 * res['night']
        assert len(eval_result) == 9
        
        return eval_result

    def evaluate_lamr(self,
                      rgb_json:str,
                      tir_json:str,
                      union_json:str,
                      jsonfile_prefix:str):
        if not jsonfile_prefix:
            jsonfile_prefix = 'work_dirs'
        
        with open(rgb_json, 'r') as f:
            rgb_res = json.load(f)
        with open(tir_json, 'r') as f:
            tir_res = json.load(f)
        with open(union_json, 'r') as f:
            union_res = json.load(f)

        # eval rgb
        mr_rgb = dict()
        mr_rgb['all'], mr_rgb['day'], mr_rgb['night'] \
            = self.log_avg_MR(rgb_res, self, jsonfile_prefix, modal="vis")

        # eval tir
        mr_tir = dict()
        mr_tir['all'], mr_tir['day'], mr_tir['night'] \
            = self.log_avg_MR(tir_res, self, jsonfile_prefix, modal="tir")

        # eval tir
        mr_union = dict()
        if self.test_union:
            mr_union['all'], mr_union['day'], mr_union['night'] \
                = self.log_avg_MR(union_res, self, jsonfile_prefix, modal="union")
        else:
            mr_union['all'], mr_union['day'], mr_union['night'] = 0., 0., 0.

        return mr_rgb, mr_tir, mr_union

    def log_avg_MR(self, coco_results, ori_dataset, json_result_root, modal:str):
        assert modal in ['vis', 'tir', 'union']
        if hasattr(self, "shift_path_name") and modal == 'tir':
            modal = os.path.join('shift_test', f'tir_{self.shift_path_name}')
            print(f'\nUsing annotations in {modal} !')
        
        identify = lambda f: os.path.splitext("/".join(f.rsplit('/')[-3:]))[0]
        ground_truth = bbb.parse('anno_dollar',\
                                 f'/workspace/annotations/{modal}/*/*/*.txt',\
                                 identify, occlusion_tag_map=[0.0, 0.25, 0.75])
        
        bbb.filter_ignore(ground_truth, [bbb.ClassLabelFilter(['person']),  # only consider 'person' objects
                                        bbb.HeightRangeFilter((50, float('Inf'))),  # select instances of 50 pixels or higher
                                        bbb.OcclusionAreaFilter(
                                            (0.65, float('Inf')))])  # only include objects that are 65% visible or more

        for _, annos in ground_truth.items():
            for i in range(len(annos)):
                annos[i].class_label = 'person'
        
        # bbb.modify(ground_truth, [bbb.AspectRatioModifier(.41, modify_ignores=False)]);

        ground_truth_day = {key: values for key, values in ground_truth.items() if
                            key.startswith('set06') or key.startswith('set07') or key.startswith('set08')}
        ground_truth_night = {key: values for key, values in ground_truth.items() if
                            key.startswith('set09') or key.startswith('set10') or key.startswith('set11')}
        
        def parse_detections(format, input, identify_fun=identify, clslabelmap=['person']):
            dets = bbb.parse(format, input, identify_fun, class_label_map=clslabelmap)
            # bbb.modify(dets, [bbb.AspectRatioModifier(.41)])
            bbb.filter_discard(dets, [bbb.HeightRangeFilter((50 / 1.25, float('Inf')))])
            return dets

        image_info = ori_dataset.coco.dataset['images']
        id_to_name_dict = {}
        for img in image_info:
            name_list = img['file_name'][1][:-4].split('/')[1].split('_')
            name_list.remove('lwir')
            new_name = '/'.join(name_list)
            idx = img['id']
            id_to_name_dict[idx] = new_name
        assert len(id_to_name_dict) == len(ori_dataset.coco.dataset['images'])
        for item in coco_results:
            name = id_to_name_dict[item['image_id']]
            item['image_id'] = name

        json_result_file = f'{json_result_root}/lamr_bbox.json'
        with open(json_result_file, "w") as f:
            json.dump(coco_results, f)
        

        detections_all = parse_detections('det_coco', json_result_file)

        detections_day = {key: values for key, values in detections_all.items() if
                                    key.startswith('set06') or key.startswith('set07') or key.startswith('set08')}
        detections_night = {key: values for key, values in detections_all.items() if
                                    key.startswith('set09') or key.startswith('set10') or key.startswith('set11')}
        # all
        miss_rate_all, fppi_all = mr_fppi(detections_all, ground_truth)
        all_lamr = lamr(miss_rate_all, fppi_all)
        # day
        miss_rate_day, fppi_day = mr_fppi(detections_day, ground_truth_day)
        day_lamr = lamr(miss_rate_day, fppi_day)
        # night
        miss_rate_night, fppi_night = mr_fppi(detections_night, ground_truth_night)
        night_lamr = lamr(miss_rate_night, fppi_night)

        # msg = 'Task-LAMR All @[IOU=0.5] = {:.3f}\n'.format(all_lamr) \
        #     + 'Task-LAMR Day @[IOU=0.5] = {:.3f}\n'.format(day_lamr) \
        #     + 'Task-LAMR Night @[IOU=0.5] = {:.3f}\n'.format(night_lamr)
        # print(msg)
        if self.local_rank == 0:
            os.remove(json_result_file)
        return [all_lamr, day_lamr, night_lamr]
    
    @staticmethod
    def get_single_gt(ori_coco_gt, modal_pos):
        for key in ori_coco_gt.anns.keys():
            if ori_coco_gt.anns[key]['bbox'][modal_pos]:
                ori_coco_gt.anns[key]['bbox'] = ori_coco_gt.anns[key]['bbox'][modal_pos]
                ori_coco_gt.anns[key]['area'] = ori_coco_gt.anns[key]['area'][modal_pos]
                ori_coco_gt.anns[key]['width'] = ori_coco_gt.anns[key]['bbox'][2]
                ori_coco_gt.anns[key]['height'] = ori_coco_gt.anns[key]['bbox'][3]
                ori_coco_gt.anns[key]['occlusion'] = 0
            else:
                ori_coco_gt.anns.pop(key)          
        
        return ori_coco_gt

    @staticmethod
    def get_union_gt(ori_coco_gt):
        for key in ori_coco_gt.anns.keys():
            rgb_bbox, tir_bbox = ori_coco_gt.anns[key]['bbox']
            assert rgb_bbox or tir_bbox
            union_bbox = []
            if not rgb_bbox:
                union_bbox = tir_bbox
            elif not tir_bbox:
                union_bbox = rgb_bbox
            else:
                x1 = min(rgb_bbox[0], tir_bbox[0])
                y1 = min(rgb_bbox[1], tir_bbox[1])
                x2 = max(rgb_bbox[0] + rgb_bbox[2], tir_bbox[0] + tir_bbox[2])
                y2 = max(rgb_bbox[1] + rgb_bbox[3], tir_bbox[1] + tir_bbox[3])
                union_bbox = [x1, y1, x2 - x1, y2 - y1]
            area = union_bbox[2] * union_bbox[3]

            ori_coco_gt.anns[key]['bbox'] = union_bbox
            ori_coco_gt.anns[key]['area'] = area
            ori_coco_gt.anns[key]['width'] = union_bbox[2]
            ori_coco_gt.anns[key]['height'] = union_bbox[3]
            ori_coco_gt.anns[key]['occlusion'] = 0
        return ori_coco_gt


def lamr(miss_rate, fppi, num_of_samples=9):
    """ Compute the log average miss-rate from a given MR-FPPI curve.
    The log average miss-rate is defined as the average of a number of evenly spaced log miss-rate samples
    on the :math:`{log}(FPPI)` axis within the range :math:`[10^{-2}, 10^{0}]`

    Args:
        miss_rate (list): miss-rate values
        fppi (list): FPPI values
        num_of_samples (int, optional): Number of samples to take from the curve to measure the average precision; Default **9**

    Returns:
        Number: log average miss-rate
    """
    samples = np.logspace(-2., 0., num_of_samples)
    m = np.array(miss_rate)
    f = np.array(fppi)
    interpolated = scipy.interpolate.interp1d(f, m, fill_value=(1., 0.), bounds_error=False)(samples)
    for i, v in enumerate(interpolated):
        if i == 0:
            continue
        if v <= 0:
            interpolated[i] = interpolated[i-1]

    log_interpolated = np.log(interpolated)
    avg = sum(log_interpolated) / len(log_interpolated)
    return np.exp(avg)

