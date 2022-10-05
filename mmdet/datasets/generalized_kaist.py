# Copyright (c) OpenMMLab. All rights reserved.
from ctypes import Union
import itertools
import logging
from ntpath import join
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict
import datetime, sys
import pdb, traceback
import copy, os, json

import mmcv
import numpy as np
from mmcv.utils import print_log
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
            # rgb_bboxes, tir_bboxes, rgb_ids, tir_ids, rgb_scores, tir_scores, union_bboxes, anchor_scores\
            #      = results[idx]
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
        rgb_result_files['bbox'] = f'{outfile_prefix}.rgb_bbox.json'
        tir_result_files['bbox'] = f'{outfile_prefix}.tir_bbox.json'
        union_result_files['bbox'] = f'{outfile_prefix}.union_bbox.json'

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

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox',]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        
        # jsonfile_prefix = '/workspace/work_dirs/yolox_kaist_3stream_2nc_coattention/brambox_res'
        # get det in each modal
        rgb_result_files, tir_result_files, union_result_files, tmp_dir = \
                                        self.format_results(results, jsonfile_prefix)
        
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
    
        # ori_coco_gt = self.coco      
        # # get GT in each modal
        # rgb_coco_gt = self.get_single_gt(copy.deepcopy(ori_coco_gt), modal_pos=0)
        # tir_coco_gt = self.get_single_gt(copy.deepcopy(ori_coco_gt), modal_pos=1)
        # union_coco_gt = self.get_union_gt(copy.deepcopy(ori_coco_gt))
        # self.rgb_eval_result = self.evaluate_kaist_mr(rgb_coco_gt, rgb_result_files['bbox'], 'RGB', logger=logger)
        # self.tir_eval_result = self.evaluate_kaist_mr(tir_coco_gt, tir_result_files['bbox'], 'TIR', logger=logger)
        # self.union_eval_result = self.evaluate_kaist_mr(union_coco_gt, union_result_files['bbox'], 'Union', logger=logger)
      
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
        with open(rgb_json, 'r') as f:
            rgb_res = json.load(f)
        with open(tir_json, 'r') as f:
            tir_res = json.load(f)
        with open(union_json, 'r') as f:
            union_res = json.load(f)

        # eval rgb
        mr_rgb = dict()
        mr_rgb['all'], mr_rgb['day'], mr_rgb['night'] = self.log_avg_MR(rgb_res, self, jsonfile_prefix, modal="vis")

        # eval tir
        mr_tir = dict()
        mr_tir['all'], mr_tir['day'], mr_tir['night'] = self.log_avg_MR(tir_res, self, jsonfile_prefix, modal="tir")

        # eval tir
        mr_union = dict()
        mr_union['all'], mr_union['day'], mr_union['night'] = self.log_avg_MR(union_res, self, jsonfile_prefix, modal="union")

        return mr_rgb, mr_tir, mr_union

    @staticmethod
    def log_avg_MR(coco_results, ori_dataset, json_result_root, modal:str):
        assert modal in ['vis', 'tir', 'union']
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

        json_result_file = f'{json_result_root}.lamr_bbox.json'
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
        os.remove(json_result_file)
        return [all_lamr, day_lamr, night_lamr]

    # def evaluate_kaist_mr(self, kaistGt, kaistDt_file, modal, logger=None):
        
    #     kaistDt = kaistGt.loadRes(kaistDt_file)
    #     imgIds = sorted(kaistGt.getImgIds())
    #     kaistEval = KAISTPedEval(kaistGt, kaistDt, 'bbox')
        
    #     kaistEval.params.catIds = [1]

    #     eval_result = {
    #     'all': copy.deepcopy(kaistEval),
    #     'day': copy.deepcopy(kaistEval),
    #     'night': copy.deepcopy(kaistEval),
    #     }

    #     eval_result['all'].params.imgIds = imgIds
    #     eval_result['all'].evaluate(0)
    #     eval_result['all'].accumulate()
    #     MR_all = eval_result['all'].summarize(0)

    #     eval_result['day'].params.imgIds = imgIds[:1455]
    #     eval_result['day'].evaluate(0)
    #     eval_result['day'].accumulate()
    #     MR_day = eval_result['day'].summarize(0)

    #     eval_result['night'].params.imgIds = imgIds[1455:]
    #     eval_result['night'].evaluate(0)
    #     eval_result['night'].accumulate()
    #     MR_night = eval_result['night'].summarize(0)

    #     recall_all = 1 - eval_result['all'].eval['yy'][0][-1]
        
    #     msg = f'\n############# Modal: {modal} #############\n' \
    #         + f'MR_all: {MR_all * 100:.2f}\n' \
    #         + f'MR_day: {MR_day * 100:.2f}\n' \
    #         + f'MR_night: {MR_night * 100:.2f}\n' \
    #         + f'recall_all: {recall_all * 100:.2f}\n' \
    #         + '######################################\n\n'
    #     print_log(msg, logger=logger)

    #     return eval_result
    
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

