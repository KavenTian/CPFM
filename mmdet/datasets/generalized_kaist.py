# Copyright (c) OpenMMLab. All rights reserved.
from ctypes import Union
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict
import datetime, sys
import pdb, traceback
import copy

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import anchor, eval_recalls
from .api_wrappers import COCO, COCOeval
from pycocotools.cocoeval import Params
from .builder import DATASETS
from .custom import CustomDataset


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

    def _det2json(self, results) -> list:
        """Convert detection results to COCO json style."""
        assert len(results) == len(self)

        rgb_json_results = []
        tir_json_results = []
        union_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            rgb_bboxes, tir_bboxes, rgb_ids, tir_ids, rgb_scores, tir_scores, anchor_scores = results[idx]
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
        
        ori_coco_gt = self.coco
        
        # get GT in each modal
        rgb_coco_gt = self.get_single_gt(copy.deepcopy(ori_coco_gt), modal_pos=0)
        tir_coco_gt = self.get_single_gt(copy.deepcopy(ori_coco_gt), modal_pos=1)
        union_coco_gt = self.get_union_gt(copy.deepcopy(ori_coco_gt))

        # get det in each modal
        rgb_result_files, tir_result_files, union_result_files, tmp_dir = \
                                        self.format_results(results, jsonfile_prefix)

        self.rgb_eval_result = self.evaluate_kaist_mr(rgb_coco_gt, rgb_result_files['bbox'], 'RGB', logger=logger)
        self.tir_eval_result = self.evaluate_kaist_mr(tir_coco_gt, tir_result_files['bbox'], 'TIR', logger=logger)
        self.union_eval_result = self.evaluate_kaist_mr(union_coco_gt, union_result_files['bbox'], 'Union', logger=logger)
      
        if tmp_dir is not None:
            tmp_dir.cleanup()

        eval_result = OrderedDict()
        for modal in ['rgb', 'tir', 'union']:
            res = getattr(self, modal + '_eval_result')
            mr_all = 100 * res['all'].summarize(0)
            mr_day = 100 * res['day'].summarize(0)
            mr_night = 100 * res['night'].summarize(0)
            recall_all = 100 * (1 - res['all'].eval['yy'][0][-1])
            eval_result[f'{modal}_mr_all'] = mr_all
            eval_result[f'{modal}_mr_day'] = mr_day
            eval_result[f'{modal}_mr_night'] = mr_night
            eval_result[f'{modal}_recall_all'] = recall_all
        assert len(eval_result) == 12
        
        return eval_result

    def evaluate_kaist_mr(self, kaistGt, kaistDt_file, modal, logger=None):
        
        kaistDt = kaistGt.loadRes(kaistDt_file)
        imgIds = sorted(kaistGt.getImgIds())
        kaistEval = KAISTPedEval(kaistGt, kaistDt, 'bbox')
        
        kaistEval.params.catIds = [1]

        eval_result = {
        'all': copy.deepcopy(kaistEval),
        'day': copy.deepcopy(kaistEval),
        'night': copy.deepcopy(kaistEval),
        }

        eval_result['all'].params.imgIds = imgIds
        eval_result['all'].evaluate(0)
        eval_result['all'].accumulate()
        MR_all = eval_result['all'].summarize(0)

        eval_result['day'].params.imgIds = imgIds[:1455]
        eval_result['day'].evaluate(0)
        eval_result['day'].accumulate()
        MR_day = eval_result['day'].summarize(0)

        eval_result['night'].params.imgIds = imgIds[1455:]
        eval_result['night'].evaluate(0)
        eval_result['night'].accumulate()
        MR_night = eval_result['night'].summarize(0)

        recall_all = 1 - eval_result['all'].eval['yy'][0][-1]
        
        msg = f'\n############# Modal: {modal} #############\n' \
            + f'MR_all: {MR_all * 100:.2f}\n' \
            + f'MR_day: {MR_day * 100:.2f}\n' \
            + f'MR_night: {MR_night * 100:.2f}\n' \
            + f'recall_all: {recall_all * 100:.2f}\n' \
            + '######################################\n\n'
        print_log(msg, logger=logger)

        return eval_result
    
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


class KAISTPedEval(COCOeval):

    def __init__(self, kaistGt=None, kaistDt=None, iouType='bbox', method='unknown'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        super().__init__(kaistGt, kaistDt, iouType)

        self.params = KAISTParams(iouType=iouType)   # parameters
        self.method = method

    def _prepare(self, id_setup):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gbox = gt['bbox']
            gt['ignore'] = 1 \
                if gt['height'] < self.params.HtRng[id_setup][0] \
                or gt['height'] > self.params.HtRng[id_setup][1] \
                or gt['occlusion'] not in self.params.OccRng[id_setup] \
                or gbox[0] < self.params.bndRng[0] \
                or gbox[1] < self.params.bndRng[1] \
                or gbox[0] + gbox[2] > self.params.bndRng[2] \
                or gbox[1] + gbox[3] > self.params.bndRng[3] \
                else gt['ignore']

        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)

        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval = {}                      # accumulated evaluation results

    def evaluate(self, id_setup):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if p.useSegm is not None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare(id_setup)
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        computeIoU = self.computeIoU

        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        HtRng = self.params.HtRng[id_setup]
        OccRng = self.params.OccRng[id_setup]
        self.evalImgs = [evaluateImg(imgId, catId, HtRng, OccRng, maxDet)
                         for catId in catIds
                         for imgId in p.imgIds]

        self._paramsEval = copy.deepcopy(self.params)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['ignore']) for o in gt]
        ious = self.iou(d, g, iscrowd)
        return ious

    def iou(self, dts, gts, pyiscrowd):
        dts = np.asarray(dts)
        gts = np.asarray(gts)
        pyiscrowd = np.asarray(pyiscrowd)
        ious = np.zeros((len(dts), len(gts)))
        for j, gt in enumerate(gts):
            gx1 = gt[0]
            gy1 = gt[1]
            gx2 = gt[0] + gt[2]
            gy2 = gt[1] + gt[3]
            garea = gt[2] * gt[3]
            for i, dt in enumerate(dts):
                dx1 = dt[0]
                dy1 = dt[1]
                dx2 = dt[0] + dt[2]
                dy2 = dt[1] + dt[3]
                darea = dt[2] * dt[3]

                unionw = min(dx2, gx2) - max(dx1, gx1)
                if unionw <= 0:
                    continue
                unionh = min(dy2, gy2) - max(dy1, gy1)
                if unionh <= 0:
                    continue
                t = unionw * unionh
                if pyiscrowd[j]:
                    unionarea = darea
                else:
                    unionarea = darea + garea - t

                ious[i, j] = float(t) / unionarea
        return ious

    def evaluateImg(self, imgId, catId, hRng, oRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        try:
            p = self.params
            if p.useCats:
                gt = self._gts[imgId, catId]
                dt = self._dts[imgId, catId]
            else:
                gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
                dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
            
            if len(gt) == 0 and len(dt) == 0:
                return None

            for g in gt:
                if g['ignore']:
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0

            # sort dt highest score first, sort gt ignore last
            gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
            gt = [gt[i] for i in gtind]
            dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
            dt = [dt[i] for i in dtind[0:maxDet]]

            if len(dt) == 0:
                return None

            # load computed ious        
            ious = self.ious[imgId, catId][dtind, :] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
            ious = ious[:, gtind]

            T = len(p.iouThrs)
            G = len(gt)
            D = len(dt)
            gtm = np.zeros((T, G))
            dtm = np.zeros((T, D))
            gtIg = np.array([g['_ignore'] for g in gt])
            dtIg = np.zeros((T, D))

            if not len(ious) == 0:
                for tind, t in enumerate(p.iouThrs):
                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        iou = min([t, 1 - 1e-10])
                        bstOa = iou
                        bstg = -2
                        bstm = -2
                        for gind, g in enumerate(gt):
                            m = gtm[tind, gind]
                            # if this gt already matched, and not a crowd, continue
                            if m > 0:
                                continue
                            # if dt matched to reg gt, and on ignore gt, stop
                            if bstm != -2 and gtIg[gind] == 1:
                                break
                            # continue to next gt unless better match made
                            if ious[dind, gind] < bstOa:
                                continue
                            # if match successful and best so far, store appropriately
                            bstOa = ious[dind, gind]
                            bstg = gind
                            if gtIg[gind] == 0:
                                bstm = 1
                            else:
                                bstm = -1

                        # if match made store id of match for both dt and gt
                        if bstg == -2:
                            continue
                        dtIg[tind, dind] = gtIg[bstg]
                        dtm[tind, dind] = gt[bstg]['id']
                        if bstm == 1:
                            gtm[tind, bstg] = d['id']

        except Exception:

            ex_type, ex_value, ex_traceback = sys.exc_info()            

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

            sys.stderr.write("[Error] Exception type : %s \n" % ex_type.__name__)
            sys.stderr.write("[Error] Exception message : %s \n" % ex_value)
            for trace in stack_trace:
                sys.stderr.write("[Error] (Stack trace) %s\n" % trace)

            pdb.set_trace()

        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'hRng': hRng,
            'oRng': oRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.fppiThrs)
        K = len(p.catIds) if p.useCats else 1
        M = len(p.maxDets)
        ys = -np.ones((T, R, K, M))     # -1 for the precision of absent categories

        xx_graph = []
        yy_graph = []

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = [1]                    # _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * I0
            for m, maxDet in enumerate(m_list):
                E = [self.evalImgs[Nk + i] for i in i_list]
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                # different sorting method generates slightly different results.
                # mergesort is used to be consistent as Matlab implementation.

                inds = np.argsort(-dtScores, kind='mergesort')

                dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                gtIg = np.concatenate([e['gtIgnore'] for e in E])
                npig = np.count_nonzero(gtIg == 0)
                if npig == 0:
                    continue
                tps = np.logical_and(dtm, np.logical_not(dtIg))
                fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                inds = np.where(dtIg == 0)[1]
                tps = tps[:, inds]
                fps = fps[:, inds]

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
            
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fppi = np.array(fp) / I0
                    nd = len(tp)
                    recall = tp / npig
                    q = np.zeros((R,))

                    xx_graph.append(fppi)
                    yy_graph.append(1 - recall)

                    # numpy is slow without cython optimization for accessing elements
                    # use python array gets significant speed improvement
                    recall = recall.tolist()
                    q = q.tolist()

                    for i in range(nd - 1, 0, -1):
                        if recall[i] < recall[i - 1]:
                            recall[i - 1] = recall[i]

                    inds = np.searchsorted(fppi, p.fppiThrs, side='right') - 1
                    try:
                        for ri, pi in enumerate(inds):
                            q[ri] = recall[pi]
                    except Exception:
                        pass
                    ys[t, :, k, m] = np.array(q)
        
        self.eval = {
            'params': p,
            'counts': [T, R, K, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'TP': ys,
            'xx': xx_graph,
            'yy': yy_graph
        }

    @staticmethod
    def draw_figure(ax, eval_results, methods, colors):
        """Draw figure"""
        assert len(eval_results) == len(methods) == len(colors)

        for eval_result, method, color in zip(eval_results, methods, colors):
            mrs = 1 - eval_result['TP']
            mean_s = np.log(mrs[mrs < 2])
            mean_s = np.mean(mean_s)
            mean_s = float(np.exp(mean_s) * 100)

            xx = eval_result['xx']
            yy = eval_result['yy']

            ax.plot(xx[0], yy[0], color=color, linewidth=3, label=f'{mean_s:.2f}%, {method}')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()

        yt = [1, 5] + list(range(10, 60, 10)) + [64, 80]
        yticklabels = ['.{:02d}'.format(num) for num in yt]

        yt += [100]
        yt = [yy / 100.0 for yy in yt]
        yticklabels += [1]
        
        ax.set_yticks(yt)
        ax.set_yticklabels(yticklabels)
        ax.grid(which='major', axis='both')
        ax.set_ylim(0.01, 1)
        ax.set_xlim(2e-4, 50)
        ax.set_ylabel('miss rate')
        ax.set_xlabel('false positives per image')

    def summarize(self, id_setup, res_file=None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None, maxDets=100):
            OCC_TO_TEXT = ['none', 'partial_occ', 'heavy_occ']

            p = self.params
            iStr = ' {:<18} {} @ {:<18} [ IoU={:<9} | height={:>6s} | visibility={:>6s} ] = {:0.2f}%'
            titleStr = 'Average Miss Rate'
            typeStr = '(MR)'
            setupStr = p.SetupLbl[id_setup]
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            heightStr = '[{:0.0f}:{:0.0f}]'.format(p.HtRng[id_setup][0], p.HtRng[id_setup][1])
            occlStr = '[' + '+'.join(['{:s}'.format(OCC_TO_TEXT[occ]) for occ in p.OccRng[id_setup]]) + ']'

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            # dimension of precision: [TxRxKxAxM]
            s = self.eval['TP']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            mrs = 1 - s[:, :, :, mind]

            if len(mrs[mrs < 2]) == 0:
                mean_s = -1
            else:
                mean_s = np.log(mrs[mrs < 2])
                mean_s = np.mean(mean_s)
                mean_s = np.exp(mean_s)

            if res_file:
                res_file.write(iStr.format(titleStr, typeStr, setupStr, iouStr, heightStr, occlStr, mean_s * 100))
                res_file.write('\n')
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        return _summarize(iouThr=.5, maxDets=1000)


class KAISTParams(Params):
    """Params for KAISTPed evaluation api"""

    def setDetParams(self):
        super().setDetParams()

        # Override variables for KAISTPed benchmark
        self.iouThrs = np.array([0.5])
        self.maxDets = [1000]

        # KAISTPed specific settings
        self.fppiThrs = np.array([0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000])
        self.HtRng = [[55, 1e5 ** 2], [50, 75], [50, 1e5 ** 2], [20, 1e5 ** 2]]
        self.OccRng = [[0, 1], [0, 1], [2], [0, 1, 2]]
        self.SetupLbl = ['Reasonable', 'Reasonable_small', 'Reasonable_occ=heavy', 'All']

        self.bndRng = [5, 5, 635, 507]  # discard bbs outside this pixel range

