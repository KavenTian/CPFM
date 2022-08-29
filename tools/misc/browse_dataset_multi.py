# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path

import mmcv, cv2
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import update_data_root


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='dataset config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['RandomAffine','PhotoMetricDistortion', 'RandomFlip','Pad','DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default='/workspace/work_dirs/visualize/',
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    if 'gt_semantic_seg' in cfg.train_pipeline[-1]['keys']:
        cfg.data.train.pipeline = [
            p for p in cfg.data.train.pipeline if p['type'] != 'SegRescale'
        ]
    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename'][0]).name.replace('_visible', '')
                                ) if args.output_dir is not None else None
        people_num = item['people_num']
        img_rgb = item['img'][:,:,:3].copy()
        img_tir = item['img'][:,:,3:].copy()
        bbox_rgb = [list(map(int, it)) for it in item['gt_bboxes'][0].tolist()]
        bbox_tir = [list(map(int, it)) for it in item['gt_bboxes'][1].tolist()]
        bbox_union = [list(map(int, it)) for it in item['gt_bboxes'][2].tolist()]
        person_id_rgb = item['local_person_ids'][0].tolist()
        person_id_tir = item['local_person_ids'][1].tolist()
        person_id_union = item['local_person_ids'][2].tolist()

        assert len(bbox_rgb) == len(person_id_rgb)
        for bbox, id in zip(bbox_rgb, person_id_rgb):
            assert len(bbox) == 4
            cv2.rectangle(img_rgb, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(img_rgb, str(id), (bbox[0], bbox[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        assert len(bbox_tir) == len(person_id_tir)
        for bbox, id in zip(bbox_tir, person_id_tir):
            assert len(bbox) == 4
            cv2.rectangle(img_tir, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(img_tir, str(id), (bbox[0], bbox[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # assert len(bbox_union) == len(person_id_union)
        # for bbox, id in zip(bbox_union, person_id_union):
        #     assert len(bbox) == 4
        #     cv2.rectangle(img_rgb, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #     cv2.putText(img_rgb, str(id), (bbox[0], bbox[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        #     cv2.rectangle(img_tir, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #     cv2.putText(img_tir, str(id), (bbox[0], bbox[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        img_out = np.concatenate((img_rgb, img_tir), axis=1)
        cv2.imwrite(filename, img_out)
        
        progress_bar.update()



if __name__ == '__main__':
    main()
