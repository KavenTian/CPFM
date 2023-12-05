import copy
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class UnionBox:
    def __init__(self, single_modal=None) -> None:
        # if single_modal is setted, choose only one modality as a surpervision
        assert single_modal in [None, 'rgb', 'tir']
        self.single_modal = single_modal

    def __call__(self, results:dict):
        bboxes = copy.deepcopy(results['gt_bboxes'])
        labels = copy.deepcopy(results['gt_labels'])
        local_person_ids = copy.deepcopy(results['local_person_ids'])
        people_num = len(sorted(list(set(np.concatenate(local_person_ids, 0).tolist()))))
        results['people_num'] = people_num
        
        union_bboxes = np.zeros((people_num, 4), dtype=np.float32)
        union_labels = np.zeros(people_num, dtype=np.int64)
        for i in range(people_num):
            rgb_inds = np.where(local_person_ids[0] == i)[0]    # rgb_inds: 'np.array'
            rgb_bbox = bboxes[0][rgb_inds]
            rgb_label = labels[0][rgb_inds]
            tir_inds = np.where(local_person_ids[1] == i)[0]
            tir_bbox = bboxes[1][tir_inds]
            tir_label = labels[1][tir_inds]
            # assert rgb_inds.shape[0] <= 1 and tir_inds.shape[0] <= 1
            if rgb_inds.shape[0] > 1 or tir_inds.shape[0] > 1:
                print('Person ID Error!')
                print(local_person_ids[0])
                print(local_person_ids[1])
                assert False
            assert rgb_inds.shape[0] + tir_inds.shape[0] > 0
            if rgb_bbox.shape[0] > 0:
                assert rgb_bbox.shape[-1] == 4
            else:
                assert tir_bbox.shape[-1] == 4

            if rgb_bbox.shape[0] + tir_bbox.shape[0] == 2:
                if self.single_modal == 'rgb':
                    union_bboxes[i, :] = rgb_bbox[0]
                elif self.single_modal == 'tir':
                    union_bboxes[i, :] = tir_bbox[0]
                else:
                    x1 = min(rgb_bbox[0, 0], tir_bbox[0, 0])
                    x2 = max(rgb_bbox[0, 2], tir_bbox[0, 2])
                    y1 = min(rgb_bbox[0, 1], tir_bbox[0, 1])
                    y2 = max(rgb_bbox[0, 3], tir_bbox[0, 3])
                    union_bboxes[i, :] = np.array([x1, y1, x2, y2], dtype=np.float32)
                union_labels[i] = rgb_label
            elif rgb_bbox.shape[0] == 1:
                union_bboxes[i, :] = rgb_bbox[0]
                union_labels[i] = rgb_label
            else:
                union_bboxes[i, :] = tir_bbox[0]
                union_labels[i] = tir_label
        union_ids = np.array(range(people_num))
        
        bboxes.append(union_bboxes)
        labels.append(union_labels)
        local_person_ids.append(union_ids)
        results['gt_bboxes'] = bboxes
        results['gt_labels'] = labels
        results['local_person_ids'] = local_person_ids
        
        return results


