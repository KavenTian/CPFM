import copy
from unittest import result
import numpy as np
import cv2

from ..builder import PIPELINES

@PIPELINES.register_module()
class Homography:
    def __init__(self, 
                 mode='test',
                 transform='homo',
                 shifts=None,
                 points_pair=None,
                 delta=8
                 ):
        '''
        shifts:[d_x, d_y]
        points_pair:[[[],[],[],[]],[[],[],[],[]]]
        '''
        assert mode in ['test', 'aug']
        assert transform in ['homo', 'shift', 'stay']
        if mode == 'test' and transform == 'homo':
            assert not shifts and points_pair, \
                'Points pair needed in Homograph, without shifts.'
        if mode == 'test' and transform == 'shift':
            assert shifts and not points_pair,\
                'Shifts needed in shift transform, without point pairs.'
        if mode == 'aug':
            assert not shifts and not points_pair, \
                'The transform scale connot be specific in Augmentation!'

        self.mode = mode
        self.transform = transform
        self.delta = delta
        if self.transform == 'homo' and self.mode == 'test':
            srcp=np.array(points_pair[0], dtype=np.float64).reshape(-1, 1, 2)
            tgtp=np.array(points_pair[1], dtype=np.float64).reshape(-1, 1, 2)
            self.H, _ = cv2.findHomography(srcp, tgtp, 0)
        
        elif self.transform == 'shift' and self.mode == 'test':
            self.H = np.array([[1, 0, shifts[0]], [0, 1, shifts[1]]], dtype=np.float32)
        

    def __call__(self, results):
        if self.transform == 'stay':
            return results
        
        if self.mode == 'aug':
            self.H = self.get_rand_H(self.delta)

        f = getattr(self, f'{self.transform}_transform')
        results = f(results)

        return results

    def homo_transform(self, results):
        '''make transform on tir'''
        
        imgs_in = results['img']
        img_tir = imgs_in[:, :, 3:]
        tir_out = cv2.warpPerspective(img_tir, self.H, (img_tir.shape[1],img_tir.shape[0]), 
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        imgs_out = np.concatenate((imgs_in[:, :, :3], tir_out), axis=2)
        assert imgs_out.shape[2] == 6
        results['img'] = imgs_out

        if self.mode == 'test':
            return results

        # For Aug
        fore_bboxes = results['gt_bboxes'][1]   # tir bboxes [x1, y1, x2, y2]: (N, 4)
        _bboxes = fore_bboxes.reshape(-1, 1, 2)
        bboxes = cv2.perspectiveTransform(_bboxes, self.H).reshape(-1, 4)

        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_tir.shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_tir.shape[0])
        
        # filter
        valid_mask = (bboxes[:, 0] != bboxes[:, 2]) & (bboxes[:, 1] != bboxes[:, 3]) 
        assert len(valid_mask) == len(bboxes)
        bboxes = bboxes[valid_mask]
        results['gt_bboxes'][1] = bboxes

        results['local_person_ids'][1] = results['local_person_ids'][1][valid_mask]
        results['gt_labels'][1] = results['gt_labels'][1][valid_mask]

        return results

    def shift_transform(self, results):
        imgs_in = results['img']
        img_tir = imgs_in[:, :, 3:]
        tir_out = cv2.warpAffine(img_tir, self.H, (img_tir.shape[1], img_tir.shape[0]), 
                        borderMode=cv2.BORDER_REPLICATE)
        imgs_out = np.concatenate((imgs_in[:, :, :3], tir_out), axis=2)
        assert imgs_out.shape[2] == 6
        results['img'] = imgs_out

        if self.mode == 'test':
            return results

        fore_bboxes = results['gt_bboxes'][1]   # tir bboxes [x1, y1, x2, y2]: (N, 4)
        _bboxes = fore_bboxes.reshape(-1, 1, 2)
        H = np.concatenate((self.H, np.array([[0,0,1]], dtype=self.H.dtype)), axis=0)
        bboxes = cv2.perspectiveTransform(_bboxes, H).reshape(-1, 4)

        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_tir.shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_tir.shape[0])

        # filter
        valid_mask = (bboxes[:, 0] != bboxes[:, 2]) & (bboxes[:, 1] != bboxes[:, 3]) 
        assert len(valid_mask) == len(bboxes)
        bboxes = bboxes[valid_mask]
        results['gt_bboxes'][1] = bboxes

        results['local_person_ids'][1] = results['local_person_ids'][1][valid_mask]
        results['gt_labels'][1] = results['gt_labels'][1][valid_mask]

        return results


    def get_rand_H(self, delta:int):
        if self.transform == 'homo':
            src_points = np.array([[0,0], [0,511], [639,0], [639,511]], dtype=np.float64).reshape(-1, 1, 2)
            noise = np.random.randint(-1 * delta, delta + 1, src_points.shape).astype(np.float64)
            tgt_points = src_points + noise
            H, _ = cv2.findHomography(src_points, tgt_points, 0)
        elif self.transform == 'shift':
            noise_x = np.random.randint(-1 * delta, delta + 1)
            noise_y = np.random.randint(-1 * delta, delta + 1)
            H = np.array([[1, 0, noise_x], [0, 1, noise_y]], dtype=np.float32)

        return H

