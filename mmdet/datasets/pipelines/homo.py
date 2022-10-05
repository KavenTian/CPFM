import copy
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
                 ):
        assert mode in ['test', 'aug']
        assert transform in ['homo', 'shift']
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
        
