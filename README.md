# Code for CPFM

1. A pre-trained YOLOX weights is needed, please download from [here](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) and put it at /(path_for_project)/checkpoints/   .          
2. The configs used in paper are:      
   ./configs/multispec/yolox_kaist_3stream_2nc_coattention.py           
   ./configs/multispec/yolox_kaist_3stream_2nc_coattention_cvc14.py    
3. You should have some minor changes on configs(i.e. some PATH) to fit your environment.     
4. Other usage is the same as [mmdet](https://github.com/open-mmlab/mmdetection).
