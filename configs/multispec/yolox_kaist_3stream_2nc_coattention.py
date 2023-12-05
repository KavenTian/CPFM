_base_ = [
    '../_base_/datasets/coco_detection.py',
     '../_base_/default_runtime.py'
]



model = dict(
    type='RGBT_Det_MultiStream', 
    backbone=dict(
        type='CSPDarknetCH_MultiStream',  
        in_channels=6,
        arch='P5',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
        stream=2,
        plugins=[
            dict(
            cfg=dict(type='CoAttention', pos_shape=[512//8, 640//8], pos_dim=256, d_model=256, nhead=4, dim_feedforward=1024),
            position='after_stage2'
            ),
            dict(
            cfg=dict(type='CoAttention', pos_shape=[512//16, 640//16], pos_dim=512, d_model=512, nhead=4, dim_feedforward=1024),
            position='after_stage3'
            ),
            dict(
            cfg=dict(type='CoAttention', pos_shape=[512//32, 640//32], pos_dim=1024, d_model=1024, nhead=4, dim_feedforward=1024),
            position='after_stage4'
            ),
        ]
    ),
    # backbone=dict(
    #     type='VGG_Mul',
    #     depth=16,
    #     with_last_pool=True,
    #     ceil_mode=True,
    #     out_indices=(2, 3, 4),
    #     plugins=[
    #         dict(
    #         cfg=dict(type='CoAttention', pos_shape=[512//8, 640//8], pos_dim=256, d_model=256, nhead=4, dim_feedforward=1024),
    #         position='after_stage2'
    #         ),
    #         dict(
    #         cfg=dict(type='CoAttention', pos_shape=[512//16, 640//16], pos_dim=512, d_model=512, nhead=4, dim_feedforward=1024),
    #         position='after_stage3'
    #         ),
    #         dict(
    #         cfg=dict(type='CoAttention', pos_shape=[512//32, 640//32], pos_dim=1024, d_model=1024, nhead=4, dim_feedforward=1024),
    #         position='after_stage4'
    #         ),
    #     ]
    # ),
    
    feature_fusion_module=dict(
        type='ModalFusion',    
        streams=['rgb', 'lwir'],
        in_channels=[256, 512, 1024],
        out_channels=[256*2, 512*2, 1024*2],    
        use_corrloss=False,
    ),
       
    neck=dict(
        type='YOLOXPAFPN', 
        in_channels=[256*2, 512*2, 1024*2],  
        out_channels=256*2,
        num_csp_blocks=1,
    ),

    bbox_head=dict(
        type='MultiSpeHead',
        use_cls_branch=False, 
        num_classes=1,
        in_channels=256*2, 
        stacked_convs=4,
        feat_channels=256,  
        align='deform',   # ['star', 'border', 'deform', ]
        # align=None,
        num_points = 25,
        offset_group=4,
        dcn_group=8,
        loss_bbox=dict(
                     type='DIoULoss',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
        # use_unpair_weights=True,
    ),
    init_cfg=dict(type='Pretrained', 
                  checkpoint='checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
                #   checkpoint='checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'
                ),

    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', center_radius=2.5), 
        # sampler=dict(type=''), 
    ),
    test_cfg=dict(
        nms_pre=1000,  
        score_thr=0.01, 
        nms=dict(type='nms', iou_threshold=0.65),  
    )
)
optimizer=dict(type='SGD', 
               lr=0.02,   # kaist
               weight_decay=0.0005,momentum=0.9,
               paramwise_cfg=dict(
                   custom_keys={
                       'backbone.CoCrossAttention_plugin_stage1':dict(lr_mult=0.)
                   }
               ),  
               nesterov=True,
        )

optimizer_config = dict(
    grad_clip=None  
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_ratio=0.0001,
    warmup_iters=2,
    step=29,
    # gamma=0.5,
    )


img_norm_cfg = dict(
    mean = [88.358, 82.084, 72.471, 40.749, 40.749, 40.749],   
    std=[60.129, 57.758, 57.987, 20.732, 20.732, 20.732], 
    # std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    to_rgb=False 
)


img_scale = (512, 640)


train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, min_bbox_size=4, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize_Multi',img_scale=img_scale, keep_ratio=True),
    # dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='UnionBox',
        #  single_modal='tir'
         ),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=[ 'img',
                                'gt_bboxes_rgb', 'gt_bboxes_tir', 'gt_bboxes_union',
                                'gt_labels_rgb', 'gt_labels_tir', 'gt_labels_union',
                                'local_person_ids_rgb',
                                'local_person_ids_tir',
                                'local_person_ids_union',
                                'people_num'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='GneralKaist',
        ann_file='/data/kaist-paired/annotations/id_paired_annotations/train.json',
        img_prefix='/data/kaist_dataset',
        pipeline=[
            dict(type='LoadMultiModalImageFromFiles', to_float32=True),
            dict(type='LoadAnnotations', poly2mask=False),
            dict(type='Homography', mode='aug',
                 transform='homo', delta=10
                 )
        ],
        # filter_unpaired_sample='set03',
    ),
    pipeline=train_pipeline,
    # dynamic_scale=img_scale
)

test_trans_type = 'shift'
# test_points_pair = [[[0,0], [0,511], [639,0], [639,511]],     # 0.3272
#                     [[4,-4], [4,516], [642,4], [642,507]]]
# test_points_pair = [[[0,0], [0,511], [639,0], [639,511]],     # 0.2899
#                     [[3,-4], [3,516], [641,4], [641,507]]]
test_points_pair = [[[0,0], [0,511], [639,0], [639,511]],       # 0.2372
                    [[2,-3], [2,514], [641,3], [641,508]]]
new_shifts = [15, 0]


test_pipeline = [
    dict(type='LoadMultiModalImageFromFiles'), 
    # dict(type='Homography',
    #      mode='test',
    #      transform=test_trans_type,
    #     #  points_pair=test_points_pair
    #      shifts=new_shifts
    #      ),

    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='Resize_Multi', img_scale=(512, 640), keep_ratio=True),
            # dict(type='RandomFlip'), 
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),

]


data = dict(
    samples_per_gpu=4,  
    workers_per_gpu=4,  
    train=train_dataset,
    val = dict(
        type='GneralKaist', 
        ann_file='/data/kaist-paired/annotations/id_paired_annotations/test.json',
        img_prefix='/data/kaist_dataset',
        pipeline=test_pipeline,
        # test_trans_dict=dict(type=test_trans_type, 
        #                     #  points_pair=test_points_pair,
        #                      shifts=new_shifts
        #                      ),
    ),
    test = dict(
        type='GneralKaist', 
        ann_file='/data/kaist-paired/annotations/id_paired_annotations/test.json',
        img_prefix='/data/kaist_dataset',
        pipeline=test_pipeline,
        # test_trans_dict=dict(type=test_trans_type, 
        #                     #  points_pair=test_points_pair,
        #                      shifts=new_shifts
        #                      ),
    ),
)



runner = dict(
    type='EpochBasedRunner',   
    max_epochs=28
)

# find_unused_parameters=True


log_config = dict(
    interval=50,    
    hooks=[ 
        # dict(type='TensorboardLoggerHook'),  
        dict(type='TextLoggerHook') 
    ]
)
log_level = 'INFO'


dist_params = dict(backend='nccl')  
load_from = None    
interval = 1
resume_from = None 
# workflow = [('train', 1)]   
# work_dir = 'outputs' 

evaluation = dict(
    interval=interval, 
    metric=['bbox'],    
)

checkpoint_config = dict(
    interval=1  
)

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=1, priority=48),
    # dict(
    #     type='SyncRandomSizeHook',
    #     ratio_range=(14, 26),
    #     img_scale=img_scale,
    #     interval=interval,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=1,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]
