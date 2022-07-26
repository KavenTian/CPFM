_base_ = [
    '../_base_/datasets/kaist_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RGBT_Det',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    pub_feat_module=dict(
        type="ResNet",
        in_channels=6,
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type="Pretrained", checkpoint='torchvision://resnet50')
    ),
    feature_fusion_module=dict(
        type='ModalFusion',
        in_channels=[256,512,1024,2048],
        out_channels=[256*2,512*2,1024*2,2048*2],
        streams=['rgb', 'lwir', 'pub']
    ),
    streams = ['rgb', 'lwir', 'pub'],
    neck=dict(
        type="FPN",
        in_channels=[256*2,512*2,1024*2,2048*2],
        out_channels=256*2,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5
    ),
    bbox_head=dict(
        type='TOODHead',
        num_classes=1,
        in_channels=256*2,
        stacked_convs=6,
        feat_channels=256*2,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
)


# dataset_type = 'KaistDataset'   #数据集类型，考虑此处自定义dataloader
# data_root = '/dataset/KAIST/coco_kaist/'  #数据集的根路径
# img_norm_cfg = dict(
#     mean = [88.358, 82.084, 72.471, 40.749, 40.749, 40.749],    #RGBT4通道均值    RGB=123.68, 116.779, 103.939, T=?
#     std=[60.129, 57.758, 57.987, 20.732, 20.732, 20.732], #RGBT方差
#     # std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #RGBT方差
#     to_rgb=False #输入网络的图像通道顺序
# )

# img_scale = (512, 640)


# train_pipeline = [
#     dict(type='Mosaic', img_scale=img_scale, min_bbox_size=4, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Resize_Multi',img_scale=img_scale, keep_ratio=True),
#     dict(type='Pad', pad_to_square=True, pad_val=114.0),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]

# train_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type='KaistDataset',
#         ann_file='/dataset/KAIST/coco_kaist/annotations/train.json',
#         img_prefix='/dataset/KAIST/coco_kaist/train/',
#         pipeline=[
#             dict(type='LoadMultiModalImageFromFiles', to_float32=True),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         # filter_empty_gt=True,
#     ),
#     pipeline=train_pipeline,
#     # dynamic_scale=img_scale
# )

# # TODO:测试的流程
# test_pipeline = [
#     dict(type='LoadMultiModalImageFromFiles'), #prog.1:从文件路径加载图像
#     # dict(type='LoadAnnotations', with_bbox=True),  #prog.2:对于加载的图像加载其标注信息
#     # dict(
#     #     type='',    #prog.2:封装的数据增强方法
#     # )
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize_Multi', img_scale=img_scale, keep_ratio=True),
#             dict(type='RandomFlip'),  #demo时需注释
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32), #demo时需注释
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ]),

# ]

# data = dict(
#     samples_per_gpu=2,  #单个gpu的batch size
#     workers_per_gpu=2,  #单个gpu分配的数据加载线程数
#     train=train_dataset,
#     val = dict(
#         type='KaistDataset', #数据集的类型，同上train
#         ann_file='/dataset/KAIST/coco_kaist/annotations/val.json',
#         img_prefix='/dataset/KAIST/coco_kaist/val/',
#         pipeline=test_pipeline,
        
#     ),
#     test = dict(
#         type='KaistDataset', #数据集的类型，同上train
#         ann_file='/dataset/KAIST/coco_kaist/annotations/val.json',
#         img_prefix='/dataset/KAIST/coco_kaist/val/',
#         pipeline=test_pipeline,
        
#     )
# )

# optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0005,momentum=0.9,
#                paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),  #优化器的类型，见pytorch中的优化器参数
#                nesterov=True,
#         )
# lr_config = dict(
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=3,  # 5 epoch
#     num_last_epochs=2,
#     min_lr_ratio=0.05)
data = dict(
    samples_per_gpu=1,)


custom_hooks = [dict(type='SetEpochInfoHook')]
find_unused_parameters=True