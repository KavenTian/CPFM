_base_ = './tood_r50_1x_kaist.py'
# learning policy
dataset_type = 'KaistDataset'   #数据集类型，考虑此处自定义dataloader
data_root = '/dataset/KAIST/coco_kaist/'  #数据集的根路径
img_norm_cfg = dict(
    mean = [88.358, 82.084, 72.471, 40.749, 40.749, 40.749],    #RGBT4通道均值    RGB=123.68, 116.779, 103.939, T=?
    std=[60.129, 57.758, 57.987, 20.732, 20.732, 20.732], #RGBT方差
    # std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #RGBT方差
    to_rgb=False #输入网络的图像通道顺序
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
    dict(type='Pad', pad_to_square=True, pad_val=114.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='KaistDataset',
        ann_file='/dataset/KAIST/coco_kaist/annotations/train.json',
        img_prefix='/dataset/KAIST/coco_kaist/train/',
        pipeline=[
            dict(type='LoadMultiModalImageFromFiles', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
    ),
    pipeline=train_pipeline,
    # dynamic_scale=img_scale
)

# TODO:测试的流程
test_pipeline = [
    dict(type='LoadMultiModalImageFromFiles'), #prog.1:从文件路径加载图像
    # dict(type='LoadAnnotations', with_bbox=True),  #prog.2:对于加载的图像加载其标注信息
    # dict(
    #     type='',    #prog.2:封装的数据增强方法
    # )
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize_Multi', img_scale=img_scale, keep_ratio=True),
            dict(type='RandomFlip'),  #demo时需注释
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32), #demo时需注释
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),

]

data = dict(
    samples_per_gpu=2,  #单个gpu的batch size
    workers_per_gpu=2,  #单个gpu分配的数据加载线程数
    train=train_dataset,
    val = dict(
        type='KaistDataset', #数据集的类型，同上train
        ann_file='/dataset/KAIST/coco_kaist/annotations/val.json',
        img_prefix='/dataset/KAIST/coco_kaist/val/',
        pipeline=test_pipeline,
        
    ),
    test = dict(
        type='KaistDataset', #数据集的类型，同上train
        ann_file='/dataset/KAIST/coco_kaist/annotations/val.json',
        img_prefix='/dataset/KAIST/coco_kaist/val/',
        pipeline=test_pipeline,
        
    )
)

optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0005,momentum=0.9,
               paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),  #优化器的类型，见pytorch中的优化器参数
               nesterov=True,
        )
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=3,  # 5 epoch
    num_last_epochs=2,
    min_lr_ratio=0.05)
# lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
interval = 1
resume_from = None  #从给定路径恢复checkpoints，训练模式将从checkpoints保存的位置开始训练
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    # dict(
    #     type='SyncRandomSizeHook',
    #     ratio_range=(14, 26),
    #     img_scale=img_scale,
    #     interval=interval,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=5,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]