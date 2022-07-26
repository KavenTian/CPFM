dataset_type = 'KaistDataset'   #数据集类型，考虑此处自定义dataloader
data_root = '/dataset/KAIST/coco_kaist/'  #数据集的根路径

img_norm_cfg = dict(
    mean = [103.939, 116.779, 123.68, 103.939, 116.779, 123.68],    #RGBT4通道均值    RGB=123.68, 116.779, 103.939, T=?
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375], #RGBT方差
    # std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #RGBT方差
    to_rgb=False #输入网络的图像通道顺序
)

img_scale = (512, 640)

train_pipeline = [
    dict(type='LoadMultiModalImageFromFiles',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize_Multi',img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadMultiModalImageFromFiles'),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        filter_empty_gt=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
    )
)

evaluation=dict(interval=1, metric='bbox')