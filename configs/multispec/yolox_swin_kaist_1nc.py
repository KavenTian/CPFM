_base_ = [
    '../_base_/datasets/coco_detection.py',
    # '../_base_/schedules/schedule_1x.py',
     '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    type='RGBT_Det',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1,2,3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    pub_feat_module=dict(
        type='SwinTransformer',
        in_channels=6,
        embed_dims=96,
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1,2,3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    feature_fusion_module=dict(
        type='ModalFusion',
        in_channels=[192, 384, 768],
        out_channels=[256,512,1024]
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
    ),
    train_cfg=dict(
        assigner=dict(type='SimOTAAssigner', center_radius=2.5), #正负样本的匹配机制，见mmdet/core/bbox/assigners/
        # sampler=dict(type=''),  #正负样本的采样机制，见mmdet/core/bbox/samplers/
    ),
    test_cfg=dict(
        nms_pre=1000,   #nms前的box数
        score_thr=0.01, # bbox的分数阈值
        nms=dict(type='nms', iou_threshold=0.65),   #nms的配置
    )
)
# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
optimizer_config = dict(
    grad_clip=None  #梯度限制（大多数方法不使用）
)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[18, 22])
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
runner = dict(
    type='EpochBasedRunner',    #runner的类别
    max_epochs=24
)

# lr_config = dict(warmup_iters=1000, step=[8, 11])
# runner = dict(max_epochs=12)

# TODO:数据集需自定义--定义如何处理标注信息
dataset_type = 'KaistDataset'   #数据集类型，考虑此处自定义dataloader
data_root = '/dataset/KAIST/coco_kaist/'  #数据集的根路径
img_norm_cfg = dict(
    mean = [88.358, 82.084, 72.471, 40.749, 40.749, 40.749],    #RGBT4通道均值    RGB=123.68, 116.779, 103.939, T=?
    std=[60.129, 57.758, 57.987, 20.732, 20.732, 20.732], #RGBT方差
    # std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], #RGBT方差
    to_rgb=False #输入网络的图像通道顺序
)

img_scale = (512, 640)

# classes = ('person', 'people', 'cyclist')


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
        # filter_empty_gt=True,
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
# TODO:dataloader需自定义

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




# 设置日志相关配置
log_config = dict(
    interval=50,    #打印日志的间隔
    hooks=[ #用于记录训练过程的日志
        # dict(type='TensorboardLoggerHook'),  #tensorboard日志
        dict(type='TextLoggerHook') #文本日志
    ]
)
log_level = 'INFO'

# 其他配置
dist_params = dict(backend='nccl')  #用于设置分布式训练的参数，端口也同样可以设置
load_from = None    #从给定路径里加载模型作为预训练模型
interval = 1
resume_from = None  #从给定路径恢复checkpoints，训练模式将从checkpoints保存的位置开始训练
# workflow = [('train', 1)]   #runner的工作流程，表示只有一个工作流且仅执行一次（根据前面的max_epochs训练12个回合）
workdir = 'outputs' #用于保存当前实验的模型checkpoint和log文件的地址

evaluation = dict(
    interval=interval, #验证间隔
    metric=['bbox'],    #验证期间使用的指标
)

checkpoint_config = dict(
    interval=1  #保存checkpoint的间隔
)

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
        num_last_epochs=15,
        interval=interval,
        priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=resume_from, priority=49)
]