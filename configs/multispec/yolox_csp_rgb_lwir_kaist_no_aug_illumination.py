_base_ = [
    '../_base_/datasets/coco_detection.py',
    # '../_base_/schedules/schedule_1x.py',
     '../_base_/default_runtime.py'
]
# # 自定义模块导入

# custom_imports = dict(
#     imports=[
#         'mmdet.models.detectors.baseline',
#         'mmdet.datasets.kaist',
#         # 'mmdet.dataset.pipeline.loading'
#     ],    #从文件导入自定义的模块（.py文件）
#     allow_failed_imports=False  #此参数还不清楚什么意思，大概是若导入有错会不会报错？
# )



# TODO:此处model需自定义
model = dict(
    type='RGBT_Det', #检测器名称
    backbone=dict(
        type='CSPDarknetCH',  #主干网络的类别，不同主干网络有不同的参数，具体主干网络参照mmdet/models/backbones/
        arch='P5',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(2, 3, 4),
        # frozen_stages=1,  
        # init_cfg=
    ),
    pub_feat_module=dict(
        type='CSPDarknetCH',
        in_channels=6,
        deepen_factor=1.0,
        widen_factor=1.0,
    ),
    neck=dict(
        type='YOLOXPAFPN', #neck的类别，见mmdet/models/neck/
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=1,
    ),
    use_ill_module=True,
    illum_aware_module=dict(
        type='Illum_Aware_Module',    #光照感知网络
        in_channels=3,
    ),
    feature_fusion_module=dict(
        type='ModalFusion',    #特征融合模块
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        streams=['rgb', 'lwir'],
    ),
    streams = ['rgb', 'lwir'],
    # feature_fusion_module=dict(
    #     type='ModalFusion',
    #     in_channels=[256 * 3, 512 * 3, 1024 * 3],
    #     out_channels=[256, 512, 1024]
    # ),
    # middle_parameters=dict( # 一些中间参数
    #     num_channels=6, #公有特征输入维度
    # ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        # loss_bbox=dict(
        #     type="CIoULoss",
        #     reduction="mean",
        #     eps=1e-8,
        #     loss_weight=5.0
        # ),
    ),
    init_cfg=dict(type='Pretrained', checkpoint='checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'),
    # TODO:训练时和测试的设置，如样本的匹配和nms等
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
optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0005,momentum=0.9,
               paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),  #优化器的类型，见pytorch中的优化器参数
               nesterov=True,
        )
# optimizer = dict(type='AdamW',
#                 lr=2e-4,
#                 weight_decay=0.0001,
#                 paramwise_cfg=dict(
#                 custom_keys={
#                     'backbone': dict(lr_mult=0.1),
#                     # 'feature_fusion_module': dict(lr_mult=0.1),
#                     # 'reference_points': dict(lr_mult=0.1)
#                 })
#                 )
optimizer_config = dict(
    grad_clip=None  #梯度限制（大多数方法不使用）
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
train_pipeline = [
    # dict(type='LoadMultiModalImageFromFiles',to_float32=True),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize_Multi',img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','gt_illumination'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='KaistDataset',
        ann_file='/dataset/KAIST/coco_kaist/annotations/train.json',
        img_prefix='/dataset/KAIST/coco_kaist/train/',
        pipeline=[
            dict(type='LoadMultiModalImageFromFiles', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_illumination=True)
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



runner = dict(
    type='EpochBasedRunner',    #runner的类别
    max_epochs=24
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
find_unused_parameters=True
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