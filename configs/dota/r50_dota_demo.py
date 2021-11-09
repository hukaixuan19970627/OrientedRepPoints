work_dir = 'work_dirs/r50_dota_demo/'

# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='OrientedRepPointsDetector',
    pretrained=None,#'torchvision://resnet50', 
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
    neck=
        dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg
        ),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=16,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.3,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_rbox_init=dict(type='GIoULoss', loss_weight=0.375),
        loss_rbox_refine=dict(type='GIoULoss', loss_weight=1.0),
        loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        top_ratio=0.4,))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=1),  # 每个gtbox仅选一个正样本
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='MaxIoUAssigner', #pre-assign to select more samples for samples selection
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='rnms', iou_thr=0.4),
    max_per_img=2000)

# dataset settings
dataset_type = 'DotaDatasetv1'
data_root = 'data/dataset_demo_split/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CorrectRBBox', correct_rbbox=True, refine_rbbox=True),
    dict(type='PolyResize',
        img_scale=[(1024, 512), (1024, 512)],
        keep_ratio=True,
        multiscale_mode='range',
        clamp_rbbox=False),
    dict(type='PolyRandomFlip', flip_ratio=0.5),
    #dict(type='HSVAugment', hgain=0.015, sgain=0.7, vgain=0.4),
    #dict(type='PolyRandomRotate', rotate_ratio=0.5, angles_range=180, auto_bound=False),
    dict(type='Pad', size_divisor=32),
    #dict(type='Poly_Mosaic_RandomPerspective', mosaic_ratio=0.5, ifcrop=True, degrees=0, translate=0.1, scale=0.2, shear=0, perspective=0.0),
    #dict(type='MixUp', mixup_ratio=0.5),
    dict(type='PolyImgPlot', img_save_path=work_dir, save_img_num=4, class_num=15, thickness=2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='PolyResize', keep_ratio=True),
            dict(type='PolyRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_datasetdemo.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline,
        Mosaic4=False,
        Mosaic9=False,
        Mixup=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train_datasetdemo.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_datasetdemo.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)   # 默认总batch_size=16 对应 SGD lr=0.01
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32, 38])
checkpoint_config = dict(interval=20)
# yapf:disable
log_config = dict(
    interval=5,          # 迭代n次时打印一次
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 40
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None#'work_dirs/orientedreppoints_r50_demo/latest.pth'
workflow = [('train', 1)]

