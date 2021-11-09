# log save path
work_dir = 'work_dirs/swin_tiny_patch4_window7_dotav2/'

# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='OrientedRepPointsDetector',
    pretrained='work_dirs/swin_tiny_patch4_window7_224_torch2.pth',
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,         # tiny 96    small 96        base 128       large 192
        depths=[2, 2, 6, 2],  # tiny 2262  small 2 2 18 2  base 2 2 18 2  large 2 2 18 2
        num_heads=[3, 6, 12, 24],
        window_size=7,        # tiny 7     samall 7
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,  # 训练时间小于1×最好置为0.1
        ape=False,    # 是否需要对嵌入向量进行相对位置编码
        patch_norm=True,
        out_indices=(1, 2, 3),  # strides: [4, 8, 16, 32]  channel:[96, 192, 384, 768]
        use_checkpoint=False  # 为True则显存消耗量更少，但前向传播次数加倍/ 为False则正常训练
    ),
    neck=
        dict(
        type='FPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        #start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg
        ),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=19,
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
dataset_type = 'DotaDatasetv2'
data_root = '/media/test/4d846cae-2315-4928-8d1b-ca6d3a61a3c6/DOTA/DOTAv2.0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CorrectRBBox', correct_rbbox=True, refine_rbbox=True),
    dict(type='PolyResize',
        img_scale=[(1152, 768), (1152, 1024)],  # 建议根据显存来确定长边的值，在线多尺度缩放幅度控制在25%左右为佳
        keep_ratio=True,
        multiscale_mode='range',
        clamp_rbbox=False),
    dict(type='PolyRandomFlip', flip_ratio=0.5),
    dict(type='HSVAugment', hgain=0.015, sgain=0.7, vgain=0.4),
    dict(type='PolyRandomRotate', rotate_ratio=0.5, angles_range=180, auto_bound=False),
    dict(type='Pad', size_divisor=32),
   # dict(type='Poly_Mosaic_RandomPerspective', mosaic_ratio=0, ifcrop=True, degrees=0, translate=0.1, scale=0.2, shear=0, perspective=0.0),
    dict(type='MixUp', mixup_ratio=0.5),
    dict(type='PolyImgPlot', img_save_path=work_dir, save_img_num=16, class_num=18, thickness=2),
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
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval_split_1024/Train_dotav2_trainval1024_poly.json',
        img_prefix=data_root + 'trainval_split_1024/images/',
        pipeline=train_pipeline,
        Mosaic4=False,
        Mosaic9=False,
        Mixup=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval_split_1024/Train_dotav2_trainval1024_poly.json',
        img_prefix=data_root + 'trainval_split_1024/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test-dev_split/Test_datav2_test1024.json',
        img_prefix=data_root + 'test-dev_split/images/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])

runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
total_epochs = 36

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,          # 迭代n次时打印一次
    hooks=[
        dict(type='TextLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/swin_tiny_patch4_window7_dotav2/latest.pth'
workflow = [('train', 1)]

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(
#  #   type="DistOptimizerHook",
#  #   update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#  #   use_fp16=True,
# )