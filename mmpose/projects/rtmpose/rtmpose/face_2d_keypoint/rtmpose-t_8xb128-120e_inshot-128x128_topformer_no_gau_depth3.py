_base_ = ['mmpose::_base_/default_runtime.py']

# common setting
num_keypoints = 235
input_size = (128, 128)

# runtime
max_epochs = 300
stage2_num_epochs = 50
base_lr = 4e-3
train_batch_size = 128
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.005,
        begin=30,
        end=max_epochs,
        T_max=max_epochs - 30,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

norm_cfg = dict(type='BN', requires_grad=True)

cfgs_md2_middle = dict(
    cfg=[
    # k,  t,  c, s
        [3,   1,  16, 1], # 1/2        0.464K  17.461M
        [3,   4,  16, 2], # 1/4 1      3.44K   64.878M
        [3,   3,  16, 1], #            4.44K   41.772M
        [5,   3,  32, 2], # 1/8 3      6.776K  29.146M
        [5,   3,  32, 1], #            13.16K  30.952M
        [3,   3,  64, 2], # 1/16 5     16.12K  18.369M
        [3,   3,  64, 1], #            41.68K  24.508M
        [5,   6,  96, 2], # 1/32 7     0.129M  36.385M
        [5,   6,  96, 1], #            0.335M  49.298M
    ],
    channels=[16, 32, 64, 96],
    out_channels=[None, None, None, None],
    embed_out_indice=[2, 4, 6, 8],
    decode_out_indices=[3, ],
    num_heads=4,
    c2t_stride=1,
)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
     backbone=dict(
        type='Topformer',
        cfgs=cfgs_md2_middle['cfg'], 
        channels=cfgs_md2_middle['channels'],
        out_channels=cfgs_md2_middle['out_channels'], 
        embed_out_indice=cfgs_md2_middle['embed_out_indice'],
        decode_out_indices=cfgs_md2_middle['decode_out_indices'],
        depths=3,
        num_heads=cfgs_md2_middle['num_heads'],
        c2t_stride=cfgs_md2_middle['c2t_stride'],
        drop_path_rate=0.,
        norm_cfg=norm_cfg,
        init_cfg=dict(
            type='Pretrained',
            # prefix='backbone.',
            checkpoint='/home/zhangxiaoshuai/Pretrained/topformer-T-224-66.2.pth'
        ),
        injection=False),
    head=dict(
        type='RTMCCHead',
        in_channels=208,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=1,
        gau_cfg=None,
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'InshotDataset'
data_mode = 'topdown'
data_root = '/data/xiaoshuai/facial_lanmark/ldk_1face/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train_1face_annotations.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_face_annotations.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/coco_face_annotations.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='NME', rule='less', max_keep_ckpts=3, interval=1))

custom_hooks = [
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0002,
    #     update_buffers=True,
    #     priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='NME',
    norm_mode='keypoint_distance',
)
test_evaluator = val_evaluator

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])