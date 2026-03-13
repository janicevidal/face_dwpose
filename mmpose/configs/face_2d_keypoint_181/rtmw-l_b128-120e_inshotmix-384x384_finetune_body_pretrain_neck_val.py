_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
max_epochs = 120
stage2_num_epochs = 10
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

num_keypoints = 181
train_batch_size = 64
val_batch_size = 32

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.1),
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
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(384, 384),
    sigma=(6.93, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='/home/zhangxiaoshuai/Pretrained/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'  # noqa
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=None,
        out_indices=(
            1,
            2,
        ),
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='neck.',
            checkpoint='/home/zhangxiaoshuai/Pretrained/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'  # noqa
        )),
    head=dict(
        type='RTMWHead',
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='MaskKLDiscretLoss',
            use_target_weight=True,
            beta=1.,
            label_softmax=True,
            label_beta=10.
        ),
        decoder=codec),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'InshotDataset181'
data_mode = 'topdown'
data_root = '/data/xiaoshuai/facial_landmark_181/train_0310/'

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.2),
            dict(type='MedianBlur', p=0.2),
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
    # dict(type='RandomHalfBody'),
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

dataset_kpts181 = dict(
    type='RepeatDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train_181_annotations.json',
        data_prefix=dict(img='train/'),
        pipeline=[],
    ),
    times=2
)

kpt235_to_181 = [(0, 23), (2, 22), (4, 21), (6, 20), (8, 19), (10, 18), (12, 17), (14, 16), (16, 15), (18, 14), 
                 (20, 13), (22, 12), (24, 11), (26, 10), (28, 9), (30, 8), (32, 7), (34, 6), (36, 5), 
                 (37, 28), (38, 29), (39, 30), (40, 31), (41, 32), (42, 33), (43, 34), (44, 35), (45, 36), (46, 37), (47, 38), 
                 (48, 39), (49, 40), (50, 41), (51, 42), (52, 43), (53, 44), (54, 45), (55, 46), (56, 47), 
                 (57, 48), (58, 49), (59, 50), (60, 51), (61, 52), (62, 53), (63, 54), (64, 55), (65, 56), (66, 57), (67, 58), 
                 (68, 59), (69, 60), (70, 61), (71, 62), (72, 63), (73, 64), (74, 65), (75, 66), (76, 67), 
                 (77, 68), (78, 69), (79, 70), (80, 71), (81, 72), (82, 73), (83, 74), (84, 75), (85, 76), (86, 77), (87, 78), (88, 79), (89, 80), 
                 (90, 81), (91, 82), (92, 83), (93, 84), (94, 85), (95, 86), (96, 87), (97, 88), (98, 89), (99, 90), (100, 91), 
                 (101, 92), (102, 93), (103, 94), (104, 95), (105, 96), (106, 97), (107, 98), (108, 99), (109, 100), (110, 101), (111, 102), (112, 103), (113, 104),
                 (114, 105), (115, 106), (116, 107), (117, 108), (118, 109), (119, 110), (120, 111), (121, 112), (122, 113), (123, 114), (124, 115),
                 (125, 137), (126, 136), (127, 135), (128, 134), (129, 133), 
                 (132, 130), (133, 129), (134, 128), (135, 127), (136, 126),
                 (138, 140), (139, 141), (140, 142), 
                 (141, 143), ((142, 143), 144), (144, 145), ((145, 146), 146), (147, 147), (150, 148), 
                 (153, 149), ((154, 155), 150), (156, 151), ((157, 158), 152), (159, 153), 
                 ((160, 161), 154), (162, 155), ((163, 164), 156), (165, 157), ((166, 167), 158), (168, 159),
                 ((169, 170), 160), (171, 161), ((172, 173), 162), (174, 163), ((175, 176), 164), (177, 165),
                 ((178, 179), 166), (180, 167), ((181, 182), 168), (183, 169), 
                 ((184, 185), 170), (186, 171), ((187, 188), 172), (189, 173),
                 ((190, 191), 174), (192, 175), ((193, 194), 176), (195, 177), 
                 ((196, 197), 178), (198, 179), ((199, 200), 180),
                 (201, 120), (202, 125), (215, 116), (203, 117), (207, 118), (211, 119), (231, 121), (219, 122), (223, 123), (227, 124)
]

dataset_kpts235 = dict(
    type='InshotDataset',
    data_root='/data/xiaoshuai/facial_lanmark/train_0126/',
    data_mode=data_mode,
    ann_file='annotations/train_filtered_annotations.json',
    data_prefix=dict(img='images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=181, mapping=kpt235_to_181)
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/inshot_181.py'),
        datasets=[dataset_kpts181, dataset_kpts235],
        pipeline=train_pipeline,
        test_mode=False,
    )
)

val_kpts181 = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode=data_mode,
    ann_file='annotations/val_181_annotations.json',
    data_prefix=dict(img='val/'),
    pipeline=[],
)

val_kpts235 = dict(
    type='InshotDataset',
    data_root='/data/xiaoshuai/facial_lanmark/train_0126/',
    data_mode=data_mode,
    ann_file='annotations/val_filtered_annotations.json',
    data_prefix=dict(img='val_1229/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=181, mapping=kpt235_to_181)
    ],
)

test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/inshot_181.py'),
        datasets=[val_kpts181, val_kpts235],
        pipeline=val_pipeline,
        test_mode=True,
    )
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/inshot_181.py'),
        datasets=[val_kpts181, val_kpts235],
        pipeline=val_pipeline,
        test_mode=True,
    )
)

# val_dataloader = dict(
#     batch_size=val_batch_size,
#     num_workers=10,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='annotations/val_181_annotations.json',
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=val_pipeline,
#     )
# )

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='NME', rule='less', max_keep_ckpts=3, interval=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
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

work_dir='/home/zhangxiaoshuai/Checkpoint/FacialLandmark181/rtmw-l_b128-120e_inshotmix-384x384_finetune_body_pretrain_neck_val'
