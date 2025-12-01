_base_ = ['mmpose::_base_/default_runtime.py']

# common setting
num_keypoints = 235
input_size = (96, 96)

# runtime
max_epochs = 420
stage2_num_epochs = 100
base_lr = 4e-3
train_batch_size = 256
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
        eta_min=1.0e-5,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(type='RepGhostHook')
]

# codec settings
codec = dict(
    type='RegressionLabel',
    input_size=input_size)

norm_cfg = dict(type='BN', requires_grad=True)

enable_se = False
cfgs_md2_middle = dict(
    cfg = [
        # k, t, c, SE, s
        # stage1
        # [[3, 8, 16, 0, 1]],
        # stage2
        [[3, 24, 24, 0, 2]],
        [[3, 36, 24, 0, 1]],
        # stage3
        [[5, 36, 40, 0.25 if enable_se else 0, 2]],
        [[5, 60, 40, 0.25 if enable_se else 0, 1]],
        # stage4
        [[3, 120, 80, 0, 2]],
        [
            [3, 100, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 120, 80, 0, 1],
            [3, 240, 112, 0.25 if enable_se else 0, 1],
            [3, 336, 112, 0.25 if enable_se else 0, 1],
        ],
        # stage5
        [[5, 336, 160, 0.25 if enable_se else 0, 2]],
        [
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
            [5, 480, 160, 0, 1],
            [5, 480, 160, 0.25 if enable_se else 0, 1],
        ],
    ],
    embed_out_indice=[5, 7],
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
        type='RepGhostNet',
        cfgs=cfgs_md2_middle['cfg'], 
        out_indices=cfgs_md2_middle['embed_out_indice'],
        width=0.5,
        out_channels=96,
        out_feat_chs=[56, 80],
        deploy=False,
        # deploy=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='/home/zhangxiaoshuai/Checkpoint/FacialLandmark/euler_prealign_96x96_repghost_lite_single_ema/best_NME_epoch_411.pth'
        )
        ),
    head=dict(
        type='LiteMLEHead',
        in_channels=96,
        out_channels=num_keypoints,
        hidden_dims=36,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 16 for s in codec['input_size']]),
        simcc_split_ratio=1.5,
        final_layer_kernel_size=1,
        loss_cfg=[dict(methods=[dict(type='MLECCLoss',
                                       use_target_weight=False,
                                       loss_weight=1.0
                                       )
                                ],
                       name='loss_mle',
                    ),
                    dict(methods=[dict(type='AnisotropicDirectionLoss',
                                       loss_lambda=2.0
                                       )
                                ],
                        name='loss_adl',
                    ),
                    dict(methods=[dict(type='RLELoss',
                                       use_target_weight=False,
                                       size_average=True, 
                                       residual=True,
                                       sigmoid=False,
                                       )
                                ],
                        name='loss_rle',
                    ),
                ],
        loss=None,
        decoder=codec,
        scale_norm=True,
        beta=10),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'InshotDataset'
data_mode = 'topdown'
data_root = '/data/xiaoshuai/facial_lanmark/'

backend_args = dict(backend='local')

# pipelines
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='TopdownAlign', 
         input_size=codec['input_size'], 
         shift_prob=0.0, 
         orient_prob=0.0),
    dict(type='PackPoseInputs')
]

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(type='TopdownAlign', 
         input_size=codec['input_size'], 
         offset=0.05, 
         shift_prob=0.75, 
         orient_prob=0.75),
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
        ann_file='train_1118_resized/annotations/train_all_annotations.json',
        data_prefix=dict(img='train_1118_resized/'),
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
        ann_file='val_1118/annotations/val_all_annotations.json',
        data_prefix=dict(img='val_1118/'),
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
        ann_file='val_1118/annotations/val_all_annotations.json',
        data_prefix=dict(img='val_1118/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='NME', rule='less', max_keep_ckpts=3, interval=1))

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