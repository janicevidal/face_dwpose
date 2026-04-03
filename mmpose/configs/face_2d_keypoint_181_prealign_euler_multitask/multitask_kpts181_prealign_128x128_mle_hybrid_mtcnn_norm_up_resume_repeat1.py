_base_ = ['mmpose::_base_/default_runtime.py']

# common setting
num_keypoints = 181
input_size = (128, 128)

# runtime
max_epochs = 120
stage2_num_epochs = 100
base_lr = 4e-3
train_batch_size = 256
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

mean_face='/data/xiaoshuai/facial_landmark_181/mean_face/mean_face_symmetric_centered_181.npy'

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
        eta_min=base_lr * 0.001,
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
    type='RegressionLabelWithAnglesHybrid',
    input_size=input_size,
    euler_bin=(64, 48),
    euler_bin_interval=(1, 2, 8, 16, 64),
    max_angles=64)

norm_cfg = dict(type='BN', requires_grad=True)

enable_se = True
cfgs_md2_middle = dict(
    cfg = [
        # k, t, c, SE, s
        # stage1
        [[3, 8, 16, 0, 1]],
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
    embed_out_indice=[6, 8],
)

# model settings
model = dict(
    type='TopdownPosePrealignEstimatorMultitask',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[127.5, 127.5, 127.5],
        std=[128, 128, 128],
        bgr_to_rgb=False),
    backbone=dict(
        type='RepGhostNet',
        cfgs=cfgs_md2_middle['cfg'], 
        out_indices=cfgs_md2_middle['embed_out_indice'],
        width=0.8,
        out_channels=192,
        block_shift=0,
        out_feat_chs=[88, 128],
        deploy=False,
        # deploy=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='/home/zhangxiaoshuai/Checkpoint/FacialLandmark181/_euler_kpts181_prealign/multitask_kpts181_prealign_128x128_mle_hybrid_mtcnn_norm_up/best_NME_epoch_363.pth'
        )
        ),
    head=dict(
        type='MultiTaskHybridLiteMLEHead',
        in_channels=192,
        out_channels=num_keypoints,
        hidden_dims=64,
        euler_in_channels=128,
        euler_bin=codec['euler_bin'],
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 16 for s in codec['input_size']]),
        euler_bin_interval=codec['euler_bin_interval'],
        simcc_split_ratio=2.0,
        final_layer_kernel_size=1,
        loss_cfg=[dict(methods=[dict(type='MLECCLoss',
                                       use_target_weight=True,
                                       loss_weight=1.0
                                       )
                                ],
                       name='loss_mle',
                    ),
                    dict(methods=[dict(type='AnisotropicDirectionLoss',
                                       loss_lambda=2.0,
                                       num_kpts=num_keypoints,
                                       )
                                ],
                        name='loss_adl',
                    ),
                    dict(methods=[dict(type='RLELoss',
                                       use_target_weight=True,
                                       size_average=True, 
                                       residual=True,
                                       sigmoid=False,
                                       )
                                ],
                        name='loss_rle',
                    ),
                    dict(methods=[dict(type='mmdet.CrossEntropyLoss',
                                       loss_weight=0.75
                                       )
                                ],
                        name='loss_angle_ce',
                    ),
                    dict(methods=[dict(type='MSELoss',
                                       loss_weight=1.0
                                       )
                                ],
                        name='loss_angle_mse',
                    ),
                ],
        loss=None,
        decoder=codec,
        scale_norm_std=True,
        beta=10.),
    test_cfg=dict(flip_test=False, ))

# base dataset settings
dataset_type = 'InshotDataset181'
data_mode = 'topdown'
data_root = '/data/xiaoshuai/facial_landmark_181/train_0310/'

backend_args = dict(backend='local')

# pipelines
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='TopdownAlignV4_KPTS181', 
         input_size=codec['input_size'], 
         mean_face_path=mean_face,
         point_set_types=['val'],
         jitter_prob=0.0,
         interpolation_method='linear',
         ),
    dict(type='PackPoseInputs')
]

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomDirectionalMasking', 
         mask_prob=0.1, 
         a_min=0.1, 
         a_max=0.5,
         min_visible_keypoints_ratio=0.5),
    dict(type='TopdownAlignV4_KPTS181', 
         input_size=codec['input_size'], 
         mean_face_path=mean_face,
         point_set_types = ['four'],
         point_set_weights = [1.0],
         max_tries=5,
         margin_ratio = 0.016,
         scale_range=(0.9, 1.1),
         rotation_range=(-15, 15),
         translation_params=(0, 6),
         jitter_prob=0.875
         ),
    dict(type='EyeConstrainedCoarseDropout', 
         prob=0.5,
         max_occlusions=3,
         min_size=0.1,
         max_size=0.3,
         random_color=False,
         left_eye_indices=list(range(68, 92)),
         right_eye_indices=list(range(92, 116)),
         min_visible_ratio=0.8
         ),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
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
    times=1
)

dataset_kpts235 = dict(
    type=dataset_type,
    data_root='/data/xiaoshuai/facial_landmark_181/train_0326/',
    data_mode=data_mode,
    ann_file='annotations/train_angles_annotations_181.json',
    data_prefix=dict(img='images/'),
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
    type=dataset_type,
    data_root='/data/xiaoshuai/facial_landmark_181/train_0326/',
    data_mode=data_mode,
    ann_file='annotations/val_angles_annotations_181.json',
    data_prefix=dict(img='val_images/'),
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

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='NME', rule='less', max_keep_ckpts=3, interval=1))

# evaluators
val_evaluator = [dict(type='NME', norm_mode='keypoint_distance',), dict(type='EulerMAE',)]
test_evaluator = val_evaluator

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])