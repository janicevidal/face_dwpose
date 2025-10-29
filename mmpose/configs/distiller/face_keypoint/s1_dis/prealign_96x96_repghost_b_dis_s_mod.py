_base_ = ['/home/zhangxiaoshuai/Project/face_dwpose/mmpose/configs/face_2d_keypoint_prealign/prealign_96x96_repghost_lite_single_ema.py']

# model settings
find_unused_parameters = False

# config settings
fea = True
logit = True

# method details
model = dict(
    _delete_ = True,
    type='PoseEstimatorDistiller',
    teacher_pretrained = '/home/zhangxiaoshuai/Checkpoint/FacialLandmark/prealign_96x96_repghost_base_single_ema/best_NME_epoch_326.pth',
    teacher_cfg = '/home/zhangxiaoshuai/Project/face_dwpose/mmpose/configs/face_2d_keypoint_prealign/prealign_96x96_repghost_base_single_ema.py',
    student_cfg = '/home/zhangxiaoshuai/Project/face_dwpose/mmpose/configs/face_2d_keypoint_prealign/prealign_96x96_repghost_lite_single_ema.py',
    distill_cfg = [dict(methods=[dict(type='FeaLoss',
                                       name='loss_fea',
                                       use_this = fea,
                                       student_channels = 96,
                                       teacher_channels = 192,
                                       alpha_fea=0.3,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_logit',
                                       use_this = logit,
                                       weight = 3.0,
                                       )
                                ]
                        ),
                    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)
optim_wrapper = dict(
    clip_grad=dict(max_norm=1., norm_type=2))