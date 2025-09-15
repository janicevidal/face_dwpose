_base_ = ['/home/zhangxiaoshuai/Project/face_dwpose/mmpose/projects/rtmpose/rtmpose/face_2d_keypoint/rtmpose_b256_inshot-96x96_strideformer_lite_relu.py']

# model settings
find_unused_parameters = False

# config settings
fea = True
logit = True

# method details
model = dict(
    _delete_ = True,
    type='PoseEstimatorDistiller',
    teacher_pretrained = '/home/zhangxiaoshuai/Checkpoint/FacialLandmark/rtmpose_b256_inshot-96x96_strideformer/best_NME_epoch_297.pth',
    teacher_cfg = '/home/zhangxiaoshuai/Project/face_dwpose/mmpose/projects/rtmpose/rtmpose/face_2d_keypoint/rtmpose_b256_inshot-96x96_strideformer.py',
    student_cfg = '/home/zhangxiaoshuai/Project/face_dwpose/mmpose/projects/rtmpose/rtmpose/face_2d_keypoint/rtmpose_b256_inshot-96x96_strideformer_lite.py',
    distill_cfg = [dict(methods=[dict(type='FeaLoss',
                                       name='loss_fea',
                                       use_this = fea,
                                       student_channels = 128,
                                       teacher_channels = 256,
                                       alpha_fea=0.00007,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_logit',
                                       use_this = logit,
                                       weight = 0.1,
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