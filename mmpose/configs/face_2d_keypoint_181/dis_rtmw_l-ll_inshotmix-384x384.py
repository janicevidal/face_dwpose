_base_ = ['./rtmw-l_b128-120e_inshotmix-384x384_finetune_body_pretrain_neck_refine_box1_2.py']

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=60, val_interval=1)

# method details
model = dict(
    _delete_ = True,
    type='PoseEstimatorDistiller',
    two_dis = second_dis,
    teacher_pretrained = '/home/zhangxiaoshuai/Checkpoint/FacialLandmark181/rtmw-l_b128-120e_inshotmix-384x384_finetune_body_pretrain_neck_refine_box1_2/best_NME_epoch_120.pth',
    teacher_cfg = 'configs/face_2d_keypoint_181/rtmw-l_b128-120e_inshotmix-384x384_finetune_body_pretrain_neck_refine_box1_2.py',
    student_cfg = 'configs/face_2d_keypoint_181/rtmw-l_b128-120e_inshotmix-384x384_finetune_body_pretrain_neck_refine_box1_2.py',
    distill_cfg = [
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_logit',
                                       use_this = logit,
                                       weight = 1,
                                       )
                                ]
                        ),
                    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    train_cfg=train_cfg,
)

optim_wrapper = dict(
    clip_grad=dict(max_norm=1., norm_type=2))

work_dir='/home/zhangxiaoshuai/Checkpoint/FacialLandmark181/dis_rtmw_l-ll_inshotmix-384x384'