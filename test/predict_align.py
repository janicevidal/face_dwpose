import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from utils import load_test_data, py_cpu_nms
from models import RetinaFace
# from prealign_landmark_mnn_infer import PreAlignDWPOSE
from prealign_landmark_onnx_infer import PreAlignDWPOSE
from models.retinaface import decode, decode_landm, PriorBox


class FaceAlign(object):
    def __init__(self, pts235_modelf, device="cuda:0"):
        self.DEFAULT_DEVICE = device if isinstance(device, torch.device) else torch.device(device)

        model_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "weights")
        assert os.path.exists(model_root), "model_root not exists"

        # 人脸检测模型 使用 retinaFace 得到五个基本特征点 用于对齐的仿射变换
        self.cfg_re50 = {
            'name': 'Resnet50',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 24,
            'ngpu': 4,
            'epoch': 100,
            'decay1': 70,
            'decay2': 90,
            'image_size': 840,
            'pretrain': True,
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 256,
            'out_channel': 256
        }
        detect_predictor = RetinaFace(cfg=self.cfg_re50, phase='test')
        retinaFaceModel = os.path.join(model_root, "Resnet50_Final.pth")
        detect_predictor.load_state_dict(self.get_correct_modeldict(retinaFaceModel, device=self.DEFAULT_DEVICE))
        detect_predictor.to(device=self.DEFAULT_DEVICE)
        self.detect_predictor = detect_predictor
        
        # 人脸对齐推理模型
        self.align_predictor = PreAlignDWPOSE(model_file=pts235_modelf)

        # 进度条 
        self.pbar = None


    @torch.inference_mode()
    def pred_one_img(self, imgf, save_dir, show=False):
        assert save_dir and os.path.exists(save_dir), "save_dir not exists"
        # 推理模式 冻结参数
        self.detect_predictor.eval()

        bname = os.path.basename(imgf)

        with torch.no_grad():
            # step1. 首先进行人脸检测+五个特征点
            img_raw = cv2.imread(imgf, cv2.IMREAD_COLOR)
            h, w, c = img_raw.shape
            ImgScale = 1   # 缩放尺寸
            if max(h, w) > 2000:
                ImgScale = max(h, w) // 2000
                img_raw = cv2.resize(img_raw, dsize=(w//ImgScale, h//ImgScale))

            img = np.float32(img_raw)
            target_size = 1600
            max_size = 2150
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(target_size) / float(im_size_min)

            # prevent bigger axis from being more than max_size:
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)

            origin_size = True
            if origin_size:
                resize = 1

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.DEFAULT_DEVICE)

            scale = scale.to(self.DEFAULT_DEVICE)
            loc, conf, landms = self.detect_predictor(img)  # forward pass
            priorbox = PriorBox(self.cfg_re50, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.DEFAULT_DEVICE)
            prior_data = priors.data

            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_re50['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg_re50['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])

            scale1 = scale1.to(self.DEFAULT_DEVICE)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            confidence_threshold = 0.8
            inds = np.where(scores > confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]

            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            nms_threshold = 0.4
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)

            dets = dets[keep, :].tolist()  # [[人脸框x1y1x2y2, 人脸概率], ...]
            landms = (ImgScale * landms[keep]).tolist()  # fixme 分别是 [[左眼xy 右眼xy 鼻尖xy 左嘴角xy 右嘴角xy], ...]

            # print(bname, "Face Num == ", len(dets))
            self.pbar.set_postfix({"Face Num": len(dets)})

            if len(dets) <= 0:
                print("\t\tNot detect face!")
                return
            # todo 找到概率最高的人脸
            vis_thres = 0.5  # 人脸的概率小于 0.5 则忽略
            best_score = -999.999
            idx = -1
            for b in dets:
                t_score = b[4]
                if t_score > best_score and t_score >= vis_thres:
                    idx += 1
                    best_score = t_score

            if idx <= -1:
                print("\t\tdetect face, but all scores < {}!".format(vis_thres))
                return

            # 获取4个人脸特征点 不要鼻尖，顺序为左右眼 左右嘴角xy
            landms = landms[idx]
            four_landmarks = np.asarray(
                [landms[0], landms[1], landms[2], landms[3], landms[6], landms[7], landms[8], landms[9]],
                dtype=np.float32
            )

            # -------------------
            # 下一阶段 人脸对齐
            # -------------------
            img = cv2.imread(imgf)
            kpss, scores = self.align_predictor.infer(img, four_landmarks, head="ipr")
            
            basename = os.path.basename(imgf)
            
            if kpss is not None:
                kps = kpss[0]
                # score = scores[0]
                
                # print(kps[0], score[0])
                # print(kps[1], score[1])
                
                # draw kps
                for kp in kps:
                    # kp_ = kp.astype(np.int32)
                    kp_ = np.around(kp).astype(np.int32)
                    cv2.circle(img, tuple(kp_), 2, (0, 0, 255), 2)

            # 2.点位打到图片上
            if show:
                cv2.imshow("{}".format(basename), img)
                cv2.waitKey()
            else:
                cv2.imwrite(os.path.join(save_dir, "{}-235pts.jpg".format(basename)), img)
                np.save(os.path.join(save_dir, "{}.npy".format(basename)), arr=np.asarray(kpss[0]))
            
            # import pdb
            # pdb.set_trace()


    def pred_images(self, imglist, save_dir, show=False):
        self.pbar = tqdm(imglist)
        for imgf in self.pbar:
            bname = os.path.basename(os.path.splitext(imgf)[0])
            self.pbar.set_description(f"{bname} Predicting...")
            self.pred_one_img(imgf=imgf, show=show, save_dir=save_dir)
   
    @staticmethod
    def get_correct_modeldict(model_file, device=torch.device("cuda:0")):
        """
            之前训练的模型self.detection 使用了nn.Sequential包起来了
            这里统一使用self.detection=nn.x 将训好的模型detection.0.weight --> detection.weight

            对于多个GPU训练的模型 保存为只使用一个GPU训练的

            todo: fixme:
                之前训练的模型中 人脸识别任务(是否有人脸 二分类)采用了 输出1通道+sigmoid
                现在统一为了 输出2通道+softmax --- 此时模型只能重新训练
        """
        # p1 = torch.load(model_file, map_location=device, weights_only=True)
        p1 = torch.load(model_file, map_location=device)

        # todo 1.多GPU训练去掉module.
        single_model_dict = OrderedDict()
        if list(p1.keys())[0].startswith('module.'):
            for k, v in p1.items():
                modified_key = k[7:]  # todo: 去掉key 前面的 “module.” 即可。
                single_model_dict[modified_key] = v
        else:
            single_model_dict = p1

        # todo 2.之前detection以及orient! 用nn.Sequential包起来了 去掉 '.0'
        new_dict = OrderedDict()
        for k, v in single_model_dict.items():
            nk = k
            if k == "detection.0.weight":
                nk = "detection.weight"

            if k == "detection.0.bias":
                nk = "detection.bias"

            if k == "orient.0.weight":
                nk = "orient.weight"

            if k == "orient.0.bias":
                nk = "orient.bias"

            new_dict[nk] = v

        return new_dict


    @staticmethod
    def transformation_from_points(points1, points2):
        points1 = points1.astype(np.float32)  # shape (4, 2)
        points2 = points2.astype(np.float32)  # shape (4, 2)

        c1 = np.mean(points1, axis=0)  # shape (1, 2)
        c2 = np.mean(points2, axis=0)  # shape (1, 2)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)  # single value #standard deviation
        s2 = np.std(points2)  # single value #standard deviation
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)  # points1.T * points2 是2*2的矩阵   ==  2*4的矩阵和4*2的矩阵相乘
        R = (U * Vt).T  # shape (2, 2)
        M = (s2 / s1) * R  # shape (2, 2)
        B = c2.T - (s2 / s1) * R * c1.T  # shape (2, 1)
        M_inv = M.I  # shape (2, 2)
        B_inv = - B  # shape (2, 1)
        return np.hstack((M, B)), M_inv, B_inv


def pred_landmark235():

    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/mnn/prealign_96x96_repghost_lite_single_ema.mnn'
    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/mnn/prealign_96x96_repghost_lite_ema.mnn'
    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/mnn/end2end_strideformer_slim_no_post.mnn'
    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/mnn/end2end_topformer_slim_no_post.mnn'
    pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/ort/end2end_slim.onnx'
    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/ort/end2end_slim_rle_finetune.onnx'
    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/ort/end2end_slim_rle_beta_finetune.onnx'
    
    # pose_model = '/home/zhangxiaoshuai/Project/face_dwpose/mmdeploy_model/mmpose/mnn/prealign_96x96_repghost_lite_two_ema.mnn'
    images, gt_npy_files = load_test_data()

    # images = ["/data/caiachang/video-ldms-ok/TEST/ffhq/00023.jpg"]
    
    # test_savedir = "/data/xiaoshuai/facial_lanmark/vis/test_ipr/"
    # test_savedir = "/data/xiaoshuai/facial_lanmark/vis/test_ipr_finetune/"
    # test_savedir = "/data/xiaoshuai/facial_lanmark/vis/test_ipr_beta_finetune_420/"
    test_savedir = "/data/xiaoshuai/facial_lanmark/vis/test_ipr_beta_scalenorm_finetune_420/"
    if not os.path.exists(test_savedir):
        os.makedirs(test_savedir)

    predictor = FaceAlign(pts235_modelf=pose_model)

    predictor.pred_images(imglist=images, save_dir=test_savedir, show=False)


if __name__ == '__main__':
    pred_landmark235()