import cv2
import numpy as np
import onnxruntime
from typing import List, Tuple
 
    
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


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

    
class PreAlignDWPOSE:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = 'landmark'
        self.batched = False
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CPUExecutionProvider'])
        
        self.input_size = (96, 96)
        
        input_cfg = self.session.get_inputs()[0]
        # input_shape = input_cfg.shape
        # if isinstance(input_shape[2], str):
        #     self.input_size = None
        # else:
        #     self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        
    
    def warp_im_Mine(self, img_im, orgi_landmarks, net_size=(96, 96)):
        if net_size == (96, 96):
            tar_landmarks = np.array([[34.3054, 30.7081],  
                                [61.1536, 30.7081],
                                [36.2007, 60.3844],
                                [59.2562, 60.3844]], dtype=np.float32)
        elif net_size == (128, 128):
            tar_landmarks = np.array([[44.2261412, 51.89856756],
                                [82.18832764, 51.87374804],
                                [46.83327506, 91.91344941],
                                [79.52241093, 91.87033527]], dtype=np.float32)

        pts1 = np.float32(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
        pts2 = np.float32(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
        MB, M_inv, B_inv = transformation_from_points(pts1, pts2)
        dst = cv2.warpAffine(img_im, MB, net_size)
        return dst, MB, M_inv, B_inv
 
    
    def preprocess(self, img: np.ndarray, ldm: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            ldm (np.ndarray): known ldms four landmarks.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
                
        four_landmarks = np.zeros((4, 2), dtype=np.float32)
        four_landmarks[0, 0] = ldm[0]  # w_leye
        four_landmarks[0, 1] = ldm[1]  # h_leye
        four_landmarks[1, 0] = ldm[2]  # w_reye
        four_landmarks[1, 1] = ldm[3]  # h_leye
        four_landmarks[2, 0] = ldm[4]  # w_lmouth
        four_landmarks[2, 1] = ldm[5]  # h_lmouth
        four_landmarks[3, 0] = ldm[6]  # w_rmouth
        four_landmarks[3, 1] = ldm[7]  # h_rmouth

        img_align, MB, M_inv, B_inv = self.warp_im_Mine(img, four_landmarks, net_size=self.input_size)  # MB (2,3)
        
        # cv2.imwrite("align.jpg", img_align)
        
        # img_align = cv2.imread("/home/zhangxiaoshuai/Project/face_dwpose/test/align.jpg")
        # normalize image
        if img_align.shape[2] == 3:  # 确保是彩色图像
            img_align = cv2.cvtColor(img_align, cv2.COLOR_BGR2RGB)
                               
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_align = (img_align - mean) / std

        return img_align.astype(np.float32), MB, M_inv, B_inv

    def postprocess(
        self,
        outputs,
        M_inv,
        B_inv,
        simcc_split_ratio = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = np.expand_dims((np.matmul(M_inv, (keypoints[0]).T + B_inv)).T, axis=0)

        return keypoints, scores

    def forward(self, img):
        input_size = tuple(img.shape[0:2][::-1])
        
        blob = img.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)
        
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        
        simcc_x_numpy = net_outs[0]
        simcc_y_numpy = net_outs[1]
        
        return (simcc_x_numpy, simcc_y_numpy)

    def infer(self, img, ldm):
        img_align, MB, M_inv, B_inv = self.preprocess(img, ldm)
        
        outputs = self.forward(img_align)
        
        kpts, score = self.postprocess(outputs, M_inv, B_inv, simcc_split_ratio=1.5)
        
        return kpts, score