# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import cv2
import random
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine import is_seq_of

from mmpose.registry import TRANSFORMS
from mmpose.structures.bbox import get_udp_warp_matrix, get_warp_matrix


@TRANSFORMS.register_module()
class TopdownAffine(BaseTransform):
    """Get the bbox image as the model input by affine transform.

    Required Keys:

        - img
        - bbox_center
        - bbox_scale
        - bbox_rotation (optional)
        - keypoints (optional)

    Modified Keys:

        - img
        - bbox_scale

    Added Keys:

        - input_size
        - transformed_keypoints

    Args:
        input_size (Tuple[int, int]): The input image size of the model in
            [w, h]. The bbox region will be cropped and resize to `input_size`
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 use_udp: bool = False) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.use_udp = use_udp

    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        """Reshape the bbox to a fixed aspect ratio.

        Args:
            bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.darray: The reshaped bbox scales in (n, 2)
        """

        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        results['bbox_scale'] = self._fix_aspect_ratio(
            results['bbox_scale'], aspect_ratio=w / h)

        # TODO: support multi-instance
        assert results['bbox_center'].shape[0] == 1, (
            'Top-down heatmap only supports single instance. Got invalid '
            f'shape of bbox_center {results["bbox_center"].shape}.')

        center = results['bbox_center'][0]
        scale = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            rot = results['bbox_rotation'][0]
        else:
            rot = 0.

        if self.use_udp:
            warp_mat = get_udp_warp_matrix(
                center, scale, rot, output_size=(w, h))
        else:
            warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

        if isinstance(results['img'], list):
            results['img'] = [
                cv2.warpAffine(
                    img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
                for img in results['img']
            ]
        else:
            results['img'] = cv2.warpAffine(
                results['img'], warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        if results.get('keypoints', None) is not None:
            transformed_keypoints = results['keypoints'].copy()
            # Only transform (x, y) coordinates
            transformed_keypoints[..., :2] = cv2.transform(
                results['keypoints'][..., :2], warp_mat)
            results['transformed_keypoints'] = transformed_keypoints

        results['input_size'] = (w, h)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'use_udp={self.use_udp})'
        return repr_str


@TRANSFORMS.register_module()    
class TopdownAlign(BaseTransform):

    def __init__(self,
                 input_size: Tuple[int, int],
                 offset: float = 0.05,
                 shift_prob: float = 0.75,
                 orient_prob: float = 0.75) -> None:
        super().__init__()

        assert is_seq_of(input_size, int) and len(input_size) == 2, (
            f'Invalid input_size {input_size}')

        self.input_size = input_size
        self.offset = offset
        self.shift_prob = shift_prob
        self.orient_prob = orient_prob

    def transformation_from_points(self, points1, points2):
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

        U, S, Vt = np.linalg.svd(points1.T * points2) 
        R = (U * Vt).T  
        M = (s2 / s1) * R 
        B = c2.T - (s2 / s1) * R * c1.T
        M_inv = M.I
        B_inv = - B 
        return np.hstack((M, B)), M_inv, B_inv

    def warp_im_Mine(self, img_im, orgi_landmarks, net_size):
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
        else:
            tar_landmarks = np.zeros(shape=(4, 2), dtype=np.float32)
            assert False, (f'Invalid input_size {net_size}')

        pts1 = np.float32(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
        pts2 = np.float32(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
        MB, M_inv, B_inv = self.transformation_from_points(pts1, pts2)
        dst = cv2.warpAffine(img_im, MB, net_size)
        return dst, MB, M_inv, B_inv

    def random_shift_fun(self, four_ldms: np.ndarray, offset_maxv):
        """
        对四个关键点进行随机偏移，每个点在以其原始位置为圆心的圆形邻域内随机偏移
        
        参数:
            four_ldms: 形状为 (4, 2) 的 numpy 数组，包含四个关键点的坐标
            offset_maxv: 一个元组 (max_offset_x, max_offset_y)，表示在 x 和 y 方向上的最大偏移半径
            orient: 保留参数，为了兼容性，但不再使用
        
        返回:
            偏移后的四个关键点坐标，形状与输入相同
        """
        
        assert four_ldms.shape == (4, 2)
        results = np.copy(four_ldms)  # 创建输入的副本
        max_offset_x, max_offset_y = offset_maxv  # 解包最大偏移半径
        
        # 为每个点生成独立的随机偏移
        for i in range(4):
            # 生成随机角度 (0 到 2π)
            angle = random.uniform(0, 2 * np.pi)
            
            # 生成随机半径 (0 到 max_offset_x 和 max_offset_y 的较小值)
            # 这样可以确保偏移不会超出圆形区域
            max_radius = min(max_offset_x, max_offset_y)
            radius = random.uniform(0, max_radius)
            
            # 计算偏移量 (使用极坐标到笛卡尔坐标的转换)
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            
            # 应用偏移
            results[i, 0] += dx
            results[i, 1] += dy
        
        return results
    
    def random_orient_fun(self, four_ldms: np.ndarray, offset_maxv, orient="top"):
        """
        在指定随机方向上对四个点进行带随机 offset 的偏移。
        
        参数:
            four_ldms: 形状为 (4, 2) 的 numpy 数组，表示四个点的坐标。
            offset_maxv: 最大偏移量，是一个元组 (max_offset_x, max_offset_y)。
            orient: 偏移方向，可以是以下之一：
                "top", "bottom", "left", "right",
                "top left", "top right", "bottom left", "bottom right"
                如果为 None，则不进行偏移。

        返回:
            偏移后的四个点坐标，形状与输入相同。
        """
        if orient is None:  # 不做任何偏移
            return four_ldms

        assert four_ldms.shape == (4, 2)
        results = np.copy(four_ldms)
        max_offset_x, max_offset_y = offset_maxv

        offsets = np.zeros(shape=four_ldms.shape, dtype=four_ldms.dtype)
        
        # 根据方向设置偏移量
        if orient == "top":
            offsets[:, 1] -= np.random.random(size=4) * max_offset_y
            offsets[:, 0] += (np.random.randn(4) * max_offset_x / 8)
        elif orient == "bottom":
            offsets[:, 1] += np.random.random(size=4) * max_offset_y
            offsets[:, 0] += (np.random.randn(4) * max_offset_x / 8)
        elif orient == "left":
            offsets[:, 0] -= np.random.random(size=4) * max_offset_x
            offsets[:, 1] += (np.random.randn(4) * max_offset_y / 8)
        elif orient == "right":
            offsets[:, 0] += np.random.random(size=4) * max_offset_x
            offsets[:, 1] += (np.random.randn(4) * max_offset_y / 8)
        elif orient == "top left":
            offsets[:, 0] -= np.random.random(size=4) * max_offset_x
            offsets[:, 1] -= np.random.random(size=4) * max_offset_y
        elif orient == "top right":
            offsets[:, 0] += np.random.random(size=4) * max_offset_x
            offsets[:, 1] -= np.random.random(size=4) * max_offset_y
        elif orient == "bottom left":
            offsets[:, 0] -= np.random.random(size=4) * max_offset_x
            offsets[:, 1] += np.random.random(size=4) * max_offset_y
        elif orient == "bottom right":
            offsets[:, 0] += np.random.random(size=4) * max_offset_x
            offsets[:, 1] += np.random.random(size=4) * max_offset_y
        else:
            raise ValueError(f"Unsupported orientation: {orient}")

        results += offsets
        return results

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`TopdownAffine`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        w, h = self.input_size
        
        LEFT_PUPIL_IDX = 201  # 左眼瞳孔
        RIGHT_PUPIL_IDX = 202  # 右眼瞳孔
        LEFT_MOUTH_IDX = 141  # 左嘴角
        RIGHT_MOUTH_IDX = 159  # 右嘴角
        
        if results.get('keypoints', None) is not None:
            keypoints = results['keypoints'][0]
            
            four_landmarks = np.zeros((4, 2), dtype=np.float32)
            four_landmarks[0, 0] = keypoints[LEFT_PUPIL_IDX][0]  # w_leye
            four_landmarks[0, 1] = keypoints[LEFT_PUPIL_IDX][1]  # h_leye
            four_landmarks[1, 0] = keypoints[RIGHT_PUPIL_IDX][0]  # w_reye
            four_landmarks[1, 1] = keypoints[RIGHT_PUPIL_IDX][1]  # h_leye
            four_landmarks[2, 0] = keypoints[LEFT_MOUTH_IDX][0]  # w_lmouth
            four_landmarks[2, 1] = keypoints[LEFT_MOUTH_IDX][1]  # h_lmouth
            four_landmarks[3, 0] = keypoints[RIGHT_MOUTH_IDX][0]  # w_rmouth
            four_landmarks[3, 1] = keypoints[RIGHT_MOUTH_IDX][1]  # h_rmouth
        
            x1, y1, w, h = results['bbox'][0]
            max_offset_v = (w * self.offset, h * self.offset)
            
            four_landmarks_shiftted = np.copy(four_landmarks)

            if random.random() < self.shift_prob:
                four_landmarks_shiftted = self.random_shift_fun(four_ldms=four_landmarks, offset_maxv=max_offset_v)
            
            if random.random() < self.orient_prob:
                OFFSET_ORIENTS_ALL = [None, "top", "bottom", "left", "right", "top left", "top right", "bottom left", "bottom right"]
                offset_orient = random.choice(OFFSET_ORIENTS_ALL)
                four_landmarks_shiftted = self.random_orient_fun(four_ldms=four_landmarks_shiftted, offset_maxv=max_offset_v, orient=offset_orient)
            
            img_align, MB, M_inv, B_inv = self.warp_im_Mine(results['img'], four_landmarks_shiftted, self.input_size)
            
            landmarks_align = []
            keypoints = results['keypoints'][0]
            for i in range(len(keypoints)): 
                align_x = keypoints[i][0] * MB[0, 0] + keypoints[i][1] * MB[0, 1] + MB[0, 2]
                align_y = keypoints[i][0] * MB[1, 0] + keypoints[i][1] * MB[1, 1] + MB[1, 2]
                landmarks_align.append((align_x, align_y))
            
            results['transformed_keypoints'] = np.array([landmarks_align])
            
            results['img'] = img_align
            
            results['M_inv'] = np.array([M_inv])
            results['B_inv'] = np.array([B_inv])
        
        results['input_size'] = (w, h)

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        return repr_str
    

@TRANSFORMS.register_module()  
class HideAndSeek(BaseTransform):
    """Augmentation by informantion dropping in Hide-and-Seek paradigm. Paper
    ref: Huang et al. AID: Pushing the Performance Boundary of Human Pose
    Estimation with Information Dropping Augmentation (arXiv:2008.07139 2020).

    Args:
        prob (float): Probability of performing hide-and-seek.
        prob_hiding_patches (float): Probability of hiding patches.
        grid_sizes (list): List of optional grid sizes.
    """

    def __init__(self,
                 prob=1.0,
                 prob_hiding_patches=0.5,
                 grid_sizes=(0, 16, 32, 44, 56)):
        self.prob = prob
        self.prob_hiding_patches = prob_hiding_patches
        self.grid_sizes = grid_sizes

    def _hide_and_seek(self, img):
        # get width and height of the image
        height, width, _ = img.shape

        # randomly choose one grid size
        index = np.random.randint(0, len(self.grid_sizes) - 1)
        grid_size = self.grid_sizes[index]

        # hide the patches
        if grid_size != 0:
            for x in range(0, width, grid_size):
                for y in range(0, height, grid_size):
                    x_end = min(width, x + grid_size)
                    y_end = min(height, y + grid_size)
                    if np.random.rand() <= self.prob_hiding_patches:
                        img[x:x_end, y:y_end, :] = 0
        return img

    def transform(self, results: Dict) -> Optional[dict]:
        img = results['img']
        if np.random.rand() < self.prob:
            img = self._hide_and_seek(img)
        results['img'] = img
        return results


@TRANSFORMS.register_module()
class RandomDirectionalMasking(BaseTransform):
    """Randomly mask one side or corner of the image based on face bounding box.

    This transform randomly selects one of eight directions/corners (left, right, 
    top, bottom, top-left, top-right, bottom-left, bottom-right) and masks the 
    image region on that side/corner of the face bounding box.
    It ensures that at least half of the keypoints remain visible after masking.
    
    Required Keys:
        - img
        - bbox
        - keypoints (to ensure at least half remain visible)

    Modified Keys:
        - img

    Args:
        mask_prob (float): Probability of applying the masking. Default: 1.0
        a_min (float): Minimum percentage of face width/height to mask. Default: 0.1
        a_max (float): Maximum percentage of face width/height to mask. Default: 0.5
        use_corners (bool): Whether to include corner directions. Default: True
        independent_corners (bool): Whether to use independent a values for horizontal
            and vertical directions in corner masking. Default: True
        corner_prob (float): Probability of selecting a corner direction when 
            use_corners is True. Default: 0.2
        min_visible_keypoints_ratio (float): Minimum ratio of keypoints that must 
            remain visible after masking. Default: 0.5
        max_adjustment_iterations (int): Maximum number of iterations to adjust
            a values to satisfy visibility constraint. Default: 10
    """

    def __init__(self,
                 mask_prob: float = 1.0,
                 a_min: float = 0.1,
                 a_max: float = 0.5,
                 use_corners: bool = True,
                 independent_corners: bool = True,
                 corner_prob: float = 0.2,
                 min_visible_keypoints_ratio: float = 0.5,
                 max_adjustment_iterations: int = 10) -> None:
        super().__init__()
        self.mask_prob = mask_prob
        self.a_min = a_min
        self.a_max = a_max
        self.use_corners = use_corners
        self.independent_corners = independent_corners
        self.corner_prob = corner_prob
        self.min_visible_keypoints_ratio = min_visible_keypoints_ratio
        self.max_adjustment_iterations = max_adjustment_iterations
        
        # Define available directions
        self.base_directions = ['left', 'right', 'top', 'bottom']
        self.corner_directions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        
        if use_corners:
            # When use_corners is True, we have both base and corner directions
            self.all_directions = self.base_directions + self.corner_directions
        else:
            # When use_corners is False, only base directions
            self.all_directions = self.base_directions

    def _select_direction(self) -> str:
        """Select a direction with probability control.
        
        Returns:
            str: Selected direction
        """
        if not self.use_corners or random.random() >= self.corner_prob:
            # Select from base directions
            return random.choice(self.base_directions)
        else:
            # Select from corner directions
            return random.choice(self.corner_directions)

    def transform(self, results: Dict) -> Optional[dict]:
        """Apply random directional masking to the image.

        Args:
            results (dict): The result dict containing 'img', 'bbox', and 'keypoints'

        Returns:
            dict: The result dict with masked image
        """
        if random.random() > self.mask_prob:
            return results

        # Check if keypoints exist
        if 'keypoints' not in results:
            raise ValueError("RandomDirectionalMasking requires 'keypoints' in results dict")

        img = results['img']
        bbox = results['bbox']
        keypoints = results['keypoints'][0]  # Assuming single instance
        
        if isinstance(img, list):
            # If img is a list, apply masking to each image
            for i in range(len(img)):
                img[i] = self._apply_masking_with_constraint(img[i], bbox, keypoints)
        else:
            img = self._apply_masking_with_constraint(img, bbox, keypoints)
        
        results['img'] = img
        return results

    def _apply_masking_with_constraint(self, img: np.ndarray, bbox: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Apply masking to a single image while ensuring at least half of keypoints remain visible.

        Args:
            img (np.ndarray): Input image
            bbox (np.ndarray): Bounding box in format [x1, y1, w, h]
            keypoints (np.ndarray): Keypoints in format (N, 2) or (N, 3)

        Returns:
            np.ndarray: Masked image
        """
        img_h, img_w = img.shape[:2]
        x1, y1, w, h = bbox[0]  # Assuming single instance
        
        # Select masking direction with probability control
        direction = self._select_direction()
        
        # Calculate initial a values
        if direction in self.base_directions:
            a_values = [random.uniform(self.a_min, self.a_max)]
        else:
            if self.independent_corners:
                a_values = [random.uniform(self.a_min, self.a_max), 
                           random.uniform(self.a_min, self.a_max)]
            else:
                a_val = random.uniform(self.a_min, self.a_max)
                a_values = [a_val, a_val]
        
        # Try to apply masking with current parameters
        masked_img, mask_regions = self._apply_masking_with_region(img, bbox, direction, a_values)
        
        # Check visibility constraint
        visible_ratio = self._calculate_visible_keypoints_ratio(keypoints, mask_regions)
        
        # If constraint not satisfied, try to adjust a values
        iteration = 0
        while visible_ratio < self.min_visible_keypoints_ratio and iteration < self.max_adjustment_iterations:
            # Reduce a values to make mask smaller
            a_values = self._adjust_a_values(a_values, iteration)
            
            # Reapply masking with adjusted a values
            masked_img, mask_regions = self._apply_masking_with_region(img, bbox, direction, a_values)
            
            # Recalculate visible ratio
            visible_ratio = self._calculate_visible_keypoints_ratio(keypoints, mask_regions)
            iteration += 1
        
        # If still not satisfied, fallback to no masking
        if visible_ratio < self.min_visible_keypoints_ratio:
            return img
        
        return masked_img

    def _apply_masking_with_region(self, img: np.ndarray, bbox: np.ndarray, 
                                  direction: str, a_values: list) -> tuple:
        """Apply masking and return both the masked image and mask region coordinates.
        
        Args:
            img (np.ndarray): Input image
            bbox (np.ndarray): Bounding box in format [x1, y1, w, h]
            direction (str): Masking direction
            a_values (list): List of a values for masking
            
        Returns:
            tuple: (masked_image, list_of_mask_regions)
        """
        img_h, img_w = img.shape[:2]
        x1, y1, w, h = bbox[0]
        
        # Create a copy of the image
        masked_img = img.copy()
        
        # Initialize mask regions list
        mask_regions = []
        
        # Define masking region based on direction
        if direction == 'left':
            a = a_values[0]
            mask_start_x = 0
            mask_end_x = int(min(x1 + a * w, img_w))
            masked_img[:, mask_start_x:mask_end_x] = 0
            mask_regions.append(('rect', (0, 0, mask_end_x, img_h)))
        
        elif direction == 'right':
            a = a_values[0]
            mask_start_x = max(0, int(x1 + w - a * w))
            mask_end_x = img_w
            masked_img[:, mask_start_x:mask_end_x] = 0
            mask_regions.append(('rect', (mask_start_x, 0, img_w, img_h)))
        
        elif direction == 'top':
            a = a_values[0]
            mask_start_y = 0
            mask_end_y = int(min(y1 + a * h, img_h))
            masked_img[mask_start_y:mask_end_y, :] = 0
            mask_regions.append(('rect', (0, 0, img_w, mask_end_y)))
        
        elif direction == 'bottom':
            a = a_values[0]
            mask_start_y = max(0, int(y1 + h - a * h))
            mask_end_y = img_h
            masked_img[mask_start_y:mask_end_y, :] = 0
            mask_regions.append(('rect', (0, mask_start_y, img_w, img_h)))
        
        elif direction == 'top-left':
            a_h, a_v = a_values[0], a_values[1]
            
            # Apply left mask
            mask_start_x_left = 0
            mask_end_x_left = int(min(x1 + a_h * w, img_w))
            masked_img[:, mask_start_x_left:mask_end_x_left] = 0
            mask_regions.append(('rect', (0, 0, mask_end_x_left, img_h)))
            
            # Apply top mask
            mask_start_y_top = 0
            mask_end_y_top = int(min(y1 + a_v * h, img_h))
            masked_img[mask_start_y_top:mask_end_y_top, :] = 0
            mask_regions.append(('rect', (0, 0, img_w, mask_end_y_top)))
        
        elif direction == 'top-right':
            a_h, a_v = a_values[0], a_values[1]
            
            # Apply right mask
            mask_start_x_right = max(0, int(x1 + w - a_h * w))
            mask_end_x_right = img_w
            masked_img[:, mask_start_x_right:mask_end_x_right] = 0
            mask_regions.append(('rect', (mask_start_x_right, 0, img_w, img_h)))
            
            # Apply top mask
            mask_start_y_top = 0
            mask_end_y_top = int(min(y1 + a_v * h, img_h))
            masked_img[mask_start_y_top:mask_end_y_top, :] = 0
            mask_regions.append(('rect', (0, 0, img_w, mask_end_y_top)))
        
        elif direction == 'bottom-left':
            a_h, a_v = a_values[0], a_values[1]
            
            # Apply left mask
            mask_start_x_left = 0
            mask_end_x_left = int(min(x1 + a_h * w, img_w))
            masked_img[:, mask_start_x_left:mask_end_x_left] = 0
            mask_regions.append(('rect', (0, 0, mask_end_x_left, img_h)))
            
            # Apply bottom mask
            mask_start_y_bottom = max(0, int(y1 + h - a_v * h))
            mask_end_y_bottom = img_h
            masked_img[mask_start_y_bottom:mask_end_y_bottom, :] = 0
            mask_regions.append(('rect', (0, mask_start_y_bottom, img_w, img_h)))
        
        elif direction == 'bottom-right':
            a_h, a_v = a_values[0], a_values[1]
            
            # Apply right mask
            mask_start_x_right = max(0, int(x1 + w - a_h * w))
            mask_end_x_right = img_w
            masked_img[:, mask_start_x_right:mask_end_x_right] = 0
            mask_regions.append(('rect', (mask_start_x_right, 0, img_w, img_h)))
            
            # Apply bottom mask
            mask_start_y_bottom = max(0, int(y1 + h - a_v * h))
            mask_end_y_bottom = img_h
            masked_img[mask_start_y_bottom:mask_end_y_bottom, :] = 0
            mask_regions.append(('rect', (0, mask_start_y_bottom, img_w, img_h)))
        
        return masked_img, mask_regions

    def _calculate_visible_keypoints_ratio(self, keypoints: np.ndarray, mask_regions: list) -> float:
        """Calculate the ratio of keypoints that are not in the masked region.
        
        Args:
            keypoints (np.ndarray): Keypoints in format (N, 2) or (N, 3)
            mask_regions (list): List of mask region descriptions (type, coordinates)
            
        Returns:
            float: Ratio of visible keypoints (0.0 to 1.0)
        """
        if not mask_regions:
            return 1.0
        
        total_keypoints = len(keypoints)
        visible_count = 0
        
        for kp in keypoints:
            x, y = kp[0], kp[1]
            is_visible = True
            
            # Check if keypoint is outside all mask regions
            for region_type, region_coords in mask_regions:
                if region_type == 'rect':
                    x1, y1, x2, y2 = region_coords
                    # If keypoint is inside this rectangle, it's not visible
                    if x1 <= x < x2 and y1 <= y < y2:
                        is_visible = False
                        break
            
            if is_visible:
                visible_count += 1
        
        return visible_count / total_keypoints if total_keypoints > 0 else 1.0

    def _adjust_a_values(self, a_values: list, iteration: int) -> list:
        """Reduce a values to make mask smaller.
        
        Args:
            a_values (list): Current a values
            iteration (int): Current adjustment iteration
            
        Returns:
            list: Adjusted a values
        """
        adjusted_values = []
        
        for a in a_values:
            # Reduce a value by a factor, but ensure it doesn't go below a_min/2
            reduction_factor = 0.7 ** (iteration + 1)  # Exponential reduction
            new_a = a * reduction_factor
            adjusted_values.append(max(self.a_min * 0.5, new_a))
        
        return adjusted_values

    def __repr__(self) -> str:
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(mask_prob={self.mask_prob}, '
        repr_str += f'a_min={self.a_min}, '
        repr_str += f'a_max={self.a_max}, '
        repr_str += f'use_corners={self.use_corners}, '
        repr_str += f'independent_corners={self.independent_corners}, '
        repr_str += f'corner_prob={self.corner_prob}, '
        repr_str += f'min_visible_keypoints_ratio={self.min_visible_keypoints_ratio}, '
        repr_str += f'max_adjustment_iterations={self.max_adjustment_iterations})'
        return repr_str