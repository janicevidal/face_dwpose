# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, List, Any

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
        
        results['aug_masking'] = True
        if random.random() > self.mask_prob:
            results['aug_masking'] = False
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


@TRANSFORMS.register_module()
class EyeConstrainedCoarseDropout(BaseTransform):
    """Random occlusion augmentation that protects at least half of eye keypoints.
    
    This transform randomly places rectangular occlusion patches on the image,
    ensuring that at least half of the keypoints in each eye region remain visible.
    The occlusion patches are filled with random colors.

    Required Keys:
        - img
        - keypoints (to check eye keypoint visibility)

    Modified Keys:
        - img

    Args:
        prob (float): Probability of applying occlusion. Default: 0.5
        max_occlusions (int): Maximum number of occlusion patches. Default: 3
        min_size (float): Minimum occlusion size relative to image. Default: 0.1
        max_size (float): Maximum occlusion size relative to image. Default: 0.3
        min_aspect_ratio (float): Minimum aspect ratio (width/height). Default: 0.5
        max_aspect_ratio (float): Maximum aspect ratio (width/height). Default: 2.0
        max_attempts (int): Maximum attempts to place each occlusion patch. Default: 10
        random_color (bool): Whether to use random colors. If False, uses zeros. Default: True
        left_eye_indices (List[int]): Left eye contour keypoint indices (77-100). Default: range(77, 101)
        right_eye_indices (List[int]): Right eye contour keypoint indices (101-124). Default: range(101, 125)
        min_visible_ratio (float): Minimum ratio of visible eye keypoints per eye. Default: 0.5
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_occlusions: int = 3,
                 min_size: float = 0.1,
                 max_size: float = 0.3,
                 min_aspect_ratio: float = 0.5,
                 max_aspect_ratio: float = 2.0,
                 max_attempts: int = 10,
                 random_color: bool = True,
                 left_eye_indices: List[int] = None,
                 right_eye_indices: List[int] = None,
                 min_visible_ratio: float = 0.5) -> None:
        super().__init__()
        
        self.prob = prob
        self.max_occlusions = max_occlusions
        self.min_size = min_size
        self.max_size = max_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_attempts = max_attempts
        self.random_color = random_color
        self.min_visible_ratio = min_visible_ratio
        
        # Set default eye indices if not provided
        if left_eye_indices is None:
            self.left_eye_indices = list(range(77, 101))
        else:
            self.left_eye_indices = left_eye_indices
            
        if right_eye_indices is None:
            self.right_eye_indices = list(range(101, 125))
        else:
            self.right_eye_indices = right_eye_indices

    def _generate_random_rect(self, img_h: int, img_w: int) -> Tuple[int, int, int, int]:
        """Generate a random rectangle within image bounds.
        
        Args:
            img_h: Image height
            img_w: Image width
            
        Returns:
            (x1, y1, x2, y2) rectangle coordinates
        """
        # Random size
        size = random.uniform(self.min_size, self.max_size)
        # Random aspect ratio
        aspect_ratio = random.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
        
        # Calculate width and height
        if random.random() > 0.5:
            width = int(size * img_w)
            height = int(width / aspect_ratio)
        else:
            height = int(size * img_h)
            width = int(height * aspect_ratio)
        
        # Ensure minimum dimensions
        width = max(2, width)
        height = max(2, height)
        
        # Ensure width and height don't exceed image bounds
        width = min(width, img_w)
        height = min(height, img_h)
        
        # Random position
        x1 = random.randint(0, img_w - width)
        y1 = random.randint(0, img_h - height)
        x2 = x1 + width
        y2 = y1 + height
        
        return (x1, y1, x2, y2)

    def _generate_random_color(self) -> Tuple[int, int, int]:
        """Generate random RGB color.
        
        Returns:
            (R, G, B) color tuple
        """
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _is_point_in_rect(self, point: Tuple[float, float], rect: Tuple[int, int, int, int]) -> bool:
        """Check if a point is inside a rectangle.
        
        Args:
            point: (x, y) coordinates
            rect: (x1, y1, x2, y2) rectangle coordinates
            
        Returns:
            True if point is inside rectangle
        """
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x < x2 and y1 <= y < y2

    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                           rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap.
        
        Args:
            rect1: (x1, y1, x2, y2)
            rect2: (x1, y1, x2, y2)
            
        Returns:
            True if rectangles overlap
        """
        x1_overlap = max(rect1[0], rect2[0])
        y1_overlap = max(rect1[1], rect2[1])
        x2_overlap = min(rect1[2], rect2[2])
        y2_overlap = min(rect1[3], rect2[3])
        
        return x1_overlap < x2_overlap and y1_overlap < y2_overlap

    def _check_eye_visibility(self, keypoints: np.ndarray, 
                             occlusions: List[Tuple[int, int, int, int]]) -> bool:
        """Check if at least half of each eye's keypoints are visible.
        
        Args:
            keypoints: Keypoints array (N, 2) or (N, 3)
            occlusions: List of occlusion rectangles
            
        Returns:
            True if visibility constraint is satisfied
        """
        # Get eye keypoints
        left_eye_points = keypoints[self.left_eye_indices, :2]
        right_eye_points = keypoints[self.right_eye_indices, :2]
        
        # Count visible points in left eye
        left_visible = 0
        for point in left_eye_points:
            occluded = False
            for rect in occlusions:
                if self._is_point_in_rect(point, rect):
                    occluded = True
                    break
            if not occluded:
                left_visible += 1
        
        # Count visible points in right eye
        right_visible = 0
        for point in right_eye_points:
            occluded = False
            for rect in occlusions:
                if self._is_point_in_rect(point, rect):
                    occluded = True
                    break
            if not occluded:
                right_visible += 1
        
        # Calculate ratios
        left_ratio = left_visible / len(left_eye_points)
        right_ratio = right_visible / len(right_eye_points)
        
        # Check if both eyes meet the minimum visibility ratio
        return left_ratio >= self.min_visible_ratio and right_ratio >= self.min_visible_ratio

    def _apply_occlusions(self, img: np.ndarray, occlusions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Apply occlusion rectangles to image.
        
        Args:
            img: Input image
            occlusions: List of occlusion rectangles
            
        Returns:
            Occluded image
        """
        occluded_img = img.copy()
        
        for rect in occlusions:
            x1, y1, x2, y2 = rect
            
            # Generate random color
            if self.random_color:
                color = self._generate_random_color()
            else:
                color = (0, 0, 0)  # Black
            
            # Apply occlusion
            if len(occluded_img.shape) == 3:  # Color image
                occluded_img[y1:y2, x1:x2, :] = color
            else:  # Grayscale image
                gray_color = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
                occluded_img[y1:y2, x1:x2] = gray_color
        
        return occluded_img

    def transform(self, results: Dict) -> Optional[dict]:
        """Apply random occlusion with eye protection.
        
        Args:
            results: The result dict
            
        Returns:
            The result dict with occluded image
        """
        if results['aug_masking'] is True:
            return results
        
        if random.random() > self.prob:
            return results

        # Check if keypoints exist
        if 'transformed_keypoints' not in results:
            return results

        img = results['img']
        keypoints = results['transformed_keypoints'][0]  # Assuming single instance
        
        img = self._apply_occlusion_to_single_image(img, keypoints)
        
        results['img'] = img
        return results

    def _apply_occlusion_to_single_image(self, img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Apply occlusion to a single image with eye visibility constraint.
        
        Args:
            img: Input image
            keypoints: Keypoints array
            
        Returns:
            Occluded image
        """
        img_h, img_w = img.shape[:2]
        
        # Determine number of occlusions
        num_occlusions = random.randint(1, self.max_occlusions)
        
        # Try to find valid occlusion configuration
        for attempt in range(self.max_attempts):
            occlusions = []
            
            # Generate occlusion rectangles
            for _ in range(num_occlusions):
                for occl_attempt in range(10):  # Try to place each occlusion
                    rect = self._generate_random_rect(img_h, img_w)
                    
                    # Check if rectangle overlaps with existing occlusions
                    overlap = any(
                        self._rectangles_overlap(rect, existing_rect) 
                        for existing_rect in occlusions
                    )
                    
                    if not overlap:
                        occlusions.append(rect)
                        break
            
            # Check if we have enough occlusions
            if len(occlusions) < max(1, num_occlusions // 2):
                continue  # Try again
                
            # Check eye visibility constraint
            if self._check_eye_visibility(keypoints, occlusions):
                # Apply occlusions to image
                return self._apply_occlusions(img, occlusions)
        
        # If no valid configuration found, return original image
        return img

    def __repr__(self) -> str:
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_occlusions={self.max_occlusions}, '
        repr_str += f'min_size={self.min_size}, '
        repr_str += f'max_size={self.max_size}, '
        repr_str += f'min_visible_ratio={self.min_visible_ratio}, '
        repr_str += f'random_color={self.random_color})'
        return repr_str


@TRANSFORMS.register_module()
class TopdownAlignV2(BaseTransform):    
    def __init__(self,
                 input_size: Tuple[int, int],
                 mean_face_path: str,
                 point_set_types: List[str] = ['four', 'five', 'nineteen', 'twenty_three', 'base', 'val'],
                 point_set_weights: Optional[List[float]] = None,
                 max_tries: int = 20,
                 margin_ratio: float = 0.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 rotation_range: Tuple[float, float] = (-10, 10),
                 translation_ratio_range: Tuple[float, float] = (-0.05, 0.05), 
                 jitter_prob: float = 0.8,
                 interpolation_method: str = 'random',  # 'random', 'linear', 'nearest' 
        ) -> None:
        """
        初始化函数
        
        Args:
            input_size: 输出图像尺寸 (width, height)
            mean_face_path: 均值人脸npy文件路径
            point_set_types: 点集类型列表 ['four', 'five', 'nineteen', 'twenty_three', 'base', 'val']
            point_set_weights: 点集选择的权重，None表示均匀分布
            max_tries: 最大尝试次数
            margin_ratio: 边界检查的边距比例
            scale_range: 缩放抖动范围
            rotation_range: 旋转角度抖动范围（度数）
            translation_ratio_range: 平移抖动范围（相对于图像尺寸的比例）
            jitter_prob: 抖动概率
        """
        super().__init__()
        
        self.input_size = input_size
        self.mean_face_path = mean_face_path
        self.point_set_types = point_set_types
        self.max_tries = max_tries
        self.margin_ratio = margin_ratio
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_ratio_range = translation_ratio_range
        self.jitter_prob = jitter_prob
        self.interpolation_method = interpolation_method

        self.mean_face = self._load_mean_face(mean_face_path)
        
        if point_set_weights is None:
            self.point_set_weights = [1.0 / len(point_set_types)] * len(point_set_types)
        else:
            self.point_set_weights = point_set_weights

        self._define_point_sets()
        
        self.margin_w = self.input_size[0] * self.margin_ratio
        self.margin_h = self.input_size[1] * self.margin_ratio
        
        self.tx_range = self.input_size[0] * self.translation_ratio_range[0], self.input_size[0] * self.translation_ratio_range[1]
        self.ty_range = self.input_size[1] * self.translation_ratio_range[0], self.input_size[1] * self.translation_ratio_range[1]
        
        self.target_points = self._preprocess_target_points()
    
    def _load_mean_face(self, path: str) -> np.ndarray:
        mean_face = np.load(path)
        assert mean_face.shape == (235, 2), f"均值人脸应为(235, 2)，实际为{mean_face.shape}"
        mean_face = mean_face.astype(np.float32)

        return mean_face
    
    def _define_point_sets(self):
        self.eye_left_idx = 201
        self.eye_right_idx = 202
        self.nose_tip_idx = 139
        self.mouth_left_idx = 141
        self.mouth_right_idx = 159
        
        contour_idx = list(range(0, 37, 2))
        
        self.point_sets = {
            'four': [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'five': [self.eye_left_idx, self.eye_right_idx, self.nose_tip_idx, self.mouth_left_idx, self.mouth_right_idx],
            'nineteen': contour_idx,
            'twenty_three': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'base': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'val': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx]
        }
    
    def _preprocess_target_points(self) -> np.ndarray:
        target_points = self.mean_face.copy()
        target_points[:, 0] = target_points[:, 0] * self.input_size[0]
        target_points[:, 1] = target_points[:, 1] * self.input_size[1]
        
        return target_points
    
    def similarity_transform_from_points(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, Any]:
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        
        points1_centered = points1 - c1
        points2_centered = points2 - c2
        
        s1 = np.std(points1_centered) + 1e-8 
        s2 = np.std(points2_centered) + 1e-8
        
        scale = s2 / s1
        
        points1_norm = points1_centered / s1
        points2_norm = points2_centered / s2

        H = np.dot(points1_norm.T, points2_norm)
        U, S, Vt = np.linalg.svd(H)
        
        R = (U @ Vt).T
        
        # 确保是纯旋转 没有镜像（det(R) = 1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = (U @ Vt).T
        
        M = scale * R
        B = c2.reshape(2, 1) - np.dot(M, c1.reshape(2, 1))
        
        rotation_rad = np.arctan2(R[1, 0], R[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        affine_matrix = np.hstack((M, B))
        
        M_inv = np.linalg.inv(M)
        # B_inv = -np.dot(M_inv, B)
        B_inv = -B
        
        return {
            'M': M,
            'B': B,
            'R': R,
            'M_inv': M_inv, 
            'B_inv': B_inv,
            'affine_matrix': affine_matrix,
            'scale': scale,                   # 缩放因子
            'rotation_deg': rotation_deg,     # 旋转角度
            'c1': c1,
            'c2': c2
        }
    
    def jitter_transform_params(self, transform_params: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        M = [[a, -b], [b, a]]
        """
        if random.random() > self.jitter_prob:
            return transform_params.copy()

        scale_orig = transform_params['scale']
        rotation_deg_orig = transform_params['rotation_deg']
        translation_orig = transform_params['B']
        
        c1 = transform_params['c1']  # 源点集中心
        c2 = transform_params['c2']  # 目标点集中心
        
        # 抖动缩放
        if random.random() > 0.5:
            scale_jittered = scale_orig * random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale_jittered = scale_orig
        # print("scale ", scale_orig, scale_jittered, scale_jittered/scale_orig)
        
        # 抖动旋转角度
        if random.random() > 0.5:
            rotation_deg_jittered = rotation_deg_orig + random.uniform(self.rotation_range[0], self.rotation_range[1])
            rotation_rad_jittered = np.radians(rotation_deg_jittered)
            cos_theta = np.cos(rotation_rad_jittered)
            sin_theta = np.sin(rotation_rad_jittered)
            R_jittered = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]], dtype=np.float32)
        else:
            rotation_deg_jittered = rotation_deg_orig
            R_jittered = transform_params['R']
        
        if (scale_jittered == scale_orig and rotation_deg_jittered == rotation_deg_orig):
            M_jittered = transform_params['M']
            M_jittered_inv = transform_params['M_inv']
            B_translation = transform_params['B']
        else:
            M_jittered = scale_jittered * R_jittered
            M_jittered_inv = np.linalg.inv(M_jittered)
            B_translation = c2.reshape(2, 1) - np.dot(M_jittered, c1.reshape(2, 1))
        
        # print("deg  ", rotation_deg_orig, rotation_deg_jittered, rotation_deg_jittered- rotation_deg_orig)
        
        # 抖动平移
        if random.random() > 0.5:
            tx_jittered = B_translation[0, 0] + random.uniform(self.tx_range[0], self.tx_range[1])
            ty_jittered = B_translation[1, 0] + random.uniform(self.ty_range[0], self.ty_range[1])
        else:
            tx_jittered, ty_jittered = B_translation[0, 0], B_translation[1, 0]
             
        # print("transx ", tx_jittered, B_translation[0, 0], B_translation[0, 0]-tx_jittered)
        # print("transy ", ty_jittered, B_translation[1, 0], B_translation[1, 0]-ty_jittered)
        
        if (scale_jittered == scale_orig and rotation_deg_jittered == rotation_deg_orig and tx_jittered == translation_orig[0, 0] and ty_jittered == translation_orig[1, 0]):
            return transform_params.copy()
            
        B_jittered = np.array([[tx_jittered], [ty_jittered]], dtype=np.float32)
        B_jittered_inv = -B_jittered
        
        affine_matrix_jittered = np.hstack((M_jittered, B_jittered))
        
        jittered_params = transform_params.copy()
        jittered_params.update({
            'M': M_jittered,
            'B': B_jittered,
            'R': R_jittered,
            'M_inv': M_jittered_inv,
            'B_inv': B_jittered_inv,
            'affine_matrix': affine_matrix_jittered,
            'scale': scale_jittered,
            'rotation_deg': rotation_deg_jittered
        })
        
        return jittered_params
    
    def _count_out_of_bounds_points(self, points: np.ndarray) -> int:
        w, h = self.input_size        
        margin_w, margin_h = self.margin_w, self.margin_h
        
        x_in_bounds = (points[:, 0] >= margin_w) & (points[:, 0] <= w - margin_w)
        y_in_bounds = (points[:, 1] >= margin_h) & (points[:, 1] <= h - margin_h)
        
        in_bounds_mask = x_in_bounds & y_in_bounds
        
        out_of_bounds_count = len(points) - np.sum(in_bounds_mask)
        
        return int(out_of_bounds_count)
    
    def apply_affine_transform(self, img: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        if self.interpolation_method == 'random':
            interpolation = cv2.INTER_LINEAR if random.random() > 0.5 else cv2.INTER_NEAREST
        elif self.interpolation_method == 'linear':
            interpolation = cv2.INTER_LINEAR
        else:  # 'nearest'
            interpolation = cv2.INTER_NEAREST
        
        transformed_img = cv2.warpAffine(img, affine_matrix, self.input_size, flags=interpolation)
        
        return transformed_img
    
    def _transform_points(self, points: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed_points = np.dot(points_homogeneous, affine_matrix.T)
        
        return transformed_points
    
    def _generate_random_nineteen(self) -> List[int]:
        """动态生成随机19点集：每对相邻点随机选一个，18固定"""
        choices = np.random.randint(0, 2, size=18)

        first_half = 2 * np.arange(9) + choices[:9]
        second_half = 20 + 2 * np.arange(9) - choices[9:]
        
        indices = np.concatenate([first_half, [18], second_half])
        
        return indices.tolist()

    def select_point_set(self) -> Tuple[str, List[int]]:
        point_set_type = random.choices(
            self.point_set_types, 
            weights=self.point_set_weights, 
            k=1
        )[0]
        
        if point_set_type in ['nineteen', 'twenty_three']:
            indices = self._generate_random_nineteen()
            
            if point_set_type == 'twenty_three':
                indices = indices + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx]
        else:
            indices = self.point_sets[point_set_type].copy()
        
        return point_set_type, indices
    
    def transform(self, results: Dict) -> Optional[dict]:
        img = results['img']
        
        if results.get('keypoints', None) is not None:
            keypoints = results['keypoints'][0]
        
            src_points_all = np.array(keypoints, dtype=np.float32)

            best_out_of_bounds_count = 235
            match = False
            
            for try_num in range(self.max_tries):
                point_set_type, point_indices = self.select_point_set()
                
                src_selected = src_points_all[point_indices].copy()
                target_selected = self.target_points[point_indices].copy()
                
                try:
                    transform_params = self.similarity_transform_from_points(src_selected, target_selected)
                except np.linalg.LinAlgError:
                    continue  # 矩阵奇异，跳过此次尝试
               
                transform_params = self.jitter_transform_params(transform_params)
                
                transformed_points = self._transform_points(src_points_all, transform_params['affine_matrix'])
                
                out_of_bounds_count = self._count_out_of_bounds_points(transformed_points)
                
                if out_of_bounds_count == 0 or point_set_type == "val":
                    out_transform_params = transform_params
                    out_transformed_points = transformed_points
                    match = True
                    break
                
                if out_of_bounds_count < best_out_of_bounds_count:
                    best_out_of_bounds_count = out_of_bounds_count
                    out_transform_params = transform_params
                    out_transformed_points = transformed_points
                    
            # if not match :
            #     print(out_transform_params)
            #     print(out_transformed_points)
            #     print(best_out_of_bounds_count)
            #     print(results.keys())

            #     for i in range(out_transformed_points.shape[0]):  # 遍历每一行
            #         if out_transformed_points[i, 0] < 0 or out_transformed_points[i, 1] < 0:
            #             print(f"第{i}行：小于0")
            #         if out_transformed_points[i, 0] > 96 or out_transformed_points[i, 1] > 96:
            #             print(f"第{i}行：大于 96")
        
            #     print(results['img_path'])
            #     print(results['keypoints'])
            #     landmarks_path = "out_transformed_points.npy"
            #     np.save(landmarks_path, out_transformed_points)
            #     import pdb
            #     pdb.set_trace()
                
            results['img'] = self.apply_affine_transform(img, out_transform_params['affine_matrix'])
            results['transformed_keypoints'] = np.array([out_transformed_points])
            
            results['M_inv'] = np.array([out_transform_params['M_inv']])
            results['B_inv'] = np.array([out_transform_params['B_inv']])
        
        results['input_size'] = self.input_size
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'point_set_types={self.point_set_types}, '
        repr_str += f'max_tries={self.max_tries}, '
        repr_str += f'jitter_prob={self.jitter_prob})'
        return repr_str


@TRANSFORMS.register_module()
class TopdownAlignV3(BaseTransform):    
    def __init__(self,
                 input_size: Tuple[int, int],
                 mean_face_path: str,
                 point_set_types: List[str] = ['four', 'five', 'nineteen', 'twenty_three', 'base', 'val'],
                 point_set_weights: Optional[List[float]] = None,
                 max_tries: int = 20,
                 margin_ratio: float = 0.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 rotation_range: Tuple[float, float] = (-10, 10),
                 translation_params: Tuple[float, float] = (0, 3.6), 
                 jitter_prob: float = 0.8,
                 interpolation_method: str = 'random',  # 'random', 'linear', 'nearest' 
        ) -> None:
        """
        初始化函数
        
        Args:
            input_size: 输出图像尺寸 (width, height)
            mean_face_path: 均值人脸npy文件路径
            point_set_types: 点集类型列表 ['four', 'five', 'nineteen', 'twenty_three', 'base', 'val']
            point_set_weights: 点集选择的权重，None表示均匀分布
            max_tries: 最大尝试次数
            margin_ratio: 边界检查的边距比例
            scale_range: 缩放抖动范围
            rotation_range: 旋转抖动范围（度数）
            translation_params: 平移抖动范围（mu sigma）
            jitter_prob: 抖动概率
        """
        super().__init__()
        
        self.input_size = input_size
        self.mean_face_path = mean_face_path
        self.point_set_types = point_set_types
        self.max_tries = max_tries
        self.margin_ratio = margin_ratio
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_params = translation_params
        self.jitter_prob = jitter_prob
        self.interpolation_method = interpolation_method

        self.mean_face = self._load_mean_face(mean_face_path)
        
        if point_set_weights is None:
            self.point_set_weights = [1.0 / len(point_set_types)] * len(point_set_types)
        else:
            self.point_set_weights = point_set_weights

        self._define_point_sets()
        self.index_minmax = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 127, 128, 129, 132, 133, 134, 139]
        
        self.margin_w = self.input_size[0] * self.margin_ratio
        self.margin_h = self.input_size[1] * self.margin_ratio
        
        self.shift_mu = self.translation_params[0]
        self.shift_sigma = self.translation_params[1]
        
        self.target_points, self.mean_center, self.mean_size = self._preprocess_target_points()
    
    def _load_mean_face(self, path: str) -> np.ndarray:
        mean_face = np.load(path)
        assert mean_face.shape == (235, 2), f"均值人脸应为(235, 2)，实际为{mean_face.shape}"
        mean_face = mean_face.astype(np.float32)

        return mean_face
    
    def _define_point_sets(self):
        self.eye_left_idx = 201
        self.eye_right_idx = 202
        self.nose_tip_idx = 139
        self.mouth_left_idx = 141
        self.mouth_right_idx = 159
        
        contour_idx = list(range(0, 37, 2))
        
        self.point_sets = {
            'four': [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'five': [self.eye_left_idx, self.eye_right_idx, self.nose_tip_idx, self.mouth_left_idx, self.mouth_right_idx],
            'nineteen': contour_idx,
            'twenty_three': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'base': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'val': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx]
        }
    
    def _get_square_box(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        x_coords = np.array(points)[:, 0]
        y_coords = np.array(points)[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        rect_width = max_x - min_x
        rect_height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        square_size = max(rect_width, rect_height)
        center_point = np.array([center_x, center_y]).astype(np.float32)
        
        return center_point, square_size
    
    def _preprocess_target_points(self) -> Tuple[np.ndarray, np.ndarray, float]:
        target_points = self.mean_face.copy()
        target_points[:, 0] = target_points[:, 0] * self.input_size[0]
        target_points[:, 1] = target_points[:, 1] * self.input_size[1]

        mean_center, mean_size = self._get_square_box(target_points[self.index_minmax])
        
        return target_points, mean_center, mean_size
    
    def similarity_transform_from_points(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, Any]:
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        
        points1_centered = points1 - c1
        points2_centered = points2 - c2
        
        s1 = np.std(points1_centered) + 1e-8 
        s2 = np.std(points2_centered) + 1e-8
        
        scale = s2 / s1
        
        points1_norm = points1_centered / s1
        points2_norm = points2_centered / s2

        H = np.dot(points1_norm.T, points2_norm)
        U, S, Vt = np.linalg.svd(H)
        
        R = (U @ Vt).T
        
        # 确保是纯旋转 没有镜像（det(R) = 1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = (U @ Vt).T
        
        M = scale * R
        B = c2.reshape(2, 1) - np.dot(M, c1.reshape(2, 1))
        
        rotation_rad = np.arctan2(R[1, 0], R[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        affine_matrix = np.hstack((M, B))
        
        M_inv = np.linalg.inv(M)
        # B_inv = -np.dot(M_inv, B)
        B_inv = -B
        
        return {
            'M': M,
            'B': B,
            'R': R,
            'M_inv': M_inv, 
            'B_inv': B_inv,
            'affine_matrix': affine_matrix,
            'scale': scale,                   # 缩放因子
            'rotation_deg': rotation_deg,     # 旋转角度
            'c1': c1,
            'c2': c2
        }
    
    def compose_similarity_transform(self, transform_params, box_center, box_size) -> Dict[str, Any]:
        scale_box = self.mean_size / box_size
            
        M = np.array([[scale_box, 0], [0, scale_box]])
        B = self.mean_center - scale_box * box_center
        B = B.reshape(-1, 1)
                
        # T_compose(x) = M2 * (M1 * x + B1) + B2 = (M2 * M1) * x + (M2 * B1 + B2)
        M_compose = M @ transform_params['M']
        B_compose = M @ transform_params['B'] + B
        
        M_inv_compose = np.linalg.inv(M_compose)
        B_inv_compose = -B_compose
        
        affine_matrix_compose = np.hstack((M_compose, B_compose.reshape(-1, 1)))
        
        compose_params = transform_params.copy()
        compose_params.update({
            'M': M_compose,
            'B': B_compose,
            'M_inv': M_inv_compose,
            'B_inv': B_inv_compose,
            'affine_matrix': affine_matrix_compose,
            'scale': scale_box * transform_params['scale'],
        })
        
        return compose_params

    def rotate_transform_params(self, transform_params: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        M = [[a, -b], [b, a]]
        """

        scale_orig = transform_params['scale']
        rotation_deg_orig = transform_params['rotation_deg']
        
        c1 = transform_params['c1']  # 源点集中心
        c2 = transform_params['c2']  # 目标点集中心
        
        # 抖动旋转角度
        sigma = (self.rotation_range[1] - self.rotation_range[0]) / 6.0  # 99.7%数据在范围内
        rotation_deg_jittered = rotation_deg_orig + np.random.normal(0, sigma)
        rotation_rad_jittered = np.radians(rotation_deg_jittered)
        cos_theta = np.cos(rotation_rad_jittered)
        sin_theta = np.sin(rotation_rad_jittered)
        R_jittered = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]], dtype=np.float32)
        
        M_jittered = scale_orig * R_jittered
        B_translation = c2.reshape(2, 1) - np.dot(M_jittered, c1.reshape(2, 1))
            
        B_jittered = np.array([[B_translation[0, 0]], [B_translation[1, 0]]], dtype=np.float32)
        
        affine_matrix_jittered = np.hstack((M_jittered, B_jittered))
        
        jittered_params = transform_params.copy()
        jittered_params.update({
            'M': M_jittered,
            'B': B_jittered,
            'R': R_jittered,
            'affine_matrix': affine_matrix_jittered,
            'rotation_deg': rotation_deg_jittered
        })
        
        return jittered_params
    
    def scale_shift_transform_params(self, transform_params, box_center, box_size) -> Dict[str, Any]:
        scale_box = self.mean_size / box_size
        
        # 抖动缩放
        sigma = (self.scale_range[1] - self.scale_range[0]) / 6.0  # 99.7%数据在范围内
        scale_jittered = scale_box * np.clip(np.random.normal(1.0, sigma), self.scale_range[0], self.scale_range[1])
             
        M = np.array([[scale_jittered, 0], [0, scale_jittered]])
        B = self.mean_center - scale_jittered * box_center
        B = B.reshape(-1, 1)
                
        # T_compose(x) = M2 * (M1 * x + B1) + B2 = (M2 * M1) * x + (M2 * B1 + B2)
        M_compose = M @ transform_params['M']
        B_compose = M @ transform_params['B'] + B
        
        # 抖动平移   
        shift = np.random.normal(self.shift_mu, self.shift_sigma, 2)     
        tx_jittered = B_compose[0, 0] + shift[0]
        ty_jittered = B_compose[1, 0] + shift[1]
        
        M_jittered = M_compose
        B_jittered = np.array([[tx_jittered], [ty_jittered]], dtype=np.float32)
        
        M_jittered_inv = np.linalg.inv(M_jittered)
        B_jittered_inv = -B_jittered
            
        affine_matrix_jittered = np.hstack((M_jittered, B_jittered.reshape(-1, 1)))
        
        jittered_params = transform_params.copy()
        jittered_params.update({
            'M': M_jittered,
            'B': B_jittered,
            'M_inv': M_jittered_inv,
            'B_inv': B_jittered_inv,
            'affine_matrix': affine_matrix_jittered,
            'scale': scale_jittered * transform_params['scale'],
        })
        
        return jittered_params
    
    def _count_out_of_bounds_points(self, points: np.ndarray) -> int:
        w, h = self.input_size        
        margin_w, margin_h = self.margin_w, self.margin_h
        
        x_in_bounds = (points[:, 0] >= margin_w) & (points[:, 0] <= w - margin_w)
        y_in_bounds = (points[:, 1] >= margin_h) & (points[:, 1] <= h - margin_h)
        
        in_bounds_mask = x_in_bounds & y_in_bounds
        
        out_of_bounds_count = len(points) - np.sum(in_bounds_mask)
        
        return int(out_of_bounds_count)
    
    def apply_affine_transform(self, img: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        if self.interpolation_method == 'random':
            interpolation = cv2.INTER_LINEAR if random.random() > 0.5 else cv2.INTER_NEAREST
        elif self.interpolation_method == 'linear':
            interpolation = cv2.INTER_LINEAR
        else:  # 'nearest'
            interpolation = cv2.INTER_NEAREST
        
        transformed_img = cv2.warpAffine(img, affine_matrix, self.input_size, flags=interpolation)
        
        return transformed_img
    
    def _transform_points(self, points: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed_points = np.dot(points_homogeneous, affine_matrix.T)
        
        return transformed_points
    
    def _generate_random_nineteen(self) -> List[int]:
        """动态生成随机19点集：每对相邻点随机选一个，18固定"""
        choices = np.random.randint(0, 2, size=18)

        first_half = 2 * np.arange(9) + choices[:9]
        second_half = 20 + 2 * np.arange(9) - choices[9:]
        
        indices = np.concatenate([first_half, [18], second_half])
        
        return indices.tolist()

    def select_point_set(self) -> Tuple[str, List[int]]:
        point_set_type = random.choices(
            self.point_set_types, 
            weights=self.point_set_weights, 
            k=1
        )[0]
        
        if point_set_type in ['nineteen', 'twenty_three']:
            indices = self._generate_random_nineteen()
            
            if point_set_type == 'twenty_three':
                indices = indices + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx]
        else:
            indices = self.point_sets[point_set_type].copy()
        
        return point_set_type, indices
    
    def transform(self, results: Dict) -> Optional[dict]:
        img = results['img']
        
        if results.get('keypoints', None) is not None:
            keypoints = results['keypoints'][0]
        
            src_points_all = np.array(keypoints, dtype=np.float32)

            best_out_of_bounds_count = 235
            
            for try_num in range(self.max_tries):
                point_set_type, point_indices = self.select_point_set()
                
                src_selected = src_points_all[point_indices].copy()
                target_selected = self.target_points[point_indices].copy()
                
                try:
                    transform_params = self.similarity_transform_from_points(src_selected, target_selected)
                except np.linalg.LinAlgError:
                    continue  # 矩阵奇异，跳过此次尝试
                
                if random.random() < self.jitter_prob:
                    rotate_transform_params = self.rotate_transform_params(transform_params)
                    
                    tmp_transformed_points = self._transform_points(src_points_all[self.index_minmax], rotate_transform_params['affine_matrix']) 
                    box_center, box_size = self._get_square_box(tmp_transformed_points)
                    
                    transform_params = self.scale_shift_transform_params(rotate_transform_params, box_center, box_size)
                else:
                    tmp_transformed_points = self._transform_points(src_points_all[self.index_minmax], transform_params['affine_matrix']) 
                    box_center, box_size = self._get_square_box(tmp_transformed_points)
                    
                    transform_params = self.compose_similarity_transform(transform_params, box_center, box_size)
                
                transformed_points = self._transform_points(src_points_all, transform_params['affine_matrix'])
                
                out_of_bounds_count = self._count_out_of_bounds_points(transformed_points)
                
                if out_of_bounds_count == 0 or point_set_type == "val":
                    out_transform_params = transform_params
                    out_transformed_points = transformed_points
                    break
                
                if out_of_bounds_count < best_out_of_bounds_count:
                    best_out_of_bounds_count = out_of_bounds_count
                    out_transform_params = transform_params
                    out_transformed_points = transformed_points
                    
            results['img'] = self.apply_affine_transform(img, out_transform_params['affine_matrix'])
            results['transformed_keypoints'] = np.array([out_transformed_points])
            
            results['M_inv'] = np.array([out_transform_params['M_inv']])
            results['B_inv'] = np.array([out_transform_params['B_inv']])
        
        results['input_size'] = self.input_size
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'point_set_types={self.point_set_types}, '
        repr_str += f'max_tries={self.max_tries}, '
        repr_str += f'jitter_prob={self.jitter_prob})'
        return repr_str


@TRANSFORMS.register_module()
class TopdownAlignV4(BaseTransform):    
    def __init__(self,
                 input_size: Tuple[int, int],
                 mean_face_path: str,
                 point_set_types: List[str] = ['four', 'nineteen', 'twenty_three', 'base', 'val'],
                 point_set_weights: Optional[List[float]] = None,
                 max_tries: int = 20,
                 margin_ratio: float = 0.0,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 rotation_range: Tuple[float, float] = (-10, 10),
                 translation_params: Tuple[float, float] = (0, 3.6), 
                 jitter_prob: float = 0.8,
                 interpolation_method: str = 'random',  # 'random', 'linear', 'nearest' 
        ) -> None:
        """
        初始化函数
        
        Args:
            input_size: 输出图像尺寸 (width, height)
            mean_face_path: 均值人脸npy文件路径
            point_set_types: 点集类型列表 ['four', 'nineteen', 'twenty_three', 'base', 'val']
            point_set_weights: 点集选择的权重，None表示均匀分布
            max_tries: 最大尝试次数
            margin_ratio: 边界检查的边距比例
            scale_range: 缩放抖动范围
            rotation_range: 旋转抖动范围（度数）
            translation_params: 平移抖动范围（mu sigma）
            jitter_prob: 抖动概率
        """
        super().__init__()
        
        self.input_size = input_size
        self.mean_face_path = mean_face_path
        self.point_set_types = point_set_types
        self.max_tries = max_tries
        self.margin_ratio = margin_ratio
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_params = translation_params
        self.jitter_prob = jitter_prob
        self.interpolation_method = interpolation_method

        self.mean_face = self._load_mean_face(mean_face_path)
        
        if point_set_weights is None:
            self.point_set_weights = [1.0 / len(point_set_types)] * len(point_set_types)
        else:
            self.point_set_weights = point_set_weights

        self._define_point_sets()
        self.index_minmax = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 127, 128, 129, 132, 133, 134, 139]
        
        self.margin_w = self.input_size[0] * self.margin_ratio
        self.margin_h = self.input_size[1] * self.margin_ratio
        
        self.shift_mu = self.translation_params[0]
        self.shift_sigma = self.translation_params[1]
        
        self.target_points, self.mean_center, self.mean_size, self.mean_center_4, self.mean_size_4 = self._preprocess_target_points()
    
    def _load_mean_face(self, path: str) -> np.ndarray:
        mean_face = np.load(path)
        assert mean_face.shape == (235, 2), f"均值人脸应为(235, 2)，实际为{mean_face.shape}"
        mean_face = mean_face.astype(np.float32)

        return mean_face
    
    def _define_point_sets(self):
        self.eye_left_idx = 201
        self.eye_right_idx = 202
        self.nose_tip_idx = 139
        self.mouth_left_idx = 141
        self.mouth_right_idx = 159
        
        contour_idx = list(range(0, 37, 2))
        
        self.point_sets = {
            'four': [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'five': [self.eye_left_idx, self.eye_right_idx, self.nose_tip_idx, self.mouth_left_idx, self.mouth_right_idx],
            'nineteen': contour_idx,
            'twenty_three': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'base': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx],
            'val': contour_idx + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx]
        }
    
    def _get_square_box(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        x_coords = np.array(points)[:, 0]
        y_coords = np.array(points)[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        rect_width = max_x - min_x
        rect_height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        square_size = max(rect_width, rect_height)
        center_point = np.array([center_x, center_y]).astype(np.float32)
        
        return center_point, square_size
    
    def _preprocess_target_points(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        target_points = self.mean_face.copy()
        target_points[:, 0] = target_points[:, 0] * self.input_size[0]
        target_points[:, 1] = target_points[:, 1] * self.input_size[1]

        mean_center, mean_size = self._get_square_box(target_points[self.index_minmax])
        
        target_points_4 = self.mean_face[self.point_sets['four']]
        target_points_4[:, 0] = target_points_4[:, 0] * self.input_size[0]
        target_points_4[:, 1] = target_points_4[:, 1] * self.input_size[1]

        mean_center_4, mean_size_4 = self._get_square_box(target_points[self.point_sets['four']])
        
        return target_points, mean_center, mean_size, mean_center_4, mean_size_4
    
    def similarity_transform_from_points(self, points1: np.ndarray, points2: np.ndarray) -> Dict[str, Any]:
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        
        points1_centered = points1 - c1
        points2_centered = points2 - c2
        
        s1 = np.std(points1_centered) + 1e-8 
        s2 = np.std(points2_centered) + 1e-8
        
        scale = s2 / s1
        
        points1_norm = points1_centered / s1
        points2_norm = points2_centered / s2

        H = np.dot(points1_norm.T, points2_norm)
        U, S, Vt = np.linalg.svd(H)
        
        R = (U @ Vt).T
        
        # 确保是纯旋转 没有镜像（det(R) = 1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = (U @ Vt).T
        
        M = scale * R
        B = c2.reshape(2, 1) - np.dot(M, c1.reshape(2, 1))
        
        rotation_rad = np.arctan2(R[1, 0], R[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        affine_matrix = np.hstack((M, B))
        
        M_inv = np.linalg.inv(M)
        # B_inv = -np.dot(M_inv, B)
        B_inv = -B
        
        return {
            'M': M,
            'B': B,
            'R': R,
            'M_inv': M_inv, 
            'B_inv': B_inv,
            'affine_matrix': affine_matrix,
            'scale': scale,                   # 缩放因子
            'rotation_deg': rotation_deg,     # 旋转角度
            'c1': c1,
            'c2': c2
        }
    
    def compose_similarity_transform(self, transform_params, box_center, box_size) -> Dict[str, Any]:
        scale_box = self.mean_size / box_size
            
        M = np.array([[scale_box, 0], [0, scale_box]])
        B = self.mean_center - scale_box * box_center
        B = B.reshape(-1, 1)
                
        # T_compose(x) = M2 * (M1 * x + B1) + B2 = (M2 * M1) * x + (M2 * B1 + B2)
        M_compose = M @ transform_params['M']
        B_compose = M @ transform_params['B'] + B
        
        M_inv_compose = np.linalg.inv(M_compose)
        B_inv_compose = -B_compose
        
        affine_matrix_compose = np.hstack((M_compose, B_compose.reshape(-1, 1)))
        
        compose_params = transform_params.copy()
        compose_params.update({
            'M': M_compose,
            'B': B_compose,
            'M_inv': M_inv_compose,
            'B_inv': B_inv_compose,
            'affine_matrix': affine_matrix_compose,
            'scale': scale_box * transform_params['scale'],
        })
        
        return compose_params

    def rotate_transform_params(self, transform_params: Dict[str, Any]) -> Dict[str, Any]:
        """ 
        M = [[a, -b], [b, a]]
        """

        scale_orig = transform_params['scale']
        rotation_deg_orig = transform_params['rotation_deg']
        
        c1 = transform_params['c1']  # 源点集中心
        c2 = transform_params['c2']  # 目标点集中心
        
        # 抖动旋转角度
        sigma = (self.rotation_range[1] - self.rotation_range[0]) / 6.0  # 99.7%数据在范围内
        rotation_deg_jittered = rotation_deg_orig + np.random.normal(0, sigma)
        rotation_rad_jittered = np.radians(rotation_deg_jittered)
        cos_theta = np.cos(rotation_rad_jittered)
        sin_theta = np.sin(rotation_rad_jittered)
        R_jittered = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]], dtype=np.float32)
        
        M_jittered = scale_orig * R_jittered
        B_translation = c2.reshape(2, 1) - np.dot(M_jittered, c1.reshape(2, 1))
            
        B_jittered = np.array([[B_translation[0, 0]], [B_translation[1, 0]]], dtype=np.float32)
        
        affine_matrix_jittered = np.hstack((M_jittered, B_jittered))
        
        jittered_params = transform_params.copy()
        jittered_params.update({
            'M': M_jittered,
            'B': B_jittered,
            'R': R_jittered,
            'affine_matrix': affine_matrix_jittered,
            'rotation_deg': rotation_deg_jittered
        })
        
        return jittered_params
    
    def scale_shift_transform_params(self, transform_params, box_center, mean_center, scale) -> Dict[str, Any]:
        scale_box = scale
        
        # 抖动缩放
        sigma = (self.scale_range[1] - self.scale_range[0]) / 6.0  # 99.7%数据在范围内
        scale_jittered = scale_box * np.clip(np.random.normal(1.0, sigma), self.scale_range[0], self.scale_range[1])
             
        M = np.array([[scale_jittered, 0], [0, scale_jittered]])
        B = mean_center - scale_jittered * box_center
        B = B.reshape(-1, 1)
                
        # T_compose(x) = M2 * (M1 * x + B1) + B2 = (M2 * M1) * x + (M2 * B1 + B2)
        M_compose = M @ transform_params['M']
        B_compose = M @ transform_params['B'] + B
        
        # 抖动平移   
        shift = np.random.normal(self.shift_mu, self.shift_sigma, 2)     
        tx_jittered = B_compose[0, 0] + shift[0]
        ty_jittered = B_compose[1, 0] + shift[1]
        
        M_jittered = M_compose
        B_jittered = np.array([[tx_jittered], [ty_jittered]], dtype=np.float32)
        
        M_jittered_inv = np.linalg.inv(M_jittered)
        B_jittered_inv = -B_jittered
            
        affine_matrix_jittered = np.hstack((M_jittered, B_jittered.reshape(-1, 1)))
        
        jittered_params = transform_params.copy()
        jittered_params.update({
            'M': M_jittered,
            'B': B_jittered,
            'M_inv': M_jittered_inv,
            'B_inv': B_jittered_inv,
            'affine_matrix': affine_matrix_jittered,
            'scale': scale_jittered * transform_params['scale'],
        })
        
        return jittered_params
    
    def _count_out_of_bounds_points(self, points: np.ndarray) -> int:
        w, h = self.input_size        
        margin_w, margin_h = self.margin_w, self.margin_h
        
        x_in_bounds = (points[:, 0] >= margin_w) & (points[:, 0] <= w - margin_w)
        y_in_bounds = (points[:, 1] >= margin_h) & (points[:, 1] <= h - margin_h)
        
        in_bounds_mask = x_in_bounds & y_in_bounds
        
        out_of_bounds_count = len(points) - np.sum(in_bounds_mask)
        
        return int(out_of_bounds_count)
    
    def apply_affine_transform(self, img: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        if self.interpolation_method == 'random':
            interpolation = cv2.INTER_LINEAR if random.random() > 0.5 else cv2.INTER_NEAREST
        elif self.interpolation_method == 'linear':
            interpolation = cv2.INTER_LINEAR
        else:  # 'nearest'
            interpolation = cv2.INTER_NEAREST
        
        transformed_img = cv2.warpAffine(img, affine_matrix, self.input_size, flags=interpolation)
        
        return transformed_img
    
    def _transform_points(self, points: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed_points = np.dot(points_homogeneous, affine_matrix.T)
        
        return transformed_points
    
    def _generate_random_nineteen(self) -> List[int]:
        """动态生成随机19点集：每对相邻点随机选一个，18固定"""
        choices = np.random.randint(0, 2, size=18)

        first_half = 2 * np.arange(9) + choices[:9]
        second_half = 20 + 2 * np.arange(9) - choices[9:]
        
        indices = np.concatenate([first_half, [18], second_half])
        
        return indices.tolist()

    def select_point_set(self) -> Tuple[str, List[int]]:
        point_set_type = random.choices(
            self.point_set_types, 
            weights=self.point_set_weights, 
            k=1
        )[0]
        
        if point_set_type in ['nineteen', 'twenty_three']:
            indices = self._generate_random_nineteen()
            
            if point_set_type == 'twenty_three':
                indices = indices + [self.eye_left_idx, self.eye_right_idx, self.mouth_left_idx, self.mouth_right_idx]
        else:
            indices = self.point_sets[point_set_type].copy()
        
        return point_set_type, indices
    
    def transform(self, results: Dict) -> Optional[dict]:
        img = results['img']
        
        if results.get('keypoints', None) is not None:
            keypoints = results['keypoints'][0]
        
            src_points_all = np.array(keypoints, dtype=np.float32)

            best_out_of_bounds_count = 235
            
            for try_num in range(self.max_tries):
                point_set_type, point_indices = self.select_point_set()
                
                src_selected = src_points_all[point_indices].copy()
                target_selected = self.target_points[point_indices].copy()
                
                try:
                    transform_params = self.similarity_transform_from_points(src_selected, target_selected)
                except np.linalg.LinAlgError:
                    continue  # 矩阵奇异，跳过此次尝试
                
                if random.random() < self.jitter_prob:
                    rotate_transform_params = self.rotate_transform_params(transform_params)
                    
                    if point_set_type == "four":                        
                        tmp_transformed_points = self._transform_points(src_points_all[point_indices], rotate_transform_params['affine_matrix'])
                        box_center, box_size = self._get_square_box(tmp_transformed_points)
                        
                        # move center a bit to nose inv direction
                        tmp_transformed_nose = self._transform_points(src_points_all[[self.nose_tip_idx]], rotate_transform_params['affine_matrix'])
                        box_center[0] += (box_center[0] - tmp_transformed_nose[:, 0][0])
                        
                        scale_box = max(self.mean_size_4 / box_size, 0.9)
                        
                        transform_params = self.scale_shift_transform_params(rotate_transform_params, box_center, self.mean_center_4, scale_box)
                    else:
                        tmp_transformed_points = self._transform_points(src_points_all[self.index_minmax], rotate_transform_params['affine_matrix']) 
                        box_center, box_size = self._get_square_box(tmp_transformed_points)
                        
                        scale_box = self.mean_size / box_size
                    
                        transform_params = self.scale_shift_transform_params(rotate_transform_params, box_center, self.mean_center, scale_box)
                else:
                    tmp_transformed_points = self._transform_points(src_points_all[self.index_minmax], transform_params['affine_matrix']) 
                    box_center, box_size = self._get_square_box(tmp_transformed_points)
                    
                    transform_params = self.compose_similarity_transform(transform_params, box_center, box_size)
                
                transformed_points = self._transform_points(src_points_all, transform_params['affine_matrix'])
                
                out_of_bounds_count = self._count_out_of_bounds_points(transformed_points)
                
                if out_of_bounds_count == 0 or point_set_type == "val":
                    out_transform_params = transform_params
                    out_transformed_points = transformed_points
                    break
                
                if out_of_bounds_count < best_out_of_bounds_count:
                    best_out_of_bounds_count = out_of_bounds_count
                    out_transform_params = transform_params
                    out_transformed_points = transformed_points
                    
            results['img'] = self.apply_affine_transform(img, out_transform_params['affine_matrix'])
            results['transformed_keypoints'] = np.array([out_transformed_points])
            
            results['M_inv'] = np.array([out_transform_params['M_inv']])
            results['B_inv'] = np.array([out_transform_params['B_inv']])
        
        results['input_size'] = self.input_size
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'point_set_types={self.point_set_types}, '
        repr_str += f'max_tries={self.max_tries}, '
        repr_str += f'jitter_prob={self.jitter_prob})'
        return repr_str