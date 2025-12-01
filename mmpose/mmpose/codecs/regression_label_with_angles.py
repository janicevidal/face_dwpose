# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class RegressionLabelWithAngles(BaseKeypointCodec):
    r"""Generate keypoint coordinates.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_labels (np.ndarray): The normalized regression labels in
            shape (N, K, D) where D is 2 for 2d coordinates
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]

    """
    auxiliary_encode_keys = {'euler_angles'}

    def __init__(self, input_size: Tuple[int, int], euler_bin: Tuple[int, int], euler_bin_interval: int = 1, max_angles: int =90) -> None:
        super().__init__()

        self.input_size = input_size
        self.euler_bin = euler_bin
        self.euler_bin_interval = euler_bin_interval
        self.max_angles = max_angles
        
        assert self.euler_bin[0] <= self.max_angles and self.euler_bin[1] <= self.max_angles, "euler_bin must be smaller than max_angles"

    def encode(self,
               keypoints: np.ndarray,
               euler_angles: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        w, h = self.input_size
        valid = ((keypoints >= 0) &
                 (keypoints <= [w - 1, h - 1])).all(axis=-1) & (
                     keypoints_visible > 0.5)

        keypoint_labels = (keypoints / np.array([w, h])).astype(np.float32)
        keypoint_weights = np.where(valid, 1., 0.).astype(np.float32)
        
        # assert has_euler_angles is True, "Euler angles must be provided for RegressionLabelWithAngles"
        
        yaw = euler_angles[0][0]
        pitch = euler_angles[0][1]
        
        if yaw > self.max_angles:
            yaw = self.max_angles - 0.5
        if yaw < -1 * self.max_angles:
            yaw = -1 * self.max_angles + 0.5
        if pitch > self.max_angles:
            pitch = self.max_angles - 0.5
        if pitch < -1 * self.max_angles:
            pitch = -1 * self.max_angles + 0.5
        
        yaw_bins = np.array(range(-self.euler_bin[0], self.euler_bin[0], self.euler_bin_interval))
        pitch_bins = np.array(range(-self.euler_bin[1], self.euler_bin[1], self.euler_bin_interval))
        
        binned_pose = np.concatenate((np.digitize([yaw], yaw_bins) - 1, np.digitize([pitch], pitch_bins) - 1), axis=0)
        binned_pose = binned_pose[np.newaxis, :]
        
        encoded = dict(
            keypoint_labels=keypoint_labels, keypoint_weights=keypoint_weights, euler_labels=binned_pose)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        if encoded.shape[-1] == 2:
            N, K, _ = encoded.shape
            normalized_coords = encoded.copy()
            scores = np.ones((N, K), dtype=np.float32)
        elif encoded.shape[-1] == 4:
            # split coords and sigma if outputs contain output_sigma
            normalized_coords = encoded[..., :2].copy()
            output_sigma = encoded[..., 2:4].copy()

            scores = (1 - output_sigma).mean(axis=-1)
        else:
            raise ValueError(
                'Keypoint dimension should be 2 or 4 (with sigma), '
                f'but got {encoded.shape[-1]}')

        w, h = self.input_size
        keypoints = normalized_coords * np.array([w, h])

        return keypoints, scores
