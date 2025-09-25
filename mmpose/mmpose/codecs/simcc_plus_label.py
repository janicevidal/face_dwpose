# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple, Union

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .simcc_label import SimCCLabel
from .regression_label import RegressionLabel


@KEYPOINT_CODECS.register_module()
class SimCCPlusLabel(BaseKeypointCodec):
    r"""Generate keypoint representation via "SimCC" approach.
    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_x_labels (np.ndarray): The generated SimCC label for x-axis.
            The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wx=w*simcc_split_ratio`
        - keypoint_y_labels (np.ndarray): The generated SimCC label for y-axis.
            The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wy=h*simcc_split_ratio`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]
        smoothing_type (str): The SimCC label smoothing strategy. Options are
            ``'gaussian'`` and ``'standard'``. Defaults to ``'gaussian'``
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC
            label. Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        label_smooth_weight (float): Label Smoothing weight. Defaults to 0.0
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """

    def __init__(self,
                 input_size: Tuple[int, int],
                 smoothing_type: str = 'gaussian',
                 sigma: Union[float, int, Tuple[float]] = 6.0,
                 simcc_split_ratio: float = 2.0,
                 label_smooth_weight: float = 0.0,
                 normalize: bool = True,
                 use_dark: bool = False) -> None:
        super().__init__()

        self.simcc_codec = SimCCLabel(input_size, smoothing_type, sigma, simcc_split_ratio, 
                                         label_smooth_weight, normalize, use_dark)
        self.keypoint_codec = RegressionLabel(input_size)

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints to regression labels and heatmaps.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        encoded_cc = self.simcc_codec.encode(keypoints, keypoints_visible)
        encoded_kp = self.keypoint_codec.encode(keypoints, keypoints_visible)

        x_labels = encoded_cc['keypoint_x_labels']
        y_labels = encoded_cc['keypoint_y_labels']
        keypoint_weights = encoded_kp['keypoint_weights']
        keypoint_labels = encoded_kp['keypoint_labels']

        encoded = dict(
            keypoint_x_labels=x_labels,
            keypoint_y_labels=y_labels,
            keypoint_labels=keypoint_labels,
            keypoint_weights=keypoint_weights)

        return encoded

    # def decode(self, simcc_x: np.ndarray,
    #            simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """Decode keypoint coordinates from SimCC representations. The decoded
    #     coordinates are in the input image space.

    #     Args:
    #         encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis
    #             and y-axis
    #         simcc_x (np.ndarray): SimCC label for x-axis
    #         simcc_y (np.ndarray): SimCC label for y-axis

    #     Returns:
    #         tuple:
    #         - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
    #         - socres (np.ndarray): The keypoint scores in shape (N, K).
    #             It usually represents the confidence of the keypoint prediction
    #     """

    #     keypoints, scores = self.simcc_codec.decode(simcc_x, simcc_y)

    #     return keypoints, scores
    
    
    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        keypoints, scores = self.keypoint_codec.decode(encoded)

        return keypoints, scores
