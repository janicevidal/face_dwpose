# Copyright (c) OpenMMLab. All rights reserved.
from .aflw_dataset import AFLWDataset
from .coco_wholebody_face_dataset import CocoWholeBodyFaceDataset
from .cofw_dataset import COFWDataset
from .face_300w_dataset import Face300WDataset
from .lapa_dataset import LapaDataset
from .wflw_dataset import WFLWDataset
from .inshot_dataset import InshotDataset

__all__ = [
    'Face300WDataset', 'WFLWDataset', 'AFLWDataset', 'COFWDataset',
    'CocoWholeBodyFaceDataset', 'LapaDataset', 'InshotDataset'
]
