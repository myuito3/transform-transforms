import os
from pathlib import Path
from typing import List

import imageio
import numpy as np


def load_from_json() -> ...:
    pass


class FrameParams:
    file_path: str
    transform_matrix: np.ndarray
    colmap_im_id: str


class TransformsJsonParams:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    camera_model: str
    frames: List[FrameParams]
    applied_transform: np.ndarray
    ply_file_path: str


class TransformTransforms:
    def __init__(self, input_dir: Path) -> None:
        files = os.listdir(input_dir)
        assert "images" in files
        assert "transforms.json" in files

    def associate_frames(self) -> ...:
        pass

    def resize(self) -> ...:
        pass

    def crop(self) -> ...:
        pass

    def dump(self) -> ...:
        pass
