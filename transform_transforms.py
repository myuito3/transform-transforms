import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np


def load_from_json(filepath: Path) -> dict:
    assert filepath.suffix == ".json"
    with open(filepath, mode="r", encoding="utf-8") as file:
        return json.load(file)


@dataclass
class FrameParams:
    file_path: str
    transform_matrix: np.ndarray
    colmap_im_id: Optional[int]


@dataclass
class TransformsJsonParams:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    frames: List[FrameParams]
    k1: Optional[float] = None
    k2: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    camera_model: Optional[str] = None
    applied_transform: Optional[np.ndarray] = None
    ply_file_path: Optional[str] = None


class TransformTransforms:
    transforms_json_params: TransformsJsonParams

    def __init__(self, input_dir: Path) -> None:
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)

        files = os.listdir(input_dir)
        assert "images" in files
        assert "transforms.json" in files

        data = load_from_json(input_dir / "transforms.json")
        self.transforms_json_params = self.create_transforms_json_params(data)

        print(self.transforms_json_params)

    def create_transforms_json_params(self, data: dict) -> ...:
        frames = [
            FrameParams(
                file_path=frame_data["file_path"],
                transform_matrix=np.array(frame_data["transform_matrix"]),
                colmap_im_id=(
                    frame_data["colmap_im_id"] if "colmap_im_id" in frame_data else None
                ),
            )
            for frame_data in data["frames"]
        ]

        return TransformsJsonParams(
            w=data["w"],
            h=data["h"],
            fl_x=data["fl_x"],
            fl_y=data["fl_y"],
            cx=data["cx"],
            cy=data["cy"],
            frames=frames,
            k1=data["k1"] if "k1" in data else None,
            k2=data["k2"] if "k2" in data else None,
            p1=data["p1"] if "p1" in data else None,
            p2=data["p2"] if "p2" in data else None,
            camera_model=data["camera_model"] if "camera_model" in data else None,
            applied_transform=(
                np.array(data["applied_transform"])
                if "applied_transform" in data
                else None
            ),
            ply_file_path=data["ply_file_path"] if "ply_file_path" in data else None,
        )

    def associate_frames(self) -> ...:
        pass

    def resize(self) -> ...:
        pass

    def crop(self) -> ...:
        pass

    def dump(self) -> ...:
        pass


# pytest
if __name__ == "__main__":
    tt = TransformTransforms("data/fox")
