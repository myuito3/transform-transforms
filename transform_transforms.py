from __future__ import annotations

import json
import os
from dataclasses import fields, dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def load_from_json(filepath: Path) -> dict:
    assert filepath.suffix == ".json"
    with open(filepath, mode="r", encoding="utf-8") as file:
        return json.load(file)


def dataclass_to_dict(obj: object):
    if not is_dataclass(obj):
        raise TypeError("Input must be a dataclass instance")

    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        if value is None:
            continue

        if isinstance(value, np.ndarray):
            result[field.name] = value.tolist()
        elif is_dataclass(value):
            result[field.name] = dataclass_to_dict(value)
        elif isinstance(value, list):
            result[field.name] = [
                dataclass_to_dict(v) if is_dataclass(v) else v for v in value
            ]
        else:
            result[field.name] = value

    return result


@dataclass
class FrameParams:
    file_path: str
    transform_matrix: np.ndarray
    colmap_im_id: Optional[int] = None


@dataclass
class TransformsParams:
    w: Optional[int] = None
    h: Optional[int] = None
    fl_x: Optional[float] = None
    fl_y: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    k1: Optional[float] = None
    k2: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    camera_model: Optional[str] = None
    frames: Optional[list[FrameParams]] = None
    applied_transform: Optional[np.ndarray] = None
    ply_file_path: Optional[str] = None

    @classmethod
    def create(cls, data: dict) -> TransformsParams:
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

        return cls(
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

    def update(
        self,
        w=None,
        h=None,
        fl_x=None,
        fl_y=None,
        cx=None,
        cy=None,
        frames=None,
        k1=None,
        k2=None,
        p1=None,
        p2=None,
        camera_model=None,
        applied_transform=None,
        ply_file_path=None,
    ):
        self.w = w if w is not None else self.w
        self.h = h if h is not None else self.h
        self.fl_x = fl_x if fl_x is not None else self.fl_x
        self.fl_y = fl_y if fl_y is not None else self.fl_y
        self.cx = cx if cx is not None else self.cx
        self.cy = cy if cy is not None else self.cy
        self.frames = frames if frames is not None else self.frames
        self.k1 = k1 if k1 is not None else self.k1
        self.k2 = k2 if k2 is not None else self.k2
        self.p1 = p1 if p1 is not None else self.p1
        self.p2 = p2 if p2 is not None else self.p2
        self.camera_model = (
            camera_model if camera_model is not None else self.camera_model
        )
        self.applied_transform = (
            applied_transform
            if applied_transform is not None
            else self.applied_transform
        )
        self.ply_file_path = (
            ply_file_path if ply_file_path is not None else self.ply_file_path
        )


class TransformTransforms:
    xs_params: TransformsParams
    image_filenames: list[str]

    def __init__(self, input_dir: Path) -> None:
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)
        assert input_dir.exists(), f"{input_dir} does not exist"

        self.input_dir = input_dir
        self._initialize()

    def _initialize(self) -> None:
        assert "transforms.json" in os.listdir(self.input_dir)
        data = load_from_json(self.input_dir / "transforms.json")
        self.xs_params = TransformsParams.create(data)

        assert "images" in os.listdir(self.input_dir)
        self.image_filenames = os.listdir(self.input_dir / "images")

    def associate_frames(self) -> None:
        associated_images = [
            filename
            for filename in self.image_filenames
            if any(
                filename == frame.file_path.split("/")[-1]
                for frame in self.xs_params.frames
            )
        ]
        associated_frames = [
            frame
            for frame in self.xs_params.frames
            if frame.file_path.split("/")[-1] in associated_images
        ]

        self.xs_params.update(frames=associated_frames)
        self.image_filenames = associated_images

    def resize(self, width: int, height: int) -> None:
        unit_fl_x = self.xs_params.fl_x / self.xs_params.w
        unit_fl_y = self.xs_params.fl_y / self.xs_params.h
        new_fl_x = unit_fl_x * width
        new_fl_y = unit_fl_y * height

        self.xs_params.update(
            w=width,
            h=height,
            fl_x=new_fl_x,
            fl_y=new_fl_y,
            cx=width // 2,
            cy=height // 2,
        )

    def dump(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir / "transforms.json", mode="w", encoding="utf-8") as file:
            json.dump(dataclass_to_dict(self.xs_params), file, indent=4)


# pytest
if __name__ == "__main__":
    tt = TransformTransforms("data/rest copy")
    tt.associate_frames()
    tt.dump(output_dir="data/rest")
