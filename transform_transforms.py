from __future__ import annotations

import json
import os
from dataclasses import fields, dataclass, is_dataclass
from pathlib import Path
from typing import Optional

import imageio.v3 as imageio
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
class TransformsJsonParams:
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
    def create(cls, data: dict) -> TransformsJsonParams:
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


class TransformTransforms:
    transforms_json_params: TransformsJsonParams
    image_filenames: list[str]
    _resize: Optional[tuple[int, int]] = None
    _crop: Optional[tuple[int, int]] = None

    def __init__(self, input_dir: Path) -> None:
        if not isinstance(input_dir, Path):
            input_dir = Path(input_dir)
        assert input_dir.exists(), f"{input_dir} does not exist"
        self.input_dir = input_dir
        self._initialize()

    def _initialize(self) -> None:
        assert "transforms.json" in os.listdir(self.input_dir)
        data = load_from_json(self.input_dir / "transforms.json")
        self.transforms_json_params = TransformsJsonParams.create(data)

        assert "images" in os.listdir(self.input_dir)
        self.image_filenames = os.listdir(self.input_dir / "images")

    def associate_frames(self) -> None:
        associated_images = [
            filename
            for filename in self.image_filenames
            if any(
                filename == frame.file_path.split("/")[-1]
                for frame in self.transforms_json_params.frames
            )
        ]
        associated_frames = [
            frame
            for frame in self.transforms_json_params.frames
            if frame.file_path.split("/")[-1] in associated_images
        ]
        self.transforms_json_params.frames = associated_frames
        self.image_filenames = associated_images

    def resize(self, size: tuple[int, int]) -> None:
        self._resize = size

    def crop(self, size: tuple[int, int]) -> None:
        self._crop = size

    def dump(self, output_dir: str) -> None:
        output_dir = Path(output_dir)
        image_output_dir = output_dir / "images"
        os.makedirs(image_output_dir, exist_ok=True)

        for filename in self.image_filenames:
            image = imageio.imread(self.input_dir / "images" / filename)
            if self._resize is not None:
                pass
            if self._crop is not None:
                pass
            imageio.imwrite(image_output_dir / filename, image)

        with open(output_dir / "transforms.json", mode="w", encoding="utf-8") as file:
            json.dump(dataclass_to_dict(self.transforms_json_params), file, indent=4)


# pytest
if __name__ == "__main__":
    tt = TransformTransforms("data/equ")
    tt.associate_frames()
    tt.dump(output_dir="data/equ_new")
