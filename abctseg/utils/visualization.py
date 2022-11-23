import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from abctseg.utils.visualizer import Visualizer

_image_files = [
    "spine_coronal.png",
    "spine_sagittal.png",
    "T12_seg.png",
    "L3_seg.png",
    "L1_seg.png",
    "L4_seg.png",
    "L2_seg.png",
    "L5_seg.png",
]

_COLORS = (
    np.array(
        [
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.000,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.340,
            0.200,
            1.000,
            0.340,
            0.200,
            1.000,
            0.340,
            0.200,
            1.000,
            0.340,
            0.200,
            1.000,
            0.340,
            0.200,
            1.000,
            0.340,
            0.200,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

_COLOR_MAP = {"L5": 0, "L4": 1, "L3": 2, "L2": 3, "L1": 4, "T12": 5}

"""

_TISSUES = [
    "Muscle",
    "VAT",
    "SAT",
    "IMAT"
]

_SPINE_LEVELS = [
    "L5",
    "L4",
    "L3",
    "L2",
    "L1",
    "T12"
]
"""

_TEXT_SPACING = 25.0
_TEXT_START_VERTICAL_OFFSET = 10.0
_TEXT_OFFSET_FROM_RIGHT = 68


def save_binary_segmentation_overlay(
    img_in: Union[str, Path],
    mask: Union[str, Path],
    base_path: Union[str, Path],
    file_name: str,
    centroids=None,
    figure_text_key=None,
    spine_hus=None,
    spine=True,
    model_type=None,
):
    """
    Save binary segmentation overlay.
    Parameters
    ----------
    img_in: str, Path
        Path to the input image.
    mask: str, Path
        Path to the mask.
    base_path: str, Path
        Path to the output directory.
    file_name: str
        Name of the output file.
    centroids: list
        List of centroids.
    figure_text_key: str
        Key to the figure text.
    """

    if model_type.model_name == "ts_spine":
        _SPINE_LEVELS = list(model_type.categories.keys())
    elif model_type.model_name == "stanford_v0.0.1":
        _TISSUES = list(model_type.categories.keys())

    # Window image to retain most information
    img_in = np.clip(img_in, -300, 1800)

    # Normalize the image
    img_in = normalize_img(img_in) * 255.0

    # Create the folder to save the images
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)

    text_start_vertical_offset = _TEXT_START_VERTICAL_OFFSET

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))
    vis = Visualizer(img_rgb)
    for num_bin_masks in range(1, mask.shape[2] + 1):
        if centroids:
            alpha_val = 0.2
        else:
            alpha_val = 1
        vis.draw_binary_mask(
            mask[:, :, num_bin_masks - 1].astype(int),
            color=_COLORS[num_bin_masks - 1],
            alpha=alpha_val,
        )
        # Debug imat mask
        if centroids:
            if num_bin_masks > 6:
                continue
            vis.draw_text(
                text=f"{_SPINE_LEVELS[num_bin_masks - 1]} ROI HU: {spine_hus[num_bin_masks - 1]:.2f}",
                position=(
                    mask.shape[1] - _TEXT_OFFSET_FROM_RIGHT,
                    int(_TEXT_SPACING / 2.0) * (num_bin_masks - 1) + _TEXT_START_VERTICAL_OFFSET,
                ),
                color=_COLORS[num_bin_masks - 1],
                font_size=7,
            )

            vis.draw_line(
                x_data=(0, mask.shape[1] - 1),
                y_data=(
                    centroids[num_bin_masks - 1],
                    centroids[num_bin_masks - 1],
                ),
                color=_COLORS[num_bin_masks - 1],
                linestyle="dashed",
                linewidth=0.5,
            )
        else:
            if spine:
                vis.draw_box(
                    box_coord=(1, 1, mask.shape[0] - 1, mask.shape[1] - 1),
                    alpha=1,
                    edge_color=_COLORS[_COLOR_MAP[file_name.split("_")[0]]],
                )

            if figure_text_key:
                if spine:
                    hu_val = figure_text_key[file_name.split("_")[0]][(num_bin_masks - 1) * 2]
                    area_val = figure_text_key[file_name.split("_")[0]][
                        ((num_bin_masks - 1) * 2) + 1
                    ]
                else:
                    hu_val = figure_text_key[".".join(file_name.split(".")[:-1])][
                        (num_bin_masks - 1) * 2
                    ]
                    area_val = figure_text_key[".".join(file_name.split(".")[:-1])][
                        ((num_bin_masks - 1) * 2) + 1
                    ]

                vis.draw_text(
                    text=f"{_TISSUES[num_bin_masks - 1]} HU: " + hu_val,
                    position=(
                        mask.shape[1] - _TEXT_OFFSET_FROM_RIGHT,
                        _TEXT_SPACING * (num_bin_masks - 1) + text_start_vertical_offset,
                    ),
                    color=_COLORS[num_bin_masks - 1],
                    font_size=7,
                )
                vis.draw_text(
                    text=f"{_TISSUES[num_bin_masks - 1]} AREA: " + area_val,
                    position=(
                        mask.shape[1] - _TEXT_OFFSET_FROM_RIGHT,
                        _TEXT_SPACING * (num_bin_masks - 1)
                        + text_start_vertical_offset
                        + (_TEXT_SPACING / 2),
                    ),
                    color=_COLORS[num_bin_masks - 1],
                    font_size=7,
                )
    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(images_base_path, file_name))


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image.
    Parameters
    ----------
    img: np.ndarray
        Image to normalize.
    Returns
    -------
    np.ndarray
        Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())


def generate_panel(image_dir: Union[str, Path]):
    """
    Generate panel.
    Parameters
    ----------
    image_dir: str, Path
        Path to the input image directory.
    """
    image_files = [os.path.join(image_dir, path) for path in _image_files]
    new_im = Image.new("RGB", (2080, 1040))
    index = 0
    for i in range(0, 2080, 520):
        for j in range(0, 1040, 520):
            im = Image.open(image_files[index])
            im.thumbnail((512, 512))
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(os.path.join(image_dir, "panel.png"))
