import os
from pathlib import Path
from typing import Union

import numpy as np

from comp2comp.visualization.detectron_visualizer import Visualizer

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
            1.000,
            0.500,
            0.000,
            0.000,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

_SPINE_TEXT_OFFSET_FROM_TOP = 10.0
_SPINE_TEXT_OFFSET_FROM_RIGHT = 63.0
_SPINE_TEXT_VERTICAL_SPACING = 14.0

_MUSCLE_FAT_TEXT_HORIZONTAL_SPACING = 40.0
_MUSCLE_FAT_TEXT_VERTICAL_SPACING = 14.0
_MUSCLE_FAT_TEXT_OFFSET_FROM_TOP = 22.0
_MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT = 181.0

_ROI_COLOR = np.array([1.000, 0.340, 0.200])

_COLOR_MAP = {"L5": 0, "L4": 1, "L3": 2, "L2": 3, "L1": 4, "T12": 5}

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
    pixel_spacing=None,
    levels=None
):
    """Save binary segmentation overlay.
    Args:
        img_in (Union[str, Path]): Path to the input image.
        mask (Union[str, Path]): Path to the mask.
        base_path (Union[str, Path]): Path to the output directory.
        file_name (str): Output file name.
        centroids (list, optional): List of centroids. Defaults to None.
        figure_text_key (dict, optional): Figure text key. Defaults to None.
        spine_hus (list, optional): List of HU values. Defaults to None.
        spine (bool, optional): Spine flag. Defaults to True.
        model_type (Models): Model type. Defaults to None.
    """
    label_map = {"T12": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}

    _SPINE_LEVELS = levels
    img_in = np.clip(img_in, -300, 1800)
    img_in = normalize_img(img_in) * 255.0
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)

    text_start_vertical_offset = _MUSCLE_FAT_TEXT_OFFSET_FROM_TOP
    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    num_levels = len(levels)

    print("SPINE LEVELS: ", _SPINE_LEVELS)
    print("SPINE HUS: ", spine_hus)
    print("CENTROIDS: ", centroids)
    print("MASK SHAPE: ", mask.shape)
    print("NUM TOTAL: ", num_total)
    print("IMG in SHAPE: ", img_in.shape)

    # draw seg masks
    for i, level in enumerate(levels):
            color = _COLORS[_COLOR_MAP[level]]
            edge_color = None
            alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, num_levels - 1 - i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois 
    for i, level in enumerate(levels):
            color = _ROI_COLOR
            edge_color = color
        vis.draw_binary_mask(
            mask[:, :, num_levels - 1 - i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )
    
    # draw text and lines
    for i, level in enumerate(levels):
        vis.draw_text(
            text=f"{level}: {round(float(spine_hus[level]))}",
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                _SPINE_TEXT_VERTICAL_SPACING * num_bin_masks
                + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=_COLORS[num_levels - 1 - i],
            font_size=9,
            horizontal_alignment="left",
        )

        vis.draw_line(
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(centroids[i] * (pixel_spacing[2] / pixel_spacing[1])),
                int(centroids[i] * (pixel_spacing[2] / pixel_spacing[1])),
            ),
            color=_COLORS[num_levels],
            linestyle="dashed",
            linewidth=0.25,
        )

    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(images_base_path, file_name))


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())
