"""
@author: louisblankemeier
"""

import os
from pathlib import Path
from typing import Union

import numpy as np

from comp2comp.visualization.detectron_visualizer import Visualizer


def spine_binary_segmentation_overlay(
    img_in: Union[str, Path],
    mask: Union[str, Path],
    base_path: Union[str, Path],
    file_name: str,
    figure_text_key=None,
    spine_hus=None,
    spine=True,
    model_type=None,
    pixel_spacing=None,
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

    label_map = {
        "L5": 0, 
        "L4": 1, 
        "L3": 2, 
        "L2": 3, 
        "L1": 4, 
        "T12": 5,
        "T11": 6,
        "T10": 7,
        "T9": 8,
        "T8": 9,
        "T7": 10,
        "T6": 11,
        "T5": 12,
        "T4": 13,
        "T3": 14,
        "T2": 15,
        "T1": 16,
    }

    _ROI_COLOR = np.array([1.000, 0.340, 0.200])

    _SPINE_TEXT_OFFSET_FROM_TOP = 10.0
    _SPINE_TEXT_OFFSET_FROM_RIGHT = 63.0
    _SPINE_TEXT_VERTICAL_SPACING = 14.0

    img_in = np.clip(img_in, -300, 1800)
    img_in = normalize_img(img_in) * 255.0
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois
    for i, _ in enumerate(levels):
        color = _ROI_COLOR
        edge_color = color
        vis.draw_binary_mask(
            mask[:, :, num_levels + i].astype(int),
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
                _SPINE_TEXT_VERTICAL_SPACING * i + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=_COLORS[label_map[level]],
            font_size=9,
            horizontal_alignment="left",
        )

        """
        vis.draw_line(
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.25,
        )
        """

    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())
