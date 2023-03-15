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

_SPINE_TEXT_OFFSET_FROM_TOP = 10.0
_SPINE_TEXT_OFFSET_FROM_RIGHT = 63.0
_SPINE_TEXT_VERTICAL_SPACING = 14.0

_MUSCLE_FAT_TEXT_HORIZONTAL_SPACING = 40.0
_MUSCLE_FAT_TEXT_VERTICAL_SPACING = 14.0
_MUSCLE_FAT_TEXT_OFFSET_FROM_TOP = 22.0
_MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT = 181.0


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

    _2D_COLORS = (
        np.array([255, 136, 133, 140, 197, 135, 246, 190, 129, 154, 135, 224])
        .astype(np.float32)
        .reshape(-1, 3)
    )

    _2D_COLORS = _2D_COLORS / 255.0

    if model_type and (
        (model_type.model_name == "ts_spine") or (model_type.model_name == "stanford_spine_v0.0.1")
    ):
        _SPINE_LEVELS = list(model_type.categories.keys())
        # reverse the list
        _SPINE_LEVELS = _SPINE_LEVELS[::-1]
    elif model_type and (
        (model_type.model_name == "stanford_v0.0.1") or (model_type.model_name == "abCT_v0.0.1")
    ):
        _TISSUES = list(model_type.categories.keys())
        _TISSUES[_TISSUES.index("muscle")] = "Muscle"
        _TISSUES[_TISSUES.index("imat")] = "IMAT"
        _TISSUES[_TISSUES.index("vat")] = "VAT"
        _TISSUES[_TISSUES.index("sat")] = "SAT"
        _SPINE_LEVELS = ["T12", "L1", "L2", "L3", "L4", "L5"]
        if model_type.model_name == "abCT_v0.0.1":
            # put the imat color before vat and sat colors
            _2D_COLORS = np.insert(_2D_COLORS, 1, _2D_COLORS[3], axis=0)

    # Window image to retain most information
    img_in = np.clip(img_in, -300, 1800)

    # Normalize the image
    img_in = normalize_img(img_in) * 255.0

    # Create the folder to save the images
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)

    text_start_vertical_offset = _MUSCLE_FAT_TEXT_OFFSET_FROM_TOP

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))
    vis = Visualizer(img_rgb)
    if not centroids:
        vis.draw_text(
            text="Density (HU)",
            position=(
                mask.shape[1] - _MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT - 63,
                text_start_vertical_offset,
            ),
            color=[1, 1, 1],
            font_size=9,
            horizontal_alignment="left",
        )
        vis.draw_text(
            text="Area (CM²)",
            position=(
                mask.shape[1] - _MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT - 63,
                text_start_vertical_offset + _MUSCLE_FAT_TEXT_VERTICAL_SPACING,
            ),
            color=[1, 1, 1],
            font_size=9,
            horizontal_alignment="left",
        )
    for num_bin_masks in range(1, mask.shape[2] + 1):
        if num_bin_masks > 6:
            color = _COLORS[num_bin_masks - 1]
            edge_color = color
        else:
            edge_color = None
        if centroids:
            alpha_val = 0.2
            color = _COLORS[num_bin_masks - 1]
        else:
            alpha_val = 0.9
            color = _2D_COLORS[num_bin_masks - 1]
            edge_color = color
        vis.draw_binary_mask(
            mask[:, :, num_bin_masks - 1].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )
        # Debug imat mask
        if centroids:
            if num_bin_masks > 6:
                continue
            vis.draw_text(
                text=f"{_SPINE_LEVELS[num_bin_masks - 1]}: {round(float(spine_hus[num_bin_masks - 1]))}",
                position=(
                    mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                    _SPINE_TEXT_VERTICAL_SPACING * (num_bin_masks - 1)
                    + _SPINE_TEXT_OFFSET_FROM_TOP,
                ),
                color=_COLORS[5 - (num_bin_masks - 1)],
                font_size=9,
                horizontal_alignment="left",
            )

            vis.draw_line(
                x_data=(0, mask.shape[1] - 1),
                y_data=(
                    int(centroids[num_bin_masks - 1] * (pixel_spacing[2] / pixel_spacing[1])),
                    int(centroids[num_bin_masks - 1] * (pixel_spacing[2] / pixel_spacing[1])),
                ),
                color=_COLORS[num_bin_masks - 1],
                linestyle="dashed",
                linewidth=0.25,
            )
        else:
            if spine:
                vis.draw_box(
                    box_coord=(1, 1, mask.shape[0] - 1, mask.shape[1] - 1),
                    alpha=1,
                    edge_color=_COLORS[_COLOR_MAP[file_name.split("_")[0]]],
                )
                # draw the level T12 - L5 in the upper left corner
                if file_name.split("_")[0] == "T12":
                    position = (40, 15)
                else:
                    position = (30, 15)
                vis.draw_text(
                    text=f"{file_name.split('_')[0]}",
                    position=position,
                    color=_COLORS[_COLOR_MAP[file_name.split("_")[0]]],
                    font_size=24,
                )

            if figure_text_key:
                if spine:
                    hu_val = round(
                        float(figure_text_key[file_name.split("_")[0]][(num_bin_masks - 1) * 2])
                    )
                    area_val = round(
                        float(
                            figure_text_key[file_name.split("_")[0]][((num_bin_masks - 1) * 2) + 1]
                        )
                    )
                else:
                    hu_val = round(
                        float(
                            figure_text_key[".".join(file_name.split(".")[:-1])][
                                (num_bin_masks - 1) * 2
                            ]
                        )
                    )
                    area_val = round(
                        float(
                            figure_text_key[".".join(file_name.split(".")[:-1])][
                                ((num_bin_masks - 1) * 2) + 1
                            ]
                        )
                    )
                vis.draw_text(
                    text=f"{_TISSUES[num_bin_masks - 1]}",
                    position=(
                        mask.shape[1]
                        - _MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT
                        + _MUSCLE_FAT_TEXT_HORIZONTAL_SPACING * (num_bin_masks),
                        text_start_vertical_offset - _MUSCLE_FAT_TEXT_VERTICAL_SPACING,
                    ),
                    color=_2D_COLORS[num_bin_masks - 1],
                    font_size=9,
                    horizontal_alignment="center",
                )

                vis.draw_text(
                    text=hu_val,
                    position=(
                        mask.shape[1]
                        - _MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT
                        + _MUSCLE_FAT_TEXT_HORIZONTAL_SPACING * (num_bin_masks),
                        text_start_vertical_offset,
                    ),
                    color=_2D_COLORS[num_bin_masks - 1],
                    font_size=9,
                    horizontal_alignment="center",
                )
                vis.draw_text(
                    text=area_val,
                    position=(
                        mask.shape[1]
                        - _MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT
                        + _MUSCLE_FAT_TEXT_HORIZONTAL_SPACING * (num_bin_masks),
                        text_start_vertical_offset + _MUSCLE_FAT_TEXT_VERTICAL_SPACING,
                    ),
                    color=_2D_COLORS[num_bin_masks - 1],
                    font_size=9,
                    horizontal_alignment="center",
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
