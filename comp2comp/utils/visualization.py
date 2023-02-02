import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from comp2comp.utils.visualizer import Visualizer

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
    pixel_spacing=None
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
        np.array(
            [
                255,
                136, 
                133,
                140,
                197,
                135,
                246,
                190,
                129,
                154,
                135,
                224
            ]
        )
    .astype(np.float32)
    .reshape(-1, 3)
    )

    _2D_COLORS = _2D_COLORS / 255.0

    if model_type and (model_type.model_name == "ts_spine"):
        _SPINE_LEVELS = list(model_type.categories.keys())
        # reverse the list
        _SPINE_LEVELS = _SPINE_LEVELS[::-1]
    elif model_type and ((model_type.model_name == "stanford_v0.0.1") or (model_type.model_name == "abCT_v0.0.1")):
        _TISSUES = list(model_type.categories.keys())
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

    text_start_vertical_offset = _TEXT_START_VERTICAL_OFFSET

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))
    vis = Visualizer(img_rgb)
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
            area_threshold = 0
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
                color=_COLORS[5 - (num_bin_masks - 1)],
                font_size=7,
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
                if file_name.split('_')[0] == "T12":
                    position = (40, 15)
                else:
                    position = (30, 15)
                vis.draw_text(
                    text = f"{file_name.split('_')[0]}",
                    position = position,
                    color = _COLORS[_COLOR_MAP[file_name.split("_")[0]]],
                    font_size = 24
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
                    color=_2D_COLORS[num_bin_masks - 1],
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
                    color=_2D_COLORS[num_bin_masks - 1],
                    font_size=7,
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


def generate_panel(image_dir: Union[str, Path]):
    """Generate panel.

    Args:
        image_dir (Union[str, Path]): Path to the image directory.
    """
    image_files = [os.path.join(image_dir, path) for path in _image_files]
    im_cor = Image.open(image_files[0])
    im_sag = Image.open(image_files[1])
    im_cor_width = int(im_cor.width / im_cor.height * 512)
    width = (8 + im_cor_width + 8) + ((512 + 8) * 3)
    height = 1048
    new_im = Image.new("RGB", (width, height))

    index = 2 
    for i in range(8 + im_cor_width + 8, width, 520):
        for j in range(8, height, 520):
            im = Image.open(image_files[index])
            im.thumbnail((512, 512))
            new_im.paste(im, (i, j))
            index += 1
            im.close()
    
    im_cor.thumbnail((im_cor_width, 512))
    new_im.paste(im_cor, (8, 8))
    im_sag.thumbnail((im_cor_width, 512))
    new_im.paste(im_sag, (8, 528))
    new_im.save(os.path.join(image_dir, "panel.png"))
    im_cor.close()
    im_sag.close()
    new_im.close()



