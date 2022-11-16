from detectron2.utils.visualizer import Visualizer, VisImage
from pathlib import Path
import numpy as np
import os
from PIL import Image
from os import listdir
from os.path import isfile, join
from glob import glob
from typing import Union, List

_image_files = ["spine_coronal.png", "spine_sagittal.png", "T12_seg.png", "L3_seg.png", "L1_seg.png", "L4_seg.png", "L2_seg.png", "L5_seg.png"]

_COLORS = np.array(
    [
        1.000, 0.000, 0.000,
        0.000, 0.000, 1.000, 
        0.000, 1.000, 0.000, 
        1.000, 1.000, 0.000,
        0.000, 1.000, 1.000,
        1.000, 0.000, 1.000

    ]
).astype(np.float32).reshape(-1, 3)

_COLOR_MAP = {
    "L5": 0,
    "L4": 1,
    "L3": 2,
    "L2": 3,
    "L1": 4,
    "T12": 5
}

_TISSUES = [
    "Muscle",
    "Bone",
    "VAT",
    "SAT"
]


def save_binary_segmentation_overlay(img_in: Union[str, Path], mask: Union[str, Path], base_path: Union[str, Path], file_name: str, centroids = None, figure_text_key = None):
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
    # Window image to retain most information
    img_in = np.clip(img_in, -300, 1800)

    # Normalize the image
    img_in = normalize_img(img_in) * 255.0

    # Create the folder to save the images
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)
    
    text_spacing = 25.0
    text_start_offset = 10.0

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))
    vis = Visualizer(img_rgb)
    for num_bin_masks in range(1, mask.shape[2]):
        if centroids:
            alpha_val = 0.2
        else:
            alpha_val = 1
            if num_bin_masks == 2:
                continue
        vis.draw_binary_mask(mask[:, :, num_bin_masks].astype(int), color = _COLORS[num_bin_masks - 1], alpha=alpha_val)
        if centroids:
            #print(centroids)
            vis.draw_line(x_data = (0, mask.shape[1] - 1), y_data = (centroids[num_bin_masks - 1], centroids[num_bin_masks - 1]), color = _COLORS[num_bin_masks - 1], linestyle="dashed", linewidth = 0.5)
        else:
            #print(figure_text_key)
            vis.draw_box(box_coord = (1, 1, mask.shape[0] - 1, mask.shape[1] - 1), alpha = 1, edge_color = _COLORS[_COLOR_MAP[file_name.split('_')[0]]])
            if figure_text_key:
                if num_bin_masks == 3:
                    text_start_offset -= text_spacing
                vis.draw_text(text=f"{_TISSUES[num_bin_masks - 1]} HU: " + figure_text_key[file_name.split('_')[0]][(num_bin_masks - 1) * 2], position=(mask.shape[1] - 68, text_spacing * (num_bin_masks - 1) + text_start_offset), color=_COLORS[num_bin_masks - 1], font_size = 7)
                vis.draw_text(text=f"{_TISSUES[num_bin_masks - 1]} AREA: " + figure_text_key[file_name.split('_')[0]][((num_bin_masks - 1) * 2) + 1], position=(mask.shape[1] - 68, text_spacing * (num_bin_masks - 1) + text_start_offset + (text_spacing / 2)), color=_COLORS[num_bin_masks - 1], font_size = 7)
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
    new_im = Image.new('RGB', (2080, 1040))
    index = 0
    for i in range(0,2080,520):
        for j in range(0,1040,520):
            im = Image.open(image_files[index])
            im.thumbnail((512,512))
            new_im.paste(im, (i,j))
            index += 1
    new_im.save(os.path.join(image_dir, "panel.png"))