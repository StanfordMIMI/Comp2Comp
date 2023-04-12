import os
from pathlib import Path
from typing import Union

import numpy as np

from comp2comp.visualization.detectron_visualizer import Visualizer

def method_visualizer(
    sagittal_image,
    coronal_image,
    coronal_slice,
    sagittal_slice,
    center_sagittal,
    radius_sagittal,
    center_coronal,
    radius_coronal,
    output_dir,
):
    sagittal_image = sagittal_image.reshape((sagittal_image.shape[0], sagittal_image.shape[1], 1))
    img_rgb = np.tile(sagittal_image, (1, 1, 3))
    vis = Visualizer(img_rgb)
    vis.draw_circle(circle_coord=center_sagittal, color=[0, 1, 0], radius=radius_sagittal)
    vis.draw_binary_mask(sagittal_slice)

    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(output_dir, "hip_sagittal_slice.png"))

    coronal_image = coronal_image.reshape((coronal_image.shape[0], coronal_image.shape[1], 1))
    img_rgb = np.tile(coronal_image, (1, 1, 3))
    vis = Visualizer(img_rgb)
    vis.draw_circle(circle_coord=center_coronal, color=[0, 1, 0], radius=radius_coronal)
    vis.draw_binary_mask(coronal_slice)

    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(output_dir, "hip_coronal_slice.png"))


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())
