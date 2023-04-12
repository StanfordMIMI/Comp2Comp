import os
from pathlib import Path
from typing import Union

import numpy as np
from scipy.ndimage import zoom

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
    coronal_image = np.clip(coronal_image, -300, 1800)
    coronal_image = normalize_img(coronal_image) * 255.0

    sagittal_image = np.clip(sagittal_image, -300, 1800)
    sagittal_image = normalize_img(sagittal_image) * 255.0

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

def roi_visualizer(
    roi,
    medical_volume,
    centroid,
    output_dir,
):
    zooms = medical_volume.header.get_zooms()
    zoom_factor = zooms[2] / zooms[1]

    sagittal_image = medical_volume.get_fdata()[centroid[0], :, :]
    sagittal_roi = roi[centroid[0], :, :]

    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=1).round()
    sagittal_roi = zoom(sagittal_roi, (1, zoom_factor), order=3).round()
    sagittal_image = np.flip(sagittal_image.T)
    sagittal_roi = np.flip(sagittal_roi.T)

    axial_image = medical_volume.get_fdata()[:, :, round(centroid[2])]
    axial_roi = roi[:, :, round(centroid[2])]

    axial_image = np.flip(axial_image.T)
    axial_roi = np.flip(axial_roi.T)

    sagittal_image = np.clip(sagittal_image, -300, 1800)
    sagittal_image = normalize_img(sagittal_image) * 255.0
    sagittal_image = sagittal_image.reshape((sagittal_image.shape[0], sagittal_image.shape[1], 1))
    img_rgb = np.tile(sagittal_image, (1, 1, 3))
    vis = Visualizer(img_rgb)
    vis.draw_binary_mask(sagittal_roi)
    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(output_dir, "hip_roi_sagittal_slice.png"))

    axial_image = np.clip(axial_image, -300, 1800)
    axial_image = normalize_img(axial_image) * 255.0
    axial_image = axial_image.reshape((axial_image.shape[0], axial_image.shape[1], 1))
    img_rgb = np.tile(axial_image, (1, 1, 3))
    vis = Visualizer(img_rgb)
    vis.draw_binary_mask(axial_roi)
    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(output_dir, "hip_roi_axial_slice.png"))


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())
