#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from scipy.ndimage import zoom

from comp2comp.inference_class_base import InferenceClass
from comp2comp.liver_spleen_pancreas.visualization_utils import (
    generate_liver_spleen_pancreas_report,
    generate_slice_images,
)

from comp2comp.visualization.detectron_visualizer import Visualizer

_COLORS = (
    np.array(
        [
            1.000, 0.000, 0.000, # red
            0.000, 1.000, 0.000, # green
            0.000, 0.000, 1.000, # blue
            1.000, 1.000, 0.000, # yellow
            1.000, 0.000, 1.000, # magenta
            0.000, 1.000, 1.000, # cyan
            0.500, 0.000, 0.000, # dark red
            0.000, 0.500, 0.000, # dark green
            0.000, 0.000, 0.500, # dark blue
            0.500, 0.500, 0.000, # dark yellow
            0.500, 0.000, 0.500, # dark magenta
            0.000, 0.500, 0.500, # dark cyan
            0.750, 0.000, 0.000, # lighter dark red
            0.000, 0.750, 0.000, # lighter dark green
            0.000, 0.000, 0.750, # lighter dark blue
            0.750, 0.750, 0.000, # lighter dark yellow
            0.750, 0.000, 0.750, # lighter dark magenta
            0.000, 0.750, 0.750  # lighter dark cyan
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

class LungsVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

        self.class_nums = [13, 14, 15, 16, 17]
        self.lung_names = [
            "Left Upper Lobe", 
            "Left Lower Lobe", 
            "Right Upper Lobe",
            "Right Middle Lobe",
            "Right Lower Lobe"
        ]

    def __call__(self, inference_pipeline):

        self.output_dir = inference_pipeline.output_dir
        self.output_dir_images = os.path.join(self.output_dir, "images/")
        inference_pipeline.output_dir_images = self.output_dir_images

        if not os.path.exists(self.output_dir_images):
            os.makedirs(self.output_dir_images)

        seg = inference_pipeline.segmentation
        img = inference_pipeline.medical_volume

        seg_np = seg.get_fdata()
        img_in = img.get_fdata()

        # compute the average HU value for each lung
        lung_values = []
        for i, level in enumerate(self.lung_names):
            lung_values.append(img_in[seg_np == self.class_nums[i]].mean())

        pixel_spacing = img.header.get_zooms()
        zoom_factor = pixel_spacing[2] / pixel_spacing[1]

        # make all values not in the class_nums 0
        seg_np[~np.isin(seg_np, self.class_nums)] = 0
        slice_index = np.argmax(np.sum(seg_np != 0, axis=(0, 2)))
        slice_index = slice_index + 50

        # extract the corresponding image and segmentation slices
        seg_slice = seg_np[:, slice_index, :]
        
        img_in = np.clip(img_in, -1000, 1400)
        img_in = normalize_img(img_in) * 255.0
        img_in = img_in[:, slice_index, :]

        img_in = zoom(img_in, (1, zoom_factor), order=3)
        seg_slice = zoom(seg_slice, (1, zoom_factor), order=1).round()

        img_in = transpose_and_flip(img_in)
        seg_slice = transpose_and_flip(seg_slice)

        img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
        img_rgb = np.tile(img_in, (1, 1, 3))

        vis = Visualizer(img_rgb)

        # draw seg masks
        for i, level in enumerate(self.lung_names):
            color = _COLORS[i]
            edge_color = None
            alpha_val = 0.2
            vis.draw_binary_mask(
                (seg_slice == self.class_nums[i]).astype(int),
                color=color,
                edge_color=edge_color,
                alpha=alpha_val,
                area_threshold=0,
            )

        # add the lung values to the report
        for i, level in enumerate(self.lung_names):
            vis.draw_text(
                "{}: {}".format(level, round(lung_values[i])),
                (360, 10 + 14 * i),
                color=_COLORS[i],
                font_size=9,
                horizontal_alignment="left",
            )

        vis_obj = vis.get_output()
        img = vis_obj.save(os.path.join(self.output_dir_images, "lung_report.png"))

        return {}

"""
class LungsMetricsSaver(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        results = inference_pipeline.organ_metrics
        organs = list(results.keys())

        return {}
"""

def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())

def transpose_and_flip(img):
    img = np.transpose(img)
    img = np.flip(img, axis=0)
    img = np.flip(img, axis=1)
    return img
