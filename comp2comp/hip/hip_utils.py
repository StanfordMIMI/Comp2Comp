import logging
import math
import os
from glob import glob
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pydicom.filereader import dcmread
from scipy.ndimage import zoom

from comp2comp.models.models import Models
from comp2comp.hip.hip_visualization import method_visualizer, normalize_img
from comp2comp.visualization.detectron_visualizer import Visualizer


def compute_rois(medical_volume, segmentation, model, output_dir):
    left_femur_mask = segmentation.get_fdata() == model.categories["femur_left"]
    left_femur_mask = left_femur_mask.astype(np.uint8)
    right_femur_mask = segmentation.get_fdata() == model.categories["femur_right"]
    right_femur_mask = right_femur_mask.astype(np.uint8)
    left_femural_head_mask = get_femural_head_roi(left_femur_mask, medical_volume, output_dir)


def get_femural_head_roi(femur_mask, medical_volume, output_dir):
    # find the largest index that is not zero
    top = np.where(femur_mask.sum(axis=(0, 1)) != 0)[0].max()
    top_mask = femur_mask[:, :, top]
    center_of_mass = np.array(np.where(top_mask == 1)).mean(axis=1)
    coronal_slice = femur_mask[:, round(center_of_mass[1]), :]
    coronal_image = medical_volume.get_fdata()[:, round(center_of_mass[1]), :]
    sagittal_slice = femur_mask[round(center_of_mass[0]), :, :]
    sagittal_image = medical_volume.get_fdata()[round(center_of_mass[0]), :, :]
    zooms = medical_volume.header.get_zooms()
    zoom_factor = zooms[2] / zooms[1]
    coronal_slice = zoom(coronal_slice, (1, zoom_factor), order=1).round()
    sagittal_slice = zoom(sagittal_slice, (1, zoom_factor), order=1).round()
    coronal_image = zoom(coronal_image, (1, zoom_factor), order=1).round()
    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=1).round()
    dist_map = cv2.distanceTransform(sagittal_slice, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_sagittal, _, center_sagittal = cv2.minMaxLoc(dist_map)

    dist_map = cv2.distanceTransform(coronal_slice, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_coronal, _, center_coronal = cv2.minMaxLoc(dist_map)

    coronal_image = np.clip(coronal_image, -300, 1800)
    coronal_image = normalize_img(coronal_image) * 255.0

    sagittal_image = np.clip(sagittal_image, -300, 1800)
    sagittal_image = normalize_img(sagittal_image) * 255.0

    method_visualizer(
        sagittal_image,
        coronal_image,
        coronal_slice,
        sagittal_slice,
        center_sagittal,
        radius_sagittal,
        center_coronal,
        radius_coronal,
        output_dir,
    )
