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
from comp2comp.hip.hip_visualization import method_visualizer


def compute_rois(medical_volume, segmentation, model, output_dir):
    left_femur_mask = segmentation.get_fdata() == model.categories["femur_left"]
    left_femur_mask = left_femur_mask.astype(np.uint8)
    right_femur_mask = segmentation.get_fdata() == model.categories["femur_right"]
    right_femur_mask = right_femur_mask.astype(np.uint8)
    left_roi, left_centroid, left_hu = get_femural_head_roi(left_femur_mask, medical_volume, output_dir, "left")
    right_roi, right_centroid, right_hu = get_femural_head_roi(right_femur_mask, medical_volume, output_dir, "right")
    return {"left": {"roi": left_roi, "centroid": left_centroid, "hu": left_hu}, "right": {"roi": right_roi, "centroid": right_centroid, "hu": right_hu}}

def get_femural_head_roi(femur_mask, medical_volume, output_dir, anatomy, visualize_method=True):
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
    coronal_image = zoom(coronal_image, (1, zoom_factor), order=3).round()
    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=3).round()

    dist_map = cv2.distanceTransform(sagittal_slice, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_sagittal, _, center_sagittal = cv2.minMaxLoc(dist_map)

    center_sagittal = list(center_sagittal)
    center_sagittal[0] = center_sagittal[0] / zoom_factor
    centroid = [round(center_of_mass[0]), center_sagittal[1], center_sagittal[0]]

    # fit another circle using the current axial centroid to update the left / right center
    axial_slice = femur_mask[:, :, round(centroid[2])]
    dist_map = cv2.distanceTransform(axial_slice, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_axial, _, center_axial = cv2.minMaxLoc(dist_map)
    centroid[0] = round(center_axial[1])

    # update the sagittal center
    sagittal_slice = femur_mask[round(centroid[0]), :, :]
    sagittal_slice = zoom(sagittal_slice, (1, zoom_factor), order=1).round()
    dist_map = cv2.distanceTransform(sagittal_slice, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_sagittal, _, center_sagittal = cv2.minMaxLoc(dist_map)
    centroid[1] = round(center_sagittal[1])
    centroid[2] = round(center_sagittal[0] / zoom_factor)

    axial_image = medical_volume.get_fdata()[:, :, round(centroid[2])]
    sagittal_image = medical_volume.get_fdata()[round(centroid[0]), :, :]
    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=3).round()

    if visualize_method:
        method_visualizer(
            sagittal_image,
            axial_image,
            axial_slice,
            sagittal_slice,
            center_sagittal,
            radius_sagittal,
            center_axial,
            radius_axial,
            output_dir,
            anatomy
        )

    roi = compute_hip_roi(medical_volume, centroid)

    #compute mean hu using medical_volume and roi
    hu = get_mean_roi_hu(medical_volume, roi)

    return (roi, centroid, hu)

def compute_hip_roi(img, centroid):
    pixel_spacing = img.header.get_zooms()
    length_i = 12.5 / pixel_spacing[0]
    length_j = 12.5 / pixel_spacing[1]
    length_k = 12.5 / pixel_spacing[2]

    roi = np.zeros(img.get_fdata().shape, dtype=np.uint8)
    i_lower = math.floor(centroid[0] - length_i)
    j_lower = math.floor(centroid[1] - length_j)
    k_lower = math.floor(centroid[2] - length_k)
    for i in range(i_lower, i_lower + 2 * math.ceil(length_i) + 1):
        for j in range(j_lower, j_lower + 2 * math.ceil(length_j) + 1):
            for k in range(k_lower, k_lower + 2 * math.ceil(length_k) + 1):
                if (i - centroid[0]) ** 2 / length_i**2 + (
                    j - centroid[1]
                ) ** 2 / length_j**2 + (k - centroid[2]) ** 2 / length_k**2 <= 1:
                    roi[i, j, k] = 1
    return roi

def get_mean_roi_hu(medical_volume, roi):
    masked_medical_volume = medical_volume.get_fdata() * roi
    return np.mean(masked_medical_volume[masked_medical_volume != 0])
