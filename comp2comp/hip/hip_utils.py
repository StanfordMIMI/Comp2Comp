import math
import os
import shutil

import cv2
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import sys

from comp2comp.hip.hip_visualization import method_visualizer


def compute_rois(medical_volume, segmentation, model, output_dir, save=False):
    left_femur_mask = segmentation.get_fdata() == model.categories["femur_left"]
    left_femur_mask = left_femur_mask.astype(np.uint8)
    right_femur_mask = segmentation.get_fdata() == model.categories["femur_right"]
    right_femur_mask = right_femur_mask.astype(np.uint8)
    left_head_roi, left_head_centroid, left_head_hu = get_femural_head_roi(
        left_femur_mask, medical_volume, output_dir, "left_head"
    )
    right_head_roi, right_head_centroid, right_head_hu = get_femural_head_roi(
        right_femur_mask, medical_volume, output_dir, "right_head"
    )
    left_intertrochanter_roi, left_intertrochanter_centroid, left_intertrochanter_hu = get_femural_head_roi(
        left_femur_mask, medical_volume, output_dir, "left_intertrochanter"
    )
    right_intertrochanter_roi, right_intertrochanter_centroid, right_intertrochanter_hu = get_femural_head_roi(
        right_femur_mask, medical_volume, output_dir, "right_intertrochanter"
    )
    if save:
        # make roi directory if it doesn't exist
        parent_output_dir = os.path.dirname(output_dir)
        roi_output_dir = os.path.join(parent_output_dir, "rois")
        if not os.path.exists(roi_output_dir):
            os.makedirs(roi_output_dir)

        # combine the left and right rois
        combined_roi = left_head_roi + (right_head_roi * 2)

        # Convert left ROI to NIfTI
        left_roi_nifti = nib.Nifti1Image(combined_roi, medical_volume.affine)
        left_roi_path = os.path.join(roi_output_dir, "roi.nii.gz")
        nib.save(left_roi_nifti, left_roi_path)
        # copy ../visualization/tunnelvision.ipynb to parent_output_dir
        shutil.copy(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "visualization",
                "tunnelvision.ipynb",
            ),
            parent_output_dir,
        )

    return {
        "left_head": {"roi": left_head_roi, "centroid": left_head_centroid, "hu": left_head_hu},
        "right_head": {"roi": right_head_roi, "centroid": right_head_centroid, "hu": right_head_hu},
        "left_intertrochanter": {"roi": left_intertrochanter_roi, "centroid": left_intertrochanter_centroid, "hu": left_intertrochanter_hu},
        "right_intertrochanter": {"roi": right_intertrochanter_roi, "centroid": right_intertrochanter_centroid, "hu": right_intertrochanter_hu},
    }

def get_femural_head_roi(femur_mask, medical_volume, output_dir, anatomy, visualize_method=False, min_pixel_count=20):
    # find the largest index that is not zero
    top = np.where(femur_mask.sum(axis=(0, 1)) != 0)[0].max()
    top_mask = femur_mask[:, :, top]

    print(f"======== Computing {anatomy} femur ROIs ========")

    while True:
        # Label connected components in the top_mask
        labeled, num_features = ndi.label(top_mask)

        # Check if there are two connected components with at least min_pixel_count pixels
        component_sizes = np.bincount(labeled.ravel())
        valid_components = np.where(component_sizes >= min_pixel_count)[0][1:]  # exclude background (label 0)

        if len(valid_components) == 2:
            break

        # Move top_mask down one slice
        top -= 1
        if top < 0:
            print("Two connected components not found in the femur mask.")
            break
        top_mask = femur_mask[:, :, top]

    if len(valid_components) == 2:
        # Find the center of mass for each connected component
        center_of_mass_1 = list(ndi.center_of_mass(top_mask, labeled, valid_components[0]))
        center_of_mass_2 = list(ndi.center_of_mass(top_mask, labeled, valid_components[1]))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.imshow(top_mask)
        plt.savefig(os.path.join(output_dir, f"{anatomy}_top_mask.png"))

        print(f"Center of mass for first connected component: {center_of_mass_1}")
        print(f"Center of mass for second connected component: {center_of_mass_2}")

    if anatomy == 'left_intertrochanter' or anatomy == 'right_head':
        center_of_mass = center_of_mass_1
        femur_mask[round(center_of_mass_2[0]):, :, :] = 0
    elif anatomy == 'right_intertrochanter' or anatomy == 'left_head':
        center_of_mass = center_of_mass_2
        femur_mask[:round(center_of_mass_1[0]), :, :] = 0

    coronal_slice = femur_mask[:, round(center_of_mass[1]), :]
    coronal_image = medical_volume.get_fdata()[:, round(center_of_mass[1]), :]
    sagittal_slice = femur_mask[round(center_of_mass[0]), :, :]
    sagittal_image = medical_volume.get_fdata()[round(center_of_mass[0]), :, :]

    zooms = medical_volume.header.get_zooms()
    zoom_factor = zooms[2] / zooms[1]

    coronal_slice = zoom(coronal_slice, (1, zoom_factor), order=1).round()
    coronal_image = zoom(coronal_image, (1, zoom_factor), order=3).round()
    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=3).round()

    centroid = [round(center_of_mass[0]), 0, 0]

    print(f"Starting centroid: {centroid}")

    for _ in range(3):
        sagittal_slice = femur_mask[centroid[0], :, :]
        sagittal_slice = zoom(sagittal_slice, (1, zoom_factor), order=1).round()
        centroid[1], centroid[2], radius_sagittal = inscribe_sagittal(sagittal_slice, zoom_factor)

        print(f"Centroid after inscribe sagittal: {centroid}")

        axial_slice = femur_mask[:, :, centroid[2]]
        centroid[0], centroid[1], radius_axial = inscribe_axial(axial_slice)

        print(f"Centroid after inscribe axial: {centroid}")

    axial_image = medical_volume.get_fdata()[:, :, round(centroid[2])]
    sagittal_image = medical_volume.get_fdata()[round(centroid[0]), :, :]
    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=3).round()

    if visualize_method:
        method_visualizer(
            sagittal_image,
            axial_image,
            axial_slice,
            sagittal_slice,
            [centroid[2], centroid[1]],
            radius_sagittal,
            [centroid[1], centroid[0]],
            radius_axial,
            output_dir,
            anatomy,
        )

    roi = compute_hip_roi(medical_volume, centroid, radius_sagittal, radius_axial)

    hu = get_mean_roi_hu(medical_volume, roi)

    return (roi, centroid, hu)


def compute_hip_roi(img, centroid, radius_sagittal, radius_axial):
    pixel_spacing = img.header.get_zooms()
    length_i = radius_axial * 0.75 / pixel_spacing[0]
    length_j = radius_axial * 0.75 / pixel_spacing[1]
    length_k = radius_sagittal * 0.75 / pixel_spacing[2]

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


def inscribe_axial(axial_mask):
    dist_map = cv2.distanceTransform(axial_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_axial, _, center_axial = cv2.minMaxLoc(dist_map)
    center_axial = list(center_axial)
    left_right_center = round(center_axial[1])
    posterior_anterior_center = round(center_axial[0])
    return left_right_center, posterior_anterior_center, radius_axial


def inscribe_sagittal(sagittal_mask, zoom_factor):
    dist_map = cv2.distanceTransform(sagittal_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius_sagittal, _, center_sagittal = cv2.minMaxLoc(dist_map)
    center_sagittal = list(center_sagittal)
    posterior_anterior_center = round(center_sagittal[1])
    inferior_superior_center = round(center_sagittal[0])
    inferior_superior_center = round(inferior_superior_center / zoom_factor)
    return posterior_anterior_center, inferior_superior_center, radius_sagittal


def get_mean_roi_hu(medical_volume, roi):
    masked_medical_volume = medical_volume.get_fdata() * roi
    return np.mean(masked_medical_volume[masked_medical_volume != 0])
