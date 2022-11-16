import numpy as np
from pydicom.filereader import read_file_meta_info, dcmread
from glob import glob
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import dosma as dm
import logging

from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences
from abctseg.utils import visualization 

def find_spine_dicoms(seg: np.ndarray, path: str):
    """
    Find the dicom files corresponding to the spine T12 - L5 levels.
    Parameters
    ----------
    seg: np.ndarray
        Segmentation volume.
    """
    vertical_positions = []
    for label_idx in range(18, 24):
        pos = compute_centroid(seg, "axial", label_idx)
        vertical_positions.append(pos)

    # Log vertical positions
    logging.info(f"Instance numbers: {vertical_positions}")

    folder_in = path
    instance_numbers = []
    label_text = ['T12_seg', 'L1_seg', 'L2_seg', 'L3_seg', 'L4_seg', 'L5_seg']

    dicom_files = []
    for dicom_path in glob(folder_in + "/*.dcm"):
        instance_number = dcmread(dicom_path).InstanceNumber
        if instance_number in vertical_positions:
            dicom_files.append(dicom_path)
            instance_numbers.append(instance_number)

    dicom_files = [x for _, x in sorted(zip(instance_numbers, dicom_files))]
    instance_numbers.sort(reverse = True)

    return (dicom_files, label_text, instance_numbers)


def compute_centroid(seg: np.ndarray, plane: str, label: int):
    """
    Compute the centroid of a label in a given plane.
    Parameters
    ----------
    seg: np.ndarray
        Segmentation volume.
    plane: str
        Plane to compute the centroid.
    label: int
        Label to compute the centroid.
    """
    if plane == "axial":
        sum_out_axes = (0, 1)
        sum_axis = 2
    elif plane == "coronal":
        sum_out_axes = (1, 2)
        sum_axis = 0
    elif plane == "sagittal":
        sum_out_axes = (0, 2)
        sum_axis = 1
    sums = np.sum(seg == label, axis = sum_out_axes)
    normalized_sums = sums / np.sum(sums)
    pos = int(np.sum(np.arange(0, seg.shape[sum_axis]) * normalized_sums))
    return pos

def to_one_hot(label: np.ndarray):
    """
    Convert a label to one-hot encoding.
    Parameters
    ----------
    label: np.ndarray
        Label volume.
    """
    one_hot_label = np.zeros((label.shape[0], label.shape[1], 7))
    one_hot_label[:, :, 1] = (label == 18).astype(int)
    one_hot_label[:, :, 2] = (label == 19).astype(int)
    one_hot_label[:, :, 3] = (label == 20).astype(int)
    one_hot_label[:, :, 4] = (label == 21).astype(int)
    one_hot_label[:, :, 5] = (label == 22).astype(int)
    one_hot_label[:, :, 6] = (label == 23).astype(int)
    return one_hot_label


def visualize_coronal_sagittal_spine(seg: np.ndarray, mvs: dm.MedicalVolume, centroids: List[int], label_text: List[str], output_dir: str):
    """
    Visualize the coronal and sagittal planes of the spine.
    Parameters
    ----------
    seg: np.ndarray
        Segmentation volume.
    mvs: dm.MedicalVolume
        MVS volume.
    centroids: List[int]
        Centroids of the labels.
    label_text: List[str]
        Labels text.
    output_dir: str
        Output directory.
    """
    for_centroid = np.logical_and(seg >= 18, seg <= 23).astype(int) 
    sagittal_centroid = compute_centroid(for_centroid, 'sagittal', 1)
    coronal_centroid = compute_centroid(for_centroid, 'coronal', 1)

    #Spine visualizations 
    sagittal_image = mvs.volume[:, sagittal_centroid, :]
    sagittal_label = seg[:, sagittal_centroid, :]
    one_hot_sag_label = to_one_hot(sagittal_label)
    
    coronal_image = mvs.volume[coronal_centroid, :, :]
    coronal_label = seg[coronal_centroid, :, :]
    one_hot_cor_label = to_one_hot(coronal_label)

    visualization.save_binary_segmentation_overlay(np.transpose(coronal_image), np.transpose(one_hot_cor_label, (1, 0, 2)), output_dir, "spine_coronal.png", centroids)
    visualization.save_binary_segmentation_overlay(np.transpose(sagittal_image), np.transpose(one_hot_sag_label, (1, 0, 2)), output_dir, "spine_sagittal.png", centroids)