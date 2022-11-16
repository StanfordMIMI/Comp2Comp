import numpy as np
from pydicom.filereader import read_file_meta_info, dcmread
from glob import glob
from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt
import dosma as dm
import logging
import cv2
import sys
import random

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

# Function that takes a numpy array as input, computes the sagittal centroid of each label
# and returns a list of the centroids
def compute_centroids(seg: np.ndarray):
    """
    Compute the centroids of the labels.
    Parameters
    ----------
    seg: np.ndarray
        Segmentation volume.
    """
    centroids = []
    for label_idx in range(18, 24):
        pos = compute_centroid(seg, "sagittal", label_idx)
        centroids.append(pos)
    return centroids

# Function that takes a numpy array as input, as well as a list of centroids, takes a slice through the centroid on axis = 1 for each centroid
# and returns a list of the slices
def get_slices(seg: np.ndarray, centroids: List[int]):
    """
    Get the slices corresponding to the centroids.
    Parameters
    ----------
    seg: np.ndarray
        Segmentation volume.
    centroids: List[int]
        List of centroids.
    """
    slices = []
    label_idxs = list(range(18, 24))
    for i, centroid in enumerate(centroids):
        label_idx = label_idxs[i]
        slices.append((seg[:, centroid, :] == label_idx).astype(int))
    return slices

# Function that takes a mask and for each deletes the right most connected component
# Returns the mask with the right most connected component deleted
def delete_right_most_connected_component(mask: np.ndarray):
    """
    Delete the right most connected component.
    Parameters
    ----------
    mask: np.ndarray
        Mask volume.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    #print("INSIDE DELETE RIGHT MOST")
    #print(centroids)
    #plt.imshow(mask)
    #plt.savefig("./mask_before_delete_right_most.png")
    right_most_connected_component = np.argmax(centroids[1:, 1]) + 1
    mask[labels == right_most_connected_component] = 0
    return mask

#compute center of mass of 2d mask
def compute_center_of_mass(mask: np.ndarray):
    """
    Compute the center of mass of a 2D mask.
    Parameters
    ----------
    mask: np.ndarray
        Mask volume.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity = 8)
    center_of_mass = np.mean(centroids[1:, :], axis = 0)
    return center_of_mass


#Function that takes a 3d centroid and retruns a binary mask with a 3d roi around the centroid
def roi_from_mask(img: np.ndarray, centroid: np.ndarray):
    """
    Compute a 3D ROI from a 3D mask.
    Parameters
    ----------
    img: np.ndarray
        Image volume.
    centroid: np.ndarray
        Centroid.
    """
    roi = np.zeros(img.shape)
    length = 5
    roi[int(centroid[0] - length):int(centroid[0] + length), int(centroid[1] - length):int(centroid[1] + length), int(centroid[2] - length):int(centroid[2] + length)] = 1
    return roi


#Function that takes a 3d image and a 3d binary mask and returns that average value of the image inside the mask
def mean_img_mask(img: np.ndarray, mask: np.ndarray):
    """
    Compute the mean of an image inside a mask.
    Parameters
    ----------
    img: np.ndarray
        Image volume.
    mask: np.ndarray
        Mask volume.
    """
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)
    img_masked = (img * mask)[mask > 0]
    mean = np.mean(img_masked)
    '''
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k] == 1:
                    plt.imshow(img[:, j, :] + (mask[:, j, :] * 1000))
                    #generate random int
                    random_int = random.randint(0, 100000000)
                    plt.savefig(f"./img_{random_int}.png")
                    #plt.imshow(mask[i, :, :])
                    #plt.savefig("./mask.png")
                    #sys.exit()
                    return mean
    '''
    return mean


def compute_rois(seg, img):
    """
    Compute the ROIs for the spine.
    Parameters
    ----------
    seg: np.ndarray
        Segmentation volume.
    img: np.ndarray
        Image volume.
    """
    # Compute centroids
    centroids = compute_centroids(seg)
    # Get slices
    slices = get_slices(seg, centroids)
    # Delete right most connected component
    for i, slice in enumerate(slices):
        slices[i] = delete_right_most_connected_component(slice)
    # Compute ROIs
    rois = []
    spine_hus = []
    centroids_3d = []
    for i, slice in enumerate(slices):
        center_of_mass = compute_center_of_mass(slice)
        centroid = np.array([center_of_mass[1], centroids[i], center_of_mass[0]])
        roi = roi_from_mask(img, centroid)
        spine_hus.append(mean_img_mask(img, roi))
        rois.append(roi)
        centroids_3d.append(centroid)
    return (spine_hus, rois, centroids_3d)


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


def visualize_coronal_sagittal_spine(seg: np.ndarray, rois: List[np.ndarray], mvs: dm.MedicalVolume, centroids: List[int], label_text: List[str], output_dir: str, spine_hus = None):
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
    for roi in rois:
        one_hot_roi_label = roi[:, sagittal_centroid, :]
        one_hot_sag_label = np.concatenate((one_hot_sag_label, one_hot_roi_label.reshape((one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1))), axis = 2)
    
    coronal_image = mvs.volume[coronal_centroid, :, :]
    coronal_label = seg[coronal_centroid, :, :]
    one_hot_cor_label = to_one_hot(coronal_label)
    for roi in rois:
        one_hot_roi_label = roi[coronal_centroid, :, :]
        one_hot_cor_label = np.concatenate((one_hot_cor_label, one_hot_roi_label.reshape((one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1))), axis = 2)

    visualization.save_binary_segmentation_overlay(np.transpose(coronal_image), np.transpose(one_hot_cor_label, (1, 0, 2)), output_dir, "spine_coronal.png", centroids, spine_hus = spine_hus)
    visualization.save_binary_segmentation_overlay(np.transpose(sagittal_image), np.transpose(one_hot_sag_label, (1, 0, 2)), output_dir, "spine_sagittal.png", centroids, spine_hus = spine_hus)