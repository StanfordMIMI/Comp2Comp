import logging
from glob import glob
from typing import List

import cv2
import dosma as dm
import numpy as np
from pydicom.filereader import dcmread
import nibabel as nib
from itertools import groupby
import re
import matplotlib.pyplot as plt

from comp2comp.preferences import PREFERENCES
from comp2comp.models import Models
from comp2comp.utils import visualization


def find_spine_dicoms(seg: np.ndarray, path: str, model_type):
    """Find the dicom files corresponding to the spine T12 - L5 levels.

    Args:
        seg (np.ndarray): Segmentation volume.
        path (str): Path to the dicom files.
        model_type (str): Model type.

    Returns:
        List[str]: List of dicom files.
    """
    vertical_positions = []
    label_idxs = list(model_type.categories.values())
    for label_idx in label_idxs:
        pos = compute_centroid(seg, "axial", label_idx)
        vertical_positions.append(pos)

    # Log vertical positions
    logging.info(f"Instance numbers: {vertical_positions}")

    folder_in = path
    instance_numbers = []

    # TODO Make these names configurable
    label_text = ["T12_seg", "L1_seg", "L2_seg", "L3_seg", "L4_seg", "L5_seg"]

    dicom_files = []
    for dicom_path in glob(folder_in + "/*.dcm"):
        instance_number = dcmread(dicom_path).InstanceNumber
        if instance_number in vertical_positions:
            dicom_files.append(dicom_path)
            instance_numbers.append(instance_number)

    dicom_files = [x for _, x in sorted(zip(instance_numbers, dicom_files))]
    instance_numbers.sort(reverse=True)

    return (dicom_files, label_text, instance_numbers)


# Function that takes a numpy array as input, computes the
# sagittal centroid of each label and returns a list of the
# centroids
def compute_centroids(seg: np.ndarray, spine_model_type):
    """Compute the centroids of the labels.

    Args:
        seg (np.ndarray): Segmentation volume.
        spine_model_type (str): Model type.

    Returns:
        List[int]: List of centroids.
    """
    # take values of spine_model_type.categories dictionary
    # and convert to list
    label_idxs = list(spine_model_type.categories.values())
    centroids = []
    for label_idx in label_idxs:
        pos = compute_centroid(seg, "sagittal", label_idx)
        centroids.append(pos)
    return centroids


# Function that takes a numpy array as input, as well as a list of centroids,
# takes a slice through the centroid on axis = 1 for each centroid
# and returns a list of the slices
def get_slices(seg: np.ndarray, centroids: List[int], spine_model_type):
    """Get the slices corresponding to the centroids.

    Args:
        seg (np.ndarray): Segmentation volume.
        centroids (List[int]): List of centroids.
        spine_model_type (str): Model type.

    Returns:
        List[np.ndarray]: List of slices.
    """
    label_idxs = list(spine_model_type.categories.values())
    slices = []
    for i, centroid in enumerate(centroids):
        label_idx = label_idxs[i]
        slices.append((seg[:, centroid, :] == label_idx).astype(int))
    return slices


# Function that takes a mask and for each deletes the right most
# connected component. Returns the mask with the right most
# connected component deleted
def delete_right_most_connected_component(mask: np.ndarray):
    """Delete the right most connected component corresponding to spinous processes.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Mask volume.
    """
    mask = mask.astype(np.uint8)
    _, labels, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    right_most_connected_component = np.argmax(centroids[1:, 1]) + 1
    mask[labels == right_most_connected_component] = 0
    return mask


# compute center of mass of 2d mask
def compute_center_of_mass(mask: np.ndarray):
    """Compute the center of mass of a 2D mask.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Center of mass.
    """
    mask = mask.astype(np.uint8)
    _, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    center_of_mass = np.mean(centroids[1:, :], axis=0)
    return center_of_mass


# Function that takes a 3d centroid and retruns a binary mask with a 3d
# roi around the centroid
def roi_from_mask(img: np.ndarray, centroid: np.ndarray):
    """Compute a 3D ROI from a 3D mask.

    Args:
        img (np.ndarray): Image volume.
        centroid (np.ndarray): Centroid.

    Returns:
        np.ndarray: ROI volume.
    """
    roi = np.zeros(img.shape)
    length = 5
    roi[
        int(centroid[0] - length) : int(centroid[0] + length),
        int(centroid[1] - length) : int(centroid[1] + length),
        int(centroid[2] - length) : int(centroid[2] + length),
    ] = 1
    return roi


# Function that takes a 3d image and a 3d binary mask and returns that average
# value of the image inside the mask
def mean_img_mask(
    img: np.ndarray,
    mask: np.ndarray,
    rescale_slope: float,
    rescale_intercept: float,
):
    """Compute the mean of an image inside a mask.

    Args:
        img (np.ndarray): Image volume.
        mask (np.ndarray): Mask volume.
        rescale_slope (float): Rescale slope.
        rescale_intercept (float): Rescale intercept.

    Returns:
        float: Mean value.
    """
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)
    img_masked = (img * mask)[mask > 0]
    mean = (rescale_slope * np.mean(img_masked)) + rescale_intercept
    return mean


def compute_rois(seg, img, rescale_slope, rescale_intercept, spine_model_type):
    """Compute the ROIs for the spine.

    Args:
        seg (np.ndarray): Segmentation volume.
        img (np.ndarray): Image volume.
        rescale_slope (float): Rescale slope.
        rescale_intercept (float): Rescale intercept.
        spine_model_type (Models): Model type.

    Returns:
        spine_hus (List[float]): List of HU values.
        rois (List[np.ndarray]): List of ROIs.
        centroids_3d (List[np.ndarray]): List of centroids.
    """
    # Compute centroids
    centroids = compute_centroids(seg, spine_model_type)
    # Get slices
    slices = get_slices(seg, centroids, spine_model_type)
    # Delete right most connected component
    if spine_model_type.model_name == "ts_spine":
        for i, slice in enumerate(slices):
            # keep only the two largest connected components 
            two_largest = keep_two_largest_connected_components(slice)
            slices[i] = delete_right_most_connected_component(two_largest)
    # Compute ROIs
    rois = []
    spine_hus = []
    centroids_3d = []
    for i, slice in enumerate(slices):
        center_of_mass = compute_center_of_mass(slice)
        centroid = np.array([center_of_mass[1], centroids[i], center_of_mass[0]])
        roi = roi_from_mask(img, centroid)
        spine_hus.append(mean_img_mask(img, roi, rescale_slope, rescale_intercept))
        rois.append(roi)
        centroids_3d.append(centroid)
    return (spine_hus, rois, centroids_3d)

def keep_two_largest_connected_components(mask: np.ndarray):
    """Keep the two largest connected components.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Mask volume.
    """
    mask = mask.astype(np.uint8)
    # sort connected components by size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[1:, 4]
    sorted_indices = np.argsort(stats)[::-1]
    # keep only the two largest connected components
    mask = np.zeros(mask.shape)
    for i in range(2):
        mask[labels == sorted_indices[i] + 1] = 1
    return mask


def compute_centroid(seg: np.ndarray, plane: str, label: int):
    """Compute the centroid of a label in a given plane.

    Args:
        seg (np.ndarray): Segmentation volume.
        plane (str): Plane.
        label (int): Label.

    Returns:
        int: Centroid.
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
    sums = np.sum(seg == label, axis=sum_out_axes)
    normalized_sums = sums / np.sum(sums)
    pos = int(np.sum(np.arange(0, seg.shape[sum_axis]) * normalized_sums))
    return pos


def to_one_hot(label: np.ndarray, model_type):
    """Convert a label to one-hot encoding.

    Args:
        label (np.ndarray): Label volume.
        model_type (Models): Model type.

    Returns:
        np.ndarray: One-hot encoding volume.
    """
    label_idxs = list(model_type.categories.values())
    one_hot_label = np.zeros((label.shape[0], label.shape[1], len(label_idxs)))
    for i, idx in enumerate(label_idxs):
        one_hot_label[:, :, i] = (label == idx).astype(int)
    return one_hot_label


def visualize_coronal_sagittal_spine(
    seg: np.ndarray,
    rois: List[np.ndarray],
    mvs: dm.MedicalVolume,
    centroids: List[int],
    label_text: List[str],
    output_dir: str,
    spine_hus=None,
    model_type=None,
):
    """Visualize the coronal and sagittal planes of the spine.

    Args:
        seg (np.ndarray): Segmentation volume.
        rois (List[np.ndarray]): List of ROIs.
        mvs (dm.MedicalVolume): Medical volume.
        centroids (List[int]): List of centroids.
        label_text (List[str]): List of labels.
        output_dir (str): Output directory.
        spine_hus (List[float], optional): List of HU values. Defaults to None.
        model_type (Models, optional): Model type. Defaults to None.
    """
    # Get minimum and maximum values of the model_type.catogories dict values
    min_val = min(model_type.categories.values())
    max_val = max(model_type.categories.values())
    for_centroid = np.logical_and(seg >= min_val, seg <= max_val).astype(int)
    sagittal_centroid = compute_centroid(for_centroid, "sagittal", 1)
    coronal_centroid = compute_centroid(for_centroid, "coronal", 1)

    # Spine visualizations
    sagittal_image = mvs.volume[:, sagittal_centroid, :]
    sagittal_label = seg[:, sagittal_centroid, :]
    one_hot_sag_label = to_one_hot(sagittal_label, model_type)
    for roi in rois:
        one_hot_roi_label = roi[:, sagittal_centroid, :]
        one_hot_sag_label = np.concatenate(
            (
                one_hot_sag_label,
                one_hot_roi_label.reshape(
                    (one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1)
                ),
            ),
            axis=2,
        )

    coronal_image = mvs.volume[coronal_centroid, :, :]
    coronal_label = seg[coronal_centroid, :, :]
    one_hot_cor_label = to_one_hot(coronal_label, model_type)
    for roi in rois:
        one_hot_roi_label = roi[coronal_centroid, :, :]
        one_hot_cor_label = np.concatenate(
            (
                one_hot_cor_label,
                one_hot_roi_label.reshape(
                    (one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1)
                ),
            ),
            axis=2,
        )

    visualization.save_binary_segmentation_overlay(
        np.transpose(coronal_image),
        np.transpose(one_hot_cor_label, (1, 0, 2)),
        output_dir,
        "spine_coronal.png",
        centroids,
        spine_hus=spine_hus,
        model_type=model_type,
    )
    visualization.save_binary_segmentation_overlay(
        np.transpose(sagittal_image),
        np.transpose(one_hot_sag_label, (1, 0, 2)),
        output_dir,
        "spine_sagittal.png",
        centroids,
        spine_hus=spine_hus,
        model_type=model_type,
    )

def compare_ts_stanford_centroids(labels_path):
    """Compare the centroids of the Stanford dataset with the centroids of the TS dataset.

    Args:
        labels_path (str): Path to the Stanford dataset labels.
    """
    t12_diff = []
    l1_diff = []
    l2_diff = []
    l3_diff = []
    l4_diff = []
    l5_diff = []
    num_skipped = 0

    labels = glob(labels_path + "/*")
    for i, label_path in enumerate(labels):
        # modify label_path to give pred_path
        pred_path = label_path.replace('labelsTs', 'predTs_TS')
        print(label_path.split('/')[-1])
        label_nib  = nib.load(label_path)
        label = label_nib.get_fdata()
        spacing = label_nib.header.get_zooms()[2]
        pred_nib  = nib.load(pred_path)
        pred = pred_nib.get_fdata()
        if True:
            pred[pred == 18] = 6
            pred[pred == 19] = 5
            pred[pred == 20] = 4
            pred[pred == 21] = 3
            pred[pred == 22] = 2
            pred[pred == 23] = 1
        
        for label_idx in range(1, 7):
            label_level = label == label_idx
            indexes = np.array(range(label.shape[2]))
            sums = np.sum(label_level, axis = (0, 1))
            normalized_sums = sums / np.sum(sums)
            label_centroid = np.sum(indexes * normalized_sums)
            print(f"Centroid for label {label_idx}: {label_centroid}")
    
            if False:
                try:
                    pred_centroid = pred_centroids[6 - label_idx]
                except:
                    # Change this part
                    print("Something wrong with pred_centroids, skipping!")
                    num_skipped += 1
                    break
            
            #if revert_to_original:
            if True:
                pred_level = pred == label_idx
                sums = np.sum(pred_level, axis = (0, 1))
                indices = list(range(sums.shape[0]))
                groupby_input = zip(indices, list(sums))
                g = groupby(groupby_input, key=lambda x:x[1]>0.0)
                m = max([list(s) for v, s in g if v > 0], key = lambda x:np.sum(list(zip(*x))[1]))
                res = list(zip(*m))
                indexes = list(res[0])
                sums = list(res[1])
                normalized_sums = sums / np.sum(sums)
                pred_centroid = np.sum(indexes * normalized_sums)
            print(f"Centroid for prediction {label_idx}: {pred_centroid}")
            
            diff = np.absolute(pred_centroid - label_centroid) * spacing
            
            if label_idx == 1:
                t12_diff.append(diff)
            elif label_idx == 2:
                l1_diff.append(diff)
            elif label_idx == 3:
                l2_diff.append(diff)
            elif label_idx == 4:
                l3_diff.append(diff)
            elif label_idx == 5:
                l4_diff.append(diff)
            elif label_idx == 6:
                l5_diff.append(diff)
            
    print(f"Processed {i + 1} scans!\n")
    print(f"Skipped {num_skipped}")
    print(f"The final mean differences in mm:")
    print(np.mean(t12_diff), np.mean(l1_diff), np.mean(l2_diff), np.mean(l3_diff), np.mean(l4_diff), np.mean(l5_diff))
    print(f"The final median differences in mm:")
    print(np.median(t12_diff), np.median(l1_diff), np.median(l2_diff), np.median(l3_diff), np.median(l4_diff), np.median(l5_diff))

def compare_ts_stanford_roi_hus(image_path):
    """Compare the HU values of the Stanford dataset with the HU values of the TS dataset.

    image_path (str): Path to the Stanford dataset images.
    """
    img_paths = glob(image_path + "/*")
    differences = np.zeros((40, 6))
    ground_truth = np.zeros((40, 6))
    for i, img_path in enumerate(img_paths):
        print(f"Image number {i + 1}")
        image_path_no_0000 = re.sub(r'_0000', '', img_path)
        ts_seg_path = image_path_no_0000.replace('imagesTs', 'predTs_TS')
        stanford_seg_path = image_path_no_0000.replace('imagesTs', 'labelsTs')
        img = nib.load(img_path).get_fdata()
        img = np.swapaxes(img, 0, 1)
        ts_seg = nib.load(ts_seg_path).get_fdata()
        ts_seg = np.swapaxes(ts_seg, 0, 1)
        stanford_seg = nib.load(stanford_seg_path).get_fdata()
        stanford_seg = np.swapaxes(stanford_seg, 0, 1)
        ts_model_type = Models.model_from_name("ts_spine")
        (spine_hus_ts, rois, centroids_3d) = compute_rois(ts_seg, img, 1, 0, ts_model_type)
        stanford_model_type = Models.model_from_name("stanford_spine_v0.0.1")
        (spine_hus_stanford, rois, centroids_3d) = compute_rois(stanford_seg, img, 1, 0, stanford_model_type)
        difference_vals = np.abs(np.array(spine_hus_ts) - np.array(spine_hus_stanford))
        print(f"Differences {difference_vals}\n")
        differences[i, :] = difference_vals
        ground_truth[i, :] = spine_hus_stanford
        print("\n")
    # compute average percent change from ground truth
    percent_change = np.divide(differences, ground_truth) * 100
    average_percent_change = np.mean(percent_change, axis=0)
    median_percent_change = np.median(percent_change, axis=0)
    # print average percent change
    print("Average percent change from ground truth:")
    print(average_percent_change)
    print("Median percent change from ground truth:")
    print(median_percent_change)
    # print average difference
    average_difference = np.mean(differences, axis=0)
    median_difference = np.median(differences, axis = 0)
    print("Average difference from ground truth:")
    print(average_difference)
    print("Median difference from ground truth:")
    print(median_difference)

def process_post_hoc(pred_path):
    """Apply post-hoc heuristics for improving Stanford spine model vertical centroid predictions.

    Args:
        pred_path (str): Path to the prediction.
    """
    pred_nib  = nib.load(pred_path)
    spacing = pred_nib.header.get_zooms()[2]
    pred = pred_nib.get_fdata()

    pred_bodies = np.logical_and(pred >= 1, pred <= 6)
    pred_bodies = pred_bodies.astype(np.int64)
    
    labels_out, N = cc3d.connected_components(pred_bodies, return_N=True, connectivity=6)
    
    stats = cc3d.statistics(labels_out)
    print(stats)
    
    labels_out_list = []
    voxel_counts_list = list(stats['voxel_counts'])
    for idx_lab in range(1, N + 2):
        labels_out_list.append(labels_out == idx_lab)
    
    centroids_list = list(stats['centroids'][:, 2])
    
    labels = []
    centroids = []
    voxels = []
    
    for idx, count in enumerate(voxel_counts_list):
        if count > 10000:
            labels.append(labels_out_list[idx])
            centroids.append(centroids_list[idx])
            voxels.append(count)
        
    top_comps = [(counts0, labels0, centroids0) for counts0, labels0, centroids0 in sorted(zip(voxels, labels, centroids), reverse = True)]
    top_comps = top_comps[1:7]
    
    # ====== Check whether the connected components are fusing vertebral bodies ======
    revert_to_original = False
    
    volumes = list(zip(*top_comps))[0]
    if volumes[0] > 1.5 * volumes[1]:
        revert_to_original = True
        print("Reverting to original...")
    
    labels = list(zip(*top_comps))[1]
    centroids = list(zip(*top_comps))[2]
        
    top_comps = zip(centroids, labels)
    pred_centroids = [x for x, _ in sorted(top_comps)]

    for label_idx in range(1, 7):
        if not revert_to_original:
            try:
                pred_centroid = pred_centroids[6 - label_idx]
            except:
                # Change this part
                print("Something went wrong with post processing, probably < 6 predicted bodies. Reverting to original labels.")
                revert_to_original = True
                
        if revert_to_original:
            pred_level = pred == label_idx
            sums = np.sum(pred_level, axis = (0, 1))
            indices = list(range(sums.shape[0]))
            groupby_input = zip(indices, list(sums))
            #sys.exit()
            g = groupby(groupby_input, key=lambda x:x[1]>0.0)
            m = max([list(s) for v, s in g if v > 0], key = lambda x:np.sum(list(zip(*x))[1]))
            #sys.exit()
            #m = max([list(s) for v, s in g], key=lambda np.sum)
            res = list(zip(*m))
            indexes = list(res[0])
            sums = list(res[1])
            normalized_sums = sums / np.sum(sums)
            pred_centroid = np.sum(indexes * normalized_sums)
        print(f"Centroid for prediction {label_idx}: {pred_centroid}")