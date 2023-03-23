import logging
from glob import glob
from typing import List, Dict
import sys

import cv2
import numpy as np
from pydicom.filereader import dcmread
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

from comp2comp.spine import spine_visualization


def find_spine_dicoms(seg: np.ndarray, path: str, model_type, flip_si, levels):
    """Find the dicom files corresponding to the spine T12 - L5 levels.

    Args:
        seg (np.ndarray): Segmentation volume.
        path (str): Path to the dicom files.
        model_type (str): Model type.

    Returns:
        List[str]: List of dicom files.
    """
    # flip the last axis of seg
    seg = np.flip(seg, 2)
    vertical_positions = []

    # extract label_idxs using levels
    label_idxs = [model_type.categories[level] for level in levels]
    for i, label_idx in enumerate(label_idxs):
        level = levels[i]
        pos = compute_centroid(seg, "axial", label_idx)
        vertical_positions.append(pos)

    folder_in = path
    instance_numbers = []

    # if flip_si is True, then flip the vertical positions
    if flip_si:
        vertical_positions_dcm = [(seg.shape[2] - x) for x in vertical_positions.values()]
    else:
        vertical_positions_dcm = vertical_positions

    dicom_files = []
    instance_numbers = []
    for dicom_path in glob(folder_in + "/*.dcm"):
        instance_number = dcmread(dicom_path).InstanceNumber
        if instance_number in vertical_positions_dcm:
            dicom_files.append(dicom_path)
            instance_numbers.append(instance_number)

    if flip_si:
        dicom_files = [x for _, x in sorted(zip(instance_numbers, dicom_files), reverse=True)]
    else:
        dicom_files = [x for _, x in sorted(zip(instance_numbers, dicom_files))]
    vertical_positions.sort(reverse=True)

    # reverse the order of levels
    levels.reverse()

    return (dicom_files, levels, vertical_positions)


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
    centroids = {}
    for level in spine_model_type.categories:
        label_idx = spine_model_type.categories[level]
        try:
            pos = compute_centroid(seg, "sagittal", label_idx)
            centroids[level] = pos
        except Exception:
            logging.warning(f"Label {level} not found in segmentation volume.")
    return centroids


# Function that takes a numpy array as input, as well as a list of centroids,
# takes a slice through the centroid on axis = 1 for each centroid
# and returns a list of the slices
def get_slices(seg: np.ndarray, centroids: Dict, spine_model_type):
    """Get the slices corresponding to the centroids.

    Args:
        seg (np.ndarray): Segmentation volume.
        centroids (List[int]): List of centroids.
        spine_model_type (str): Model type.

    Returns:
        List[np.ndarray]: List of slices.
    """
    seg = seg.astype(np.uint8)
    slices = {}
    for level in centroids:
        label_idx = spine_model_type.categories[level]
        binary_seg = (seg[centroids[level], :, :] == label_idx).astype(int)
        if np.sum(binary_seg) > 400: # heuristic to make sure enough of the body is showing
            slices[level] = binary_seg
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
    right_most_connected_component = np.argmin(centroids[1:, 1]) + 1
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
def roi_from_mask(img, centroid: np.ndarray):
    """Compute a 3D ROI from a 3D mask.

    Args:
        img (np.ndarray): Image volume.
        centroid (np.ndarray): Centroid.

    Returns:
        np.ndarray: ROI volume.
    """
    roi = np.zeros(img.shape)

    img_np = img.get_fdata()

    pixel_spacing = img.header.get_zooms()
    length_i = int(5.0 / pixel_spacing[0])
    length_j = int(5.0 / pixel_spacing[1])
    length_k = int(5.0 / pixel_spacing[2])

    # cubic ROI around centroid
    """
    roi[
        int(centroid[0] - length) : int(centroid[0] + length),
        int(centroid[1] - length) : int(centroid[1] + length),
        int(centroid[2] - length) : int(centroid[2] + length),
    ] = 1
    """
    # spherical ROI around centroid
    roi = np.zeros(img_np.shape)
    i_lower = int(centroid[0] - length_i)
    j_lower = int(centroid[1] - length_j)
    k_lower = int(centroid[2] - length_k)
    for i in range(i_lower, i_lower + 2 * length_i):
        for j in range(j_lower, j_lower + 2 * length_j):
            for k in range(k_lower, k_lower + 2 * length_k):
                if (i - centroid[0]) ** 2 / length_i**2 + (
                    j - centroid[1]
                ) ** 2 / length_j**2 + (k - centroid[2]) ** 2 / length_k**2 <= 1:
                    roi[i, j, k] = 1
    return roi


# Function that takes a 3d image and a 3d binary mask and returns that average
# value of the image inside the mask
def mean_img_mask(img: np.ndarray, mask: np.ndarray, index: int):
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
    # mean = (rescale_slope * np.mean(img_masked)) + rescale_intercept
    # median = (rescale_slope * np.median(img_masked)) + rescale_intercept
    mean = np.mean(img_masked)
    return mean


def compute_rois(seg, img, spine_model_type):
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
    seg_np = seg.get_fdata()
    centroids = compute_centroids(seg_np, spine_model_type)
    print("centroids: ", centroids)
    slices = get_slices(seg_np, centroids, spine_model_type)
    print("slices: ", slices)
    for level in slices:
        slice = slices[level]
        print(level)
        print(slice)
        # keep only the two largest connected components
        two_largest, two = keep_two_largest_connected_components(slice)
        if two:
            slices[level] = delete_right_most_connected_component(two_largest)

    # Compute ROIs
    rois = {}
    spine_hus = {}
    centroids_3d = {}
    for i, level in enumerate(slices):
        slice = slices[level]
        center_of_mass = compute_center_of_mass(slice)
        centroid = np.array([centroids[level], center_of_mass[1], center_of_mass[0]])
        roi = roi_from_mask(img, centroid)
        spine_hus[level] = mean_img_mask(img.get_fdata(), roi, i)
        rois[level] = roi
        centroids_3d[level] = centroid
    return (spine_hus, rois, centroids_3d)


def keep_two_largest_connected_components(mask: Dict):
    """Keep the two largest connected components.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Mask volume.
    """
    plt.imshow(mask)
    plt.savefig('./mask.png')
    mask = mask.astype(np.uint8)
    # sort connected components by size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[1:, 4]
    sorted_indices = np.argsort(stats)[::-1]
    # keep only the two largest connected components
    mask = np.zeros(mask.shape)
    mask[labels == sorted_indices[0] + 1] = 1
    two = True
    try:
        mask[labels == sorted_indices[1] + 1] = 1
    except Exception:
        two = False
    return (mask, two)


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
    elif plane == "sagittal":
        sum_out_axes = (1, 2)
        sum_axis = 0
    elif plane == "coronal":
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
    mvs: np.ndarray,
    centroids: List[int],
    centroids_3d: np.ndarray,
    label_text: List[str],
    output_dir: str,
    spine_hus=None,
    model_type=None,
    pixel_spacing=None,
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

    sagittal_vals, coronal_vals = curved_planar_reformation(mvs, centroids_3d)
    sagittal_image = mvs[:, sagittal_vals, range(len(sagittal_vals))]
    sagittal_label = seg[:, sagittal_vals, range(len(sagittal_vals))]
    zoom_factor = pixel_spacing[2] / pixel_spacing[1]
    sagittal_image = zoom(sagittal_image, (1, zoom_factor), order=3)
    sagittal_label = zoom(sagittal_label, (1, zoom_factor), order=0).astype(int)

    one_hot_sag_label = to_one_hot(sagittal_label, model_type)
    for roi in rois:
        one_hot_roi_label = roi[:, sagittal_vals, range(len(sagittal_vals))]
        one_hot_roi_label = zoom(one_hot_roi_label, (1, zoom_factor), order=0).astype(int)
        one_hot_sag_label = np.concatenate(
            (
                one_hot_sag_label,
                one_hot_roi_label.reshape(
                    (one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1)
                ),
            ),
            axis=2,
        )

    coronal_image = mvs[coronal_vals, :, range(len(coronal_vals))]
    coronal_label = seg[coronal_vals, :, range(len(coronal_vals))]
    coronal_image = zoom(coronal_image, (zoom_factor, 1), order=3)
    coronal_label = zoom(coronal_label, (zoom_factor, 1), order=0).astype(int)

    one_hot_cor_label = to_one_hot(coronal_label, model_type)
    for roi in rois:
        one_hot_roi_label = roi[coronal_vals, :, range(len(coronal_vals))]
        one_hot_roi_label = zoom(one_hot_roi_label, (zoom_factor, 1), order=0).astype(int)
        one_hot_cor_label = np.concatenate(
            (
                one_hot_cor_label,
                one_hot_roi_label.reshape(
                    (one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1)
                ),
            ),
            axis=2,
        )

    # flip both axes of coronal image
    coronal_image = np.flip(coronal_image, axis=0)
    coronal_image = np.flip(coronal_image, axis=1)

    # flip both axes of coronal label
    one_hot_cor_label = np.flip(one_hot_cor_label, axis=0)
    one_hot_cor_label = np.flip(one_hot_cor_label, axis=1)

    sagittal_image = np.transpose(sagittal_image)
    one_hot_sag_label = np.transpose(one_hot_sag_label, (1, 0, 2))

    # flip both axes of sagittal image
    sagittal_image = np.flip(sagittal_image, axis=0)
    sagittal_image = np.flip(sagittal_image, axis=1)

    # flip both axes of sagittal label
    one_hot_sag_label = np.flip(one_hot_sag_label, axis=0)
    one_hot_sag_label = np.flip(one_hot_sag_label, axis=1)

    spine_visualization.save_binary_segmentation_overlay(
        coronal_image,
        one_hot_cor_label,
        output_dir,
        "spine_sagittal.png",
        centroids,
        spine_hus=spine_hus,
        model_type=model_type,
        pixel_spacing=pixel_spacing,
        levels=label_text
    )
    spine_visualization.save_binary_segmentation_overlay(
        sagittal_image,
        one_hot_sag_label,
        output_dir,
        "spine_coronal.png",
        centroids,
        spine_hus=spine_hus,
        model_type=model_type,
        pixel_spacing=pixel_spacing,
        levels=label_text
    )


def curved_planar_reformation(mvs, centroids):
    centroids = sorted(centroids, key=lambda x: x[2])
    centroids = [(int(x[0]), int(x[1]), int(x[2])) for x in centroids]
    sagittal_centroids = [centroids[i][1] for i in range(0, len(centroids))]
    coronal_centroids = [centroids[i][0] for i in range(0, len(centroids))]
    axial_centroids = [centroids[i][2] for i in range(0, len(centroids))]
    sagittal_vals = [sagittal_centroids[0]] * axial_centroids[0]
    coronal_vals = [coronal_centroids[0]] * axial_centroids[0]

    for i in range(1, len(axial_centroids)):
        num = axial_centroids[i] - axial_centroids[i - 1]
        interp = list(np.linspace(sagittal_centroids[i - 1], sagittal_centroids[i], num=num))
        sagittal_vals.extend(interp)
        interp = list(np.linspace(coronal_centroids[i - 1], coronal_centroids[i], num=num))
        coronal_vals.extend(interp)

    sagittal_vals.extend([sagittal_centroids[-1]] * (mvs.shape[2] - len(sagittal_vals)))
    coronal_vals.extend([coronal_centroids[-1]] * (mvs.shape[2] - len(coronal_vals)))
    sagittal_vals = np.array(sagittal_vals)
    coronal_vals = np.array(coronal_vals)
    sagittal_vals = sagittal_vals.astype(int)
    coronal_vals = coronal_vals.astype(int)

    return (sagittal_vals, coronal_vals)


'''
def compare_ts_stanford_centroids(labels_path, pred_centroids):
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
    for label_path in labels:
        # modify label_path to give pred_path
        pred_path = label_path.replace("labelsTs", "predTs_TS")
        print(label_path.split("/")[-1])
        label_nib = nib.load(label_path)
        label = label_nib.get_fdata()
        spacing = label_nib.header.get_zooms()[2]
        pred_nib = nib.load(pred_path)
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
            sums = np.sum(label_level, axis=(0, 1))
            normalized_sums = sums / np.sum(sums)
            label_centroid = np.sum(indexes * normalized_sums)
            print(f"Centroid for label {label_idx}: {label_centroid}")

            if False:
                try:
                    pred_centroid = pred_centroids[6 - label_idx]
                except Exception:
                    # Change this part
                    print("Something wrong with pred_centroids, skipping!")
                    num_skipped += 1
                    break

            # if revert_to_original:
            if True:
                pred_level = pred == label_idx
                sums = np.sum(pred_level, axis=(0, 1))
                indices = list(range(sums.shape[0]))
                groupby_input = zip(indices, list(sums))
                g = groupby(groupby_input, key=lambda x: x[1] > 0.0)
                m = max([list(s) for v, s in g if v > 0], key=lambda x: np.sum(list(zip(*x))[1]))
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

    print(f"Skipped {num_skipped}")
    print("The final mean differences in mm:")
    print(
        np.mean(t12_diff),
        np.mean(l1_diff),
        np.mean(l2_diff),
        np.mean(l3_diff),
        np.mean(l4_diff),
        np.mean(l5_diff),
    )
    print("The final median differences in mm:")
    print(
        np.median(t12_diff),
        np.median(l1_diff),
        np.median(l2_diff),
        np.median(l3_diff),
        np.median(l4_diff),
        np.median(l5_diff),
    )


def compare_ts_stanford_roi_hus(image_path):
    """Compare the HU values of the Stanford dataset with the HU values of the TS dataset.

    image_path (str): Path to the Stanford dataset images.
    """
    img_paths = glob(image_path + "/*")
    differences = np.zeros((40, 6))
    ground_truth = np.zeros((40, 6))
    for i, img_path in enumerate(img_paths):
        print(f"Image number {i + 1}")
        image_path_no_0000 = re.sub(r"_0000", "", img_path)
        ts_seg_path = image_path_no_0000.replace("imagesTs", "predTs_TS")
        stanford_seg_path = image_path_no_0000.replace("imagesTs", "labelsTs")
        img = nib.load(img_path).get_fdata()
        img = np.swapaxes(img, 0, 1)
        ts_seg = nib.load(ts_seg_path).get_fdata()
        ts_seg = np.swapaxes(ts_seg, 0, 1)
        stanford_seg = nib.load(stanford_seg_path).get_fdata()
        stanford_seg = np.swapaxes(stanford_seg, 0, 1)
        ts_model_type = Models.model_from_name("ts_spine")
        (spine_hus_ts, rois, centroids_3d) = compute_rois(ts_seg, img, 1, 0, ts_model_type)
        stanford_model_type = Models.model_from_name("stanford_spine_v0.0.1")
        (spine_hus_stanford, rois, centroids_3d) = compute_rois(
            stanford_seg, img, 1, 0, stanford_model_type
        )
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
    median_difference = np.median(differences, axis=0)
    print("Average difference from ground truth:")
    print(average_difference)
    print("Median difference from ground truth:")
    print(median_difference)


def process_post_hoc(pred_path):
    """Apply post-hoc heuristics for improving Stanford spine model vertical centroid predictions.

    Args:
        pred_path (str): Path to the prediction.
    """
    pred_nib = nib.load(pred_path)
    pred = pred_nib.get_fdata()

    pred_bodies = np.logical_and(pred >= 1, pred <= 6)
    pred_bodies = pred_bodies.astype(np.int64)

    labels_out, N = cc3d.connected_components(pred_bodies, return_N=True, connectivity=6)

    stats = cc3d.statistics(labels_out)
    print(stats)

    labels_out_list = []
    voxel_counts_list = list(stats["voxel_counts"])
    for idx_lab in range(1, N + 2):
        labels_out_list.append(labels_out == idx_lab)

    centroids_list = list(stats["centroids"][:, 2])

    labels = []
    centroids = []
    voxels = []

    for idx, count in enumerate(voxel_counts_list):
        if count > 10000:
            labels.append(labels_out_list[idx])
            centroids.append(centroids_list[idx])
            voxels.append(count)

    top_comps = [
        (counts0, labels0, centroids0)
        for counts0, labels0, centroids0 in sorted(zip(voxels, labels, centroids), reverse=True)
    ]
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
                print(
                    "Post processing failure, probably < 6 predicted bodies. Reverting to original labels."
                )
                revert_to_original = True

        if revert_to_original:
            pred_level = pred == label_idx
            sums = np.sum(pred_level, axis=(0, 1))
            indices = list(range(sums.shape[0]))
            groupby_input = zip(indices, list(sums))
            # sys.exit()
            g = groupby(groupby_input, key=lambda x: x[1] > 0.0)
            m = max([list(s) for v, s in g if v > 0], key=lambda x: np.sum(list(zip(*x))[1]))
            # sys.exit()
            # m = max([list(s) for v, s in g], key=lambda np.sum)
            res = list(zip(*m))
            indexes = list(res[0])
            sums = list(res[1])
            normalized_sums = sums / np.sum(sums)
            pred_centroid = np.sum(indexes * normalized_sums)
        print(f"Centroid for prediction {label_idx}: {pred_centroid}")
'''
