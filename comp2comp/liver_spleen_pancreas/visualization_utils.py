#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import ListedColormap
from PIL import Image


def extract_axial_mid_slice(ct, mask, crop=True):
    # find the slice with max surface area  of the organ
    axial_extent = np.where(mask.sum(axis=(0, 1)))[0]

    max_extent = 0
    max_extent_idx = 0

    for idx in axial_extent:
        label, num_features = scipy.ndimage.label(mask[:, :, idx])

        if num_features > 1:
            continue
        else:
            extent = label.sum()
            if extent > max_extent:
                max_extent = extent
                max_extent_idx = idx

    ct_slice_z = np.transpose(ct[:, :, max_extent_idx], axes=(1, 0))
    mask_slice_z = np.transpose(mask[:, :, max_extent_idx], axes=(1, 0))

    ct_slice_z = np.flip(ct_slice_z, axis=(0,1))
    mask_slice_z = np.flip(mask_slice_z, axis=(0,1))

    return ct_slice_z, mask_slice_z, max_extent_idx

# def extract_axial_mid_slice(ct, mask, crop=True):
#     # find the slice with max surface area  of the organ
#     slice_idx = np.argmax(mask.sum(axis=(0, 1)))

#     ct_slice_z = np.transpose(ct[:, :, slice_idx], axes=(1, 0))
#     mask_slice_z = np.transpose(mask[:, :, slice_idx], axes=(1, 0))

#     ct_slice_z = np.flip(ct_slice_z, axis=(0, 1))
#     mask_slice_z = np.flip(mask_slice_z, axis=(0, 1))

#     if crop:
#         ct_range_x = np.where(ct_slice_z.max(axis=0) > -200)[0][[0, -1]]

#         ct_slice_z = ct_slice_z[
#             ct_range_x[0] : ct_range_x[1], ct_range_x[0] : ct_range_x[1]
#         ]
#         mask_slice_z = mask_slice_z[
#             ct_range_x[0] : ct_range_x[1], ct_range_x[0] : ct_range_x[1]
#         ]

#     return ct_slice_z, mask_slice_z


def extract_coronal_mid_slice(ct, mask, crop=True):
    # find the slice with max coherent extent of the organ
    coronary_extent = np.where(mask.sum(axis=(0, 2)))[0]

    max_extent = 0
    max_extent_idx = 0

    for idx in coronary_extent:
        label, num_features = scipy.ndimage.label(mask[:, idx, :])

        if num_features > 1:
            continue
        else:
            extent = len(np.where(label.sum(axis=1))[0])
            if extent > max_extent:
                max_extent = extent
                max_extent_idx = idx

    ct_slice_y = np.transpose(ct[:, max_extent_idx, :], axes=(1, 0))
    mask_slice_y = np.transpose(mask[:, max_extent_idx, :], axes=(1, 0))

    ct_slice_y = np.flip(ct_slice_y, axis=1)
    mask_slice_y = np.flip(mask_slice_y, axis=1)

    return ct_slice_y, mask_slice_y


def save_slice(
    ct_slice,
    mask_slice,
    path,
    figsize=(12, 12),
    corner_text=None,
    unit_dict=None,
    aspect=1,
    show=False,
    xy_placement=None,
    class_color=1,
    fontsize=14,
):
    # colormap for shown segmentations
    color_array = plt.get_cmap("tab10")(range(10))
    color_array = np.concatenate((np.array([[0, 0, 0, 0]]), color_array[:, :]), axis=0)
    map_object_seg = ListedColormap(name="segmentation_cmap", colors=color_array)

    fig, axx = plt.subplots(1, figsize=figsize, frameon=False)
    axx.imshow(
        ct_slice,
        cmap="gray",
        vmin=-400,
        vmax=400,
        interpolation="spline36",
        aspect=aspect,
        origin="lower",
    )
    axx.imshow(
        mask_slice * class_color,
        cmap=map_object_seg,
        vmin=0,
        vmax=9,
        alpha=0.2,
        interpolation="nearest",
        aspect=aspect,
        origin="lower",
    )

    plt.axis("off")
    axx.axes.get_xaxis().set_visible(False)
    axx.axes.get_yaxis().set_visible(False)

    y_size, x_size = ct_slice.shape

    if corner_text is not None:
        bbox_props = dict(boxstyle="round", facecolor="gray", alpha=0.5)

        texts = []
        for k, v in corner_text.items():
            if isinstance(v, str):
                texts.append("{:<9}{}".format(k + ":", v))
            else:
                unit = unit_dict[k] if k in unit_dict else ""
                texts.append("{:<9}{:.0f} {}".format(k + ":", v, unit))

        if xy_placement is None:
            # get the extent of textbox, remove, and the plot again with correct position
            t = axx.text(
                0.5,
                0.5,
                "\n".join(texts),
                color="white",
                transform=axx.transAxes,
                fontsize=fontsize,
                family="monospace",
                bbox=bbox_props,
                va="top",
                ha="left",
            )
            xmin, xmax = t.get_window_extent().xmin, t.get_window_extent().xmax
            xmin, xmax = axx.transAxes.inverted().transform((xmin, xmax))

            xy_placement = [1 - (xmax - xmin) - (xmax - xmin) * 0.09, 0.975]
            t.remove()

        axx.text(
            xy_placement[0],
            xy_placement[1],
            "\n".join(texts),
            color="white",
            transform=axx.transAxes,
            fontsize=fontsize,
            family="monospace",
            bbox=bbox_props,
            va="top",
            ha="left",
        )

    if show:
        plt.show()
    else:
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def slicedDilationOrErosion(input_mask, num_iteration, operation):
    """
    Perform the dilation on the smallest slice that will fit the
    segmentation
    """
    
    # if empty, don't do dilation
    if input_mask.sum() == 0:
        return input_mask
    
    margin = 2 if num_iteration is None else num_iteration + 1
    
    x_size, y_size, z_size = input_mask.shape

    # find the minimum volume enclosing the organ
    x_idx = np.where(input_mask.sum(axis=(1, 2)))[0]
    x_start, x_end = max(0, x_idx[0] - margin), min(x_idx[-1] + margin, x_size)
                                                   
    y_idx = np.where(input_mask.sum(axis=(0, 2)))[0]
    y_start, y_end = max(0, y_idx[0] - margin), min(y_idx[-1] + margin, y_size)
    
    z_idx = np.where(input_mask.sum(axis=(0, 1)))[0]
    z_start, z_end = max(0, z_idx[0] - margin), min(z_idx[-1] + margin, z_size)
    
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    struct = scipy.ndimage.iterate_structure(struct, num_iteration)

    if operation == "dilate":
        mask_slice = scipy.ndimage.binary_dilation(
            input_mask[x_start:x_end, y_start:y_end, z_start:z_end], structure=struct
        ).astype(np.int8)
    elif operation == "erode":
        mask_slice = scipy.ndimage.binary_erosion(
            input_mask[x_start:x_end, y_start:y_end, z_start:z_end], structure=struct
        ).astype(np.int8)

    output_mask = input_mask.copy()

    output_mask[x_start:x_end, y_start:y_end, z_start:z_end] = mask_slice

    return output_mask


def extract_organ_metrics(
    ct, all_masks, class_num=None, vol_per_pixel=None, erode_mask=True
):
    if erode_mask:
        eroded_mask = slicedDilationOrErosion(
            input_mask=(all_masks == class_num), num_iteration=3, operation="erode"
        )
        ct_organ_vals = ct[eroded_mask == 1]
    else:
        ct_organ_vals = ct[all_masks == class_num]

    results = {}

    # in ml
    organ_vol = (all_masks == class_num).sum() * vol_per_pixel
    organ_mean = ct_organ_vals.mean()
    organ_median = np.median(ct_organ_vals)

    results = {
        "Organ": class_map_part_organs[class_num],
        "Volume": organ_vol,
        "Mean": organ_mean,
        "Median": organ_median,
    }

    return results


def generate_slice_images(
    ct,
    all_masks,
    class_nums,
    unit_dict,
    vol_per_pixel,
    pix_dims,
    root,
    fontsize=20,
    show=False,
):
    all_results = {}

    colors = [1, 3, 4]

    # create the txt files for the slices idx
    with open(os.path.join(root, 'slice_idx.csv'), 'w') as f:
        f.write('organ,mean_HU,axial_idx\n')
    
    for i, c_num in enumerate(class_nums):
        organ_name = class_map_part_organs[c_num]

        axial_path = os.path.join(root, organ_name.lower() + "_axial.png")
        coronal_path = os.path.join(root, organ_name.lower() + "_coronal.png")

        ct_slice_z, liver_slice_z, slice_idx_z = extract_axial_mid_slice(ct, all_masks == c_num)
        with open(os.path.join(root, 'slice_idx.csv'), 'a') as f:
            mean_hu = ct_slice_z[liver_slice_z == 1].mean()
            f.write(organ_name.lower()+ ',' + '{:.1f}'.format(mean_hu) + ',' + str(slice_idx_z) + '\n')
    
        results = extract_organ_metrics(
            ct, all_masks, class_num=c_num, vol_per_pixel=vol_per_pixel
        )

        save_slice(
            ct_slice_z,
            liver_slice_z,
            axial_path,
            figsize=(12, 12),
            corner_text=results,
            unit_dict=unit_dict,
            class_color=colors[i],
            fontsize=fontsize,
            show=show,
        )

        ct_slice_y, liver_slice_y = extract_coronal_mid_slice(ct, all_masks == c_num)

        save_slice(
            ct_slice_y,
            liver_slice_y,
            coronal_path,
            figsize=(12, 12),
            aspect=pix_dims[2] / pix_dims[1],
            show=show,
            class_color=colors[i],
        )

        all_results[results["Organ"]] = results

    if show:
        return

    return all_results


def generate_liver_spleen_pancreas_report(root, organ_names):
    
    axial_imgs = [
        Image.open(os.path.join(root, organ + "_axial.png")) for organ in organ_names
    ]
    coronal_imgs = [
        Image.open(os.path.join(root, organ + "_coronal.png")) for organ in organ_names
    ]

    result_width = max(
        sum([img.size[0] for img in axial_imgs]),
        sum([img.size[0] for img in coronal_imgs]),
    )
    result_height = max(
        [a.size[1] + c.size[1] for a, c in zip(axial_imgs, coronal_imgs)]
    )

    result = Image.new("RGB", (result_width, result_height))

    total_width = 0

    for a_img, c_img in zip(axial_imgs, coronal_imgs):
        a_width, a_height = a_img.size
        c_width, c_height = c_img.size

        translate = (a_width - c_width) // 2 if a_width > c_width else 0

        result.paste(im=a_img, box=(total_width, 0))
        result.paste(im=c_img, box=(translate + total_width, a_height))

        total_width += a_width

    result.save(os.path.join(root, "liver_spleen_pancreas_report.png"))


# from https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py

class_map_part_organs = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right",
        25: "sacrum",
        26: "vertebrae_S1",
        27: "vertebrae_L5",
        28: "vertebrae_L4",
        29: "vertebrae_L3",
        30: "vertebrae_L2",
        31: "vertebrae_L1",
        32: "vertebrae_T12",
        33: "vertebrae_T11",
        34: "vertebrae_T10",
        35: "vertebrae_T9",
        36: "vertebrae_T8",
        37: "vertebrae_T7",
        38: "vertebrae_T6",
        39: "vertebrae_T5",
        40: "vertebrae_T4",
        41: "vertebrae_T3",
        42: "vertebrae_T2",
        43: "vertebrae_T1",
        44: "vertebrae_C7",
        45: "vertebrae_C6",
        46: "vertebrae_C5",
        47: "vertebrae_C4",
        48: "vertebrae_C3",
        49: "vertebrae_C2",
        50: "vertebrae_C1",
        51: "heart",
        52: "aorta",
        53: "pulmonary_vein",
        54: "brachiocephalic_trunk",
        55: "subclavian_artery_right",
        56: "subclavian_artery_left",
        57: "common_carotid_artery_right",
        58: "common_carotid_artery_left",
        59: "brachiocephalic_vein_left",
        60: "brachiocephalic_vein_right",
        61: "atrial_appendage_left",
        62: "superior_vena_cava",
        63: "inferior_vena_cava",
        64: "portal_vein_and_splenic_vein",
        65: "iliac_artery_left",
        66: "iliac_artery_right",
        67: "iliac_vena_left",
        68: "iliac_vena_right",
        69: "humerus_left",
        70: "humerus_right",
        71: "scapula_left",
        72: "scapula_right",
        73: "clavicula_left",
        74: "clavicula_right",
        75: "femur_left",
        76: "femur_right",
        77: "hip_left",
        78: "hip_right",
        79: "spinal_cord",
        80: "gluteus_maximus_left",
        81: "gluteus_maximus_right",
        82: "gluteus_medius_left",
        83: "gluteus_medius_right",
        84: "gluteus_minimus_left",
        85: "gluteus_minimus_right",
        86: "autochthon_left",
        87: "autochthon_right",
        88: "iliopsoas_left",
        89: "iliopsoas_right",
        90: "brain",
        91: "skull",
        92: "rib_left_1",
        93: "rib_left_2",
        94: "rib_left_3",
        95: "rib_left_4",
        96: "rib_left_5",
        97: "rib_left_6",
        98: "rib_left_7",
        99: "rib_left_8",
        100: "rib_left_9",
        101: "rib_left_10",
        102: "rib_left_11",
        103: "rib_left_12",
        104: "rib_right_1",
        105: "rib_right_2",
        106: "rib_right_3",
        107: "rib_right_4",
        108: "rib_right_5",
        109: "rib_right_6",
        110: "rib_right_7",
        111: "rib_right_8",
        112: "rib_right_9",
        113: "rib_right_10",
        114: "rib_right_11",
        115: "rib_right_12",
        116: "sternum",
        117: "costal_cartilages"
    }
