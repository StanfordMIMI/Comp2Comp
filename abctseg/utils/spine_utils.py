import numpy as np
from pydicom.filereader import read_file_meta_info, dcmread
from glob import glob

from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences

def find_spine_dicoms(seg):
    vertical_positions = []
    for label_idx in range(18, 24):
        pos = compute_centroid(seg, "axial", label_idx)
        vertical_positions.append(pos)

    print(f"Instance numbers: {vertical_positions}")

    folder_in = PREFERENCES.INPUT_DIR
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

def compute_centroid(seg, plane, label):
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

def to_one_hot(label):
    one_hot_label = np.zeros((label.shape[0], label.shape[1], 7))
    one_hot_label[:, :, 1] = (label == 18).astype(int)
    one_hot_label[:, :, 2] = (label == 19).astype(int)
    one_hot_label[:, :, 3] = (label == 20).astype(int)
    one_hot_label[:, :, 4] = (label == 21).astype(int)
    one_hot_label[:, :, 5] = (label == 22).astype(int)
    one_hot_label[:, :, 6] = (label == 23).astype(int)
    return one_hot_label