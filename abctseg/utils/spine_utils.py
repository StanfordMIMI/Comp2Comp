import numpy as np
from pydicom.filereader import read_file_meta_info, dcmread
from glob import glob

from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences

def find_spine_dicoms(seg):
    vertical_positions = []
    for label_idx in range(18, 24):
        pos = compute_centroid("axial", label_idx)
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

    return (dicom_files, label_text)

def compute_centroid(plane: str, label: int):
    if plane == "axial":
        sum_out_axes = (0, 1)
        sum_axis = 2

    elif plane == "coronal":
        sum_out_axes = (0, 2)
        sum_axis = 1

    elif plane == "sagittal":
        sum_out_axes = (1, 2)
        sum_axis = 0

    sums = np.sum(seg == label, axis = sum_axes)
    normalized_sums = sums / np.sum(sums)
    pos = int(np.sum(np.arange(0, seg.shape[sum_axis]) * normalized_sums))
    return pos