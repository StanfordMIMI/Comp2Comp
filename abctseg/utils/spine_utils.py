import numpy as np
from pydicom.filereader import read_file_meta_info, dcmread
from glob import glob

from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences

def find_spine_dicoms(seg):
    vertical_positions = []
    for label_idx in range(18, 24):
        sums = np.sum(seg == label_idx, axis = (0, 1))
        normalized_sums = sums / np.sum(sums)
        pos = int(np.sum(np.arange(0, seg.shape[2]) * normalized_sums))
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