import os

import nibabel as nib


def find_dicom_files(input_path):
    dicom_series = []
    if not os.path.isdir(input_path):
        dicom_series = [str(os.path.abspath(input_path))]
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".dcm") or file.endswith(".dicom"):
                    dicom_series.append(os.path.join(root, file))
    return dicom_series


def get_dicom_paths_and_num(path):
    """
    Get all paths under a path that contain only dicom files.
    Args:
        path (str): Path to search.
    Returns:
        list: List of paths.
    """
    dicom_paths = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            if all(file.endswith(".dcm") or file.endswith(".dicom") for file in files):
                dicom_paths.append((root, len(files)))

    if len(dicom_paths) == 0:
        raise ValueError("No scans were found in:\n" + path)

    return dicom_paths


def get_dicom_or_nifti_paths_and_num(path):
    """Get all paths under a path that contain only dicom files or a nifti file.
    Args:
        path (str): Path to search.

    Returns:
        list: List of paths.
    """
    # if path is a nifti
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        num_slices = nib.load(path).shape[2]
        return [(path, num_slices)]

    dicom_nifti_paths = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            if all(file.endswith(".dcm") or file.endswith(".dicom") for file in files):
                dicom_nifti_paths.append((root, len(files)))
            else:
                for file in files:
                    if file.endswith(".nii") or file.endswith(".nii.gz"):
                        num_slices = nib.load(path).shape[2]
                        dicom_nifti_paths.append((os.path.join(root, file), num_slices))

    return dicom_nifti_paths
