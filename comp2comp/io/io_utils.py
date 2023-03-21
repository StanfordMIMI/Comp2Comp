import os

def find_dicom_files(input_path):
    dicom_series = []
    if not os.path.isdir(input_path):
        dicom_series = [str(os.path.abspath(input_path))]
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".dcm"):
                    dicom_series.append(os.path.join(root, file))
    return dicom_series


def get_dicom_paths_and_num(path):
    """Get all paths under a path that contain only dicom files.

    Args:
        path (str): Path to search.

    Returns:
        list: List of paths.
    """
    dicom_paths = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            if all([file.endswith(".dcm") for file in files]):
                dicom_paths.append((root, len(files)))
    return dicom_paths

def get_dicom_nifti_paths_and_num(path):
    """Get all paths under a path that contain only dicom files or a nifti file.

    Args:
        path (str): Path to search.
    
    Returns:
        list: List of paths.
    """
    # if path is a nifti
    if path.endswith(".nii"):
        return [(path, 1)]
    dicom_nifti_paths = []
    for root, _, files in os.walk(path):
        if len(files) > 0:
            if all([file.endswith(".dcm") for file in files]):
                dicom_nifti_paths.append((root, len(files)))
            else:
                for file in files:
                    if file.endswith(".nii") or file.endswith(".nii.gz"):
                        dicom_nifti_paths.append((os.path.join(root, file), 1))
    return dicom_nifti_paths
