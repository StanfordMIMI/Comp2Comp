"""
@author: louisblankemeier
"""

import csv
import os

import nibabel as nib
import pydicom


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
    dicom_nifti_paths = []

    if path.endswith(".nii") or path.endswith(".nii.gz"):
        dicom_nifti_paths.append((path, getNumSlicesNifti(path)))
    elif path.endswith(".txt"):
        dicom_nifti_paths = []
        with open(path, "r") as f:
            for dicom_folder_path in f:
                dicom_folder_path = dicom_folder_path.strip()
                if dicom_folder_path.endswith(".nii") or dicom_folder_path.endswith(
                    ".nii.gz"
                ):
                    dicom_nifti_paths.append(
                        (dicom_folder_path, getNumSlicesNifti(dicom_folder_path))
                    )
                else:
                    dicom_nifti_paths.append(
                        (dicom_folder_path, len(os.listdir(dicom_folder_path)))
                    )
    else:
        for root, dirs, files in os.walk(path):
            if len(files) > 0:
                # if all(file.endswith(".dcm") or file.endswith(".dicom") for file in files):
                dicom_nifti_paths.append((root, len(files)))

    return dicom_nifti_paths


def write_dicom_metadata_to_csv(ds, csv_filename):
    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Tag", "Keyword", "Value"])

        for element in ds:
            tag = element.tag
            keyword = pydicom.datadict.keyword_for_tag(tag)
            if keyword == "PixelData":
                continue
            value = str(element.value)
            csvwriter.writerow([tag, keyword, value])


def getNumSlicesNifti(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.shape[2]
