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
