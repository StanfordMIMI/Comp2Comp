"""
@author: louisblankemeier
"""
import os
import shutil
from pathlib import Path
from typing import Dict, Union

# import dicom2nifti
import dosma as dm
import SimpleITK as sitk
import pydicom

from comp2comp.inference_class_base import InferenceClass


class DicomLoader(InferenceClass):
    """Load a single dicom series."""

    def __init__(self, input_path: Union[str, Path]):
        super().__init__()
        self.dicom_dir = Path(input_path)
        self.dr = dm.DicomReader()

    def __call__(self, inference_pipeline) -> Dict:
        medical_volume = self.dr.load(
            self.dicom_dir, group_by=None, sort_by="InstanceNumber"
        )[0]
        return {"medical_volume": medical_volume}


class NiftiSaver(InferenceClass):
    """Save dosma medical volume object to NIfTI file."""

    def __init__(self):
        super().__init__()
        # self.output_dir = Path(output_path)
        self.nw = dm.NiftiWriter()

    def __call__(
        self, inference_pipeline, medical_volume: dm.MedicalVolume
    ) -> Dict[str, Path]:
        nifti_file = inference_pipeline.output_dir
        self.nw.write(medical_volume, nifti_file)
        return {"nifti_file": nifti_file}


class DicomFinder(InferenceClass):
    """Find dicom files in a directory."""

    def __init__(self, input_path: Union[str, Path]) -> Dict[str, Path]:
        super().__init__()
        self.input_path = Path(input_path)

    def __call__(self, inference_pipeline) -> Dict[str, Path]:
        """Find dicom files in a directory.

        Args:
            inference_pipeline (InferencePipeline): Inference pipeline.

        Returns:
            Dict[str, Path]: Dictionary containing dicom files.
        """
        dicom_files = []
        for file in self.input_path.glob("**/*.dcm"):
            dicom_files.append(file)
        inference_pipeline.dicom_file_paths = dicom_files
        return {}


class DicomToNifti(InferenceClass):
    """Convert dicom files to NIfTI files."""

    def __init__(self, input_path: Union[str, Path], save=True):
        super().__init__()
        self.input_path = Path(input_path)
        self.save = save

    def __call__(self, inference_pipeline):
        if os.path.exists(
            os.path.join(
                inference_pipeline.output_dir, "segmentations", "converted_dcm.nii.gz"
            )
        ):
            return {}
        if hasattr(inference_pipeline, "medical_volume"):
            return {}
        output_dir = inference_pipeline.output_dir
        segmentations_output_dir = os.path.join(output_dir, "segmentations")
        os.makedirs(segmentations_output_dir, exist_ok=True)

        # if self.input_path is a folder
        if self.input_path.is_dir():
            dicom_series_to_nifti(
                self.input_path,
                output_file=os.path.join(
                    segmentations_output_dir, "converted_dcm.nii.gz"
                ),
                reorient_nifti=False,
            )
            inference_pipeline.dicom_series_path = str(self.input_path)
        elif str(self.input_path).endswith(".nii"):
            shutil.copy(
                self.input_path,
                os.path.join(segmentations_output_dir, "converted_dcm.nii"),
            )
        elif str(self.input_path).endswith(".nii.gz"):
            shutil.copy(
                self.input_path,
                os.path.join(segmentations_output_dir, "converted_dcm.nii.gz"),
            )

        return {}


def dicom_series_to_nifti(input_path, output_file, reorient_nifti):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(input_path))
    print("Reading dicoms...")
    ds = pydicom.filereader.dcmread(dicom_names[0])
    image_type_list = list(ds.ImageType)
    if any("gsi" in s.lower() for s in image_type_list):
        raise ValueError("GSI Image Type detected")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if image.GetDirection() != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        raise ValueError("Image orientation is not axial")
    print(ds)
    sitk.WriteImage(image, output_file)
