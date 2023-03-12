import dosma as dm
from pathlib import Path
from typing import Dict, Union
from typing import Type

from comp2comp.inference_class_base import InferenceClass
#from comp2comp.inference_pipeline import InferencePipeline

class DicomLoader(InferenceClass):
    """Load a single dicom series.
    """
    def __init__(self, input_path: Union[str, Path]):
        super().__init__()
        self.dicom_dir = Path(input_path)
        self.dr = dm.DicomReader()

    def __call__(self, inference_pipeline) -> Dict:
        medical_volume = self.dr.load(self.dicom_dir, group_by=None, sort_by="InstanceNumber")[0]
        return {"medical_volume": medical_volume}

class NiftiSaver(InferenceClass):
    """Save dosma medical volume object to NIfTI file.
    """
    def __init__(self):
        super().__init__()
        #self.output_dir = Path(output_path)
        self.nw = dm.NiftiWriter()

    def __call__(self, inference_pipeline, medical_volume: dm.MedicalVolume) -> Dict[str, Path]:
        nifti_file = inference_pipeline.output_dir
        self.nw.write(medical_volume, nifti_file)
        return {"nifti_file": nifti_file}

class DicomFinder(InferenceClass):
    """Find dicom files in a directory.
    """
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
        return {"dicom_file_paths": dicom_files}

    