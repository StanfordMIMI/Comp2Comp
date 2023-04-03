import os
import zipfile
from pathlib import Path
from time import time
from typing import Union

from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.contrast_phase.contrast_inf import predict_phase
from comp2comp.inference_class_base import InferenceClass
from comp2comp.models.models import Models
from comp2comp.spine import spine_utils


class ContrastPhaseDetection(InferenceClass):
    """Contrast Phase Detection."""

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def __call__(self, inference_pipeline):
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)
        self.model_dir = inference_pipeline.model_dir

        seg, img = self.run_segmentation(
            self.input_path,
            self.output_dir_segmentations + "s01.nii.gz",
            inference_pipeline.model_dir,
        )

        # segArray, imgArray = self.convertNibToNumpy(self, seg, img)

        imgNiftiPath = os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz")
        segNiftPath = os.path.join(self.output_dir_segmentations, "s01.nii.gz")

        predict_phase(segNiftPath, imgNiftiPath, outputPath=self.output_dir)

        return {}

    def run_segmentation(
        self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir
    ):
        """Run segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [251, 252, 253, 254, 255]

        setup_nnunet()
        for task_id in [251, 252, 253, 254, 255]:
            download_pretrained_weights(task_id)

        from totalsegmentator.nnunet import nnUNet_predict_image

        with nostdout():

            img, seg = nnUNet_predict_image(
                input_path,
                output_path,
                task_id,
                model=model,
                folds=folds,
                trainer=trainer,
                tta=False,
                multilabel_image=True,
                resample=1.5,
                crop=None,
                crop_path=crop_path,
                task_name="total",
                nora_tag=None,
                preview=False,
                nr_threads_resampling=1,
                nr_threads_saving=6,
                quiet=False,
                verbose=False,
                test=0,
            )
        end = time()

        # Log total time for spine segmentation
        print(f"Total time for segmentation: {end-st:.2f}s.")

        return seg, img

    def convertNibToNumpy(self, TSNib, ImageNib):
        """Convert nifti to numpy array.

        Args:
            TSNib (nibabel.nifti1.Nifti1Image): TotalSegmentator output.
            ImageNib (nibabel.nifti1.Nifti1Image): Input image.

        Returns:
            numpy.ndarray: TotalSegmentator output.
            numpy.ndarray: Input image.
        """
        TS_array = TSNib.get_fdata()
        img_array = ImageNib.get_fdata()
        return TS_array, img_array
