import os
from pathlib import Path
from time import time
from typing import Union

from totalsegmentatorv2.python_api import totalsegmentator

from comp2comp.inference_class_base import InferenceClass


class LiverSpleenPancreasSegmentation(InferenceClass):
    """Organ segmentation."""

    def __init__(self):
        super().__init__()
        # self.input_path = input_path

    def __call__(self, inference_pipeline):
        # inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        seg = self.organ_seg(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            self.output_dir_segmentations + "organs.nii.gz",
            inference_pipeline.model_dir,
        )

        inference_pipeline.segmentation = seg

        return {}

    def organ_seg(
        self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir
    ):
        """Run organ segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting organs...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        seg = totalsegmentator(
            input=input_path,
            output=output_path,
            # input = os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            # output = os.path.join(self.output_dir_segmentations, "segmentation.nii"),
            task_ids=[291],
            ml=True,
            nr_thr_resamp=1,
            nr_thr_saving=6,
            fast=False,
            nora_tag="None",
            preview=False,
            task="total",
            # roi_subset = [
            #     "vertebrae_T12",
            #     "vertebrae_L1",
            #     "vertebrae_L2",
            #     "vertebrae_L3",
            #     "vertebrae_L4",
            #     "vertebrae_L5",
            # ],
            roi_subset=None,
            statistics=False,
            radiomics=False,
            crop_path=None,
            body_seg=False,
            force_split=False,
            output_type="nifti",
            quiet=False,
            verbose=False,
            test=0,
            skip_saving=True,
            device="gpu",
            license_number=None,
            statistics_exclude_masks_at_border=True,
            no_derived_masks=False,
            v1_order=False,
        )

        # Setup nnunet
        # model = "3d_fullres"
        # folds = [0]
        # trainer = "nnUNetTrainerV2_ep4000_nomirror"
        # crop_path = None
        # task_id = [251]

        # setup_nnunet()
        # download_pretrained_weights(task_id[0])

        # from totalsegmentator.nnunet import nnUNet_predict_image

        # with nostdout():
        #     seg, mvs = nnUNet_predict_image(
        #         input_path,
        #         output_path,
        #         task_id,
        #         model=model,
        #         folds=folds,
        #         trainer=trainer,
        #         tta=False,
        #         multilabel_image=True,
        #         resample=1.5,
        #         crop=None,
        #         crop_path=crop_path,
        #         task_name="total",
        #         nora_tag="None",
        #         preview=False,
        #         nr_threads_resampling=1,
        #         nr_threads_saving=6,
        #         quiet=False,
        #         verbose=True,
        #         test=0,
        #     )
        end = time()

        # Log total time for spine segmentation
        print(f"Total time for organ segmentation: {end-st:.2f}s.")

        return seg
