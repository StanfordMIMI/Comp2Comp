import math
import os
import sys
import zipfile
from pathlib import Path
from time import time
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.hip import hip_utils
from comp2comp.inference_class_base import InferenceClass
from comp2comp.models.models import Models
from comp2comp.hip.hip_visualization import hip_roi_visualizer


class HipSegmentation(InferenceClass):
    """Spine segmentation."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = Models.model_from_name(model_name)

    def __call__(self, inference_pipeline):
        # inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        seg, mv = self.hip_seg(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            self.output_dir_segmentations + "hip.nii.gz",
            inference_pipeline.model_dir,
        )

        inference_pipeline.model = self.model
        inference_pipeline.segmentation = seg
        inference_pipeline.medical_volume = mv

        return {}

    def hip_seg(self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir):
        """Run spine segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting hip...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [254]

        if self.model_name == "ts_hip":
            setup_nnunet()
            download_pretrained_weights(task_id[0])
        else:
            raise ValueError("Invalid model name.")

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

        # Log total time for hip segmentation
        print(f"Total time for hip segmentation: {end-st:.2f}s.")

        return seg, img


class HipComputeROIs(InferenceClass):
    def __init__(self, hip_model):
        super().__init__()
        self.hip_model_name = hip_model
        self.hip_model_type = Models.model_from_name(self.hip_model_name)

    def __call__(self, inference_pipeline):
        segmentation = inference_pipeline.segmentation
        medical_volume = inference_pipeline.medical_volume

        model = inference_pipeline.model
        images_folder = os.path.join(inference_pipeline.output_dir, "images")
        if not os.path.exists(images_folder):
            os.makedirs(images_folder)
        results_dict = hip_utils.compute_rois(medical_volume, segmentation, model, images_folder)
        inference_pipeline.femur_results_dict = results_dict
        return {}   


class HipMetricsSaver(InferenceClass):
    """Save metrics to a CSV file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        return {}


class HipVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        medical_volume = inference_pipeline.medical_volume
        left_femural_head_roi = inference_pipeline.femur_results_dict["left"]["roi"]
        left_femural_head_centroid = inference_pipeline.femur_results_dict["left"]["centroid"]
        right_femural_head_roi = inference_pipeline.femur_results_dict["right"]["roi"]
        right_femural_head_centroid = inference_pipeline.femur_results_dict["right"]["centroid"]
        output_dir = inference_pipeline.output_dir
        images_output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(images_output_dir):
            os.makedirs(images_output_dir)
        hip_roi_visualizer(medical_volume, left_femural_head_roi, left_femural_head_centroid, images_output_dir, "left")
        hip_roi_visualizer(medical_volume, right_femural_head_roi, right_femural_head_centroid, images_output_dir, "right")

