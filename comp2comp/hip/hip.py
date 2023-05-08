"""
@author: louisblankemeier
"""

import os
from pathlib import Path
from time import time
from typing import Union

import pandas as pd
from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.hip import hip_utils
from comp2comp.hip.hip_visualization import hip_roi_visualizer
from comp2comp.inference_class_base import InferenceClass
from comp2comp.models.models import Models


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
        images_folder = os.path.join(inference_pipeline.output_dir, "dev")
        results_dict = hip_utils.compute_rois(medical_volume, segmentation, model, images_folder)
        inference_pipeline.femur_results_dict = results_dict
        return {}


class HipMetricsSaver(InferenceClass):
    """Save metrics to a CSV file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        metrics_output_dir = os.path.join(inference_pipeline.output_dir, "metrics")
        if not os.path.exists(metrics_output_dir):
            os.makedirs(metrics_output_dir)
        results_dict = inference_pipeline.femur_results_dict
        left_head_hu = results_dict["left_head"]["hu"]
        right_head_hu = results_dict["right_head"]["hu"]
        left_intertrochanter_hu = results_dict["left_intertrochanter"]["hu"]
        right_intertrochanter_hu = results_dict["right_intertrochanter"]["hu"]
        left_neck_hu = results_dict["left_neck"]["hu"]
        right_neck_hu = results_dict["right_neck"]["hu"]
        # save to csv
        df = pd.DataFrame(
            {
                "Left Head (HU)": [left_head_hu],
                "Right Head (HU)": [right_head_hu],
                "Left Intertrochanter (HU)": [left_intertrochanter_hu],
                "Right Intertrochanter (HU)": [right_intertrochanter_hu],
                "Left Neck (HU)": [left_neck_hu],
                "Right Neck (HU)": [right_neck_hu],
            }
        )
        df.to_csv(os.path.join(metrics_output_dir, "hip_metrics.csv"), index=False)
        return {}


class HipVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        medical_volume = inference_pipeline.medical_volume

        left_head_roi = inference_pipeline.femur_results_dict["left_head"]["roi"]
        left_head_centroid = inference_pipeline.femur_results_dict["left_head"]["centroid"]
        left_head_hu = inference_pipeline.femur_results_dict["left_head"]["hu"]

        left_intertrochanter_roi = inference_pipeline.femur_results_dict["left_intertrochanter"][
            "roi"
        ]
        left_intertrochanter_centroid = inference_pipeline.femur_results_dict[
            "left_intertrochanter"
        ]["centroid"]
        left_intertrochanter_hu = inference_pipeline.femur_results_dict["left_intertrochanter"][
            "hu"
        ]

        left_neck_roi = inference_pipeline.femur_results_dict["left_neck"]["roi"]
        left_neck_centroid = inference_pipeline.femur_results_dict["left_neck"]["centroid"]
        left_neck_hu = inference_pipeline.femur_results_dict["left_neck"]["hu"]

        right_head_roi = inference_pipeline.femur_results_dict["right_head"]["roi"]
        right_head_centroid = inference_pipeline.femur_results_dict["right_head"]["centroid"]
        right_head_hu = inference_pipeline.femur_results_dict["right_head"]["hu"]

        right_intertrochanter_roi = inference_pipeline.femur_results_dict["right_intertrochanter"][
            "roi"
        ]
        right_intertrochanter_centroid = inference_pipeline.femur_results_dict[
            "right_intertrochanter"
        ]["centroid"]
        right_intertrochanter_hu = inference_pipeline.femur_results_dict["right_intertrochanter"][
            "hu"
        ]

        right_neck_roi = inference_pipeline.femur_results_dict["right_neck"]["roi"]
        right_neck_centroid = inference_pipeline.femur_results_dict["right_neck"]["centroid"]
        right_neck_hu = inference_pipeline.femur_results_dict["right_neck"]["hu"]

        output_dir = inference_pipeline.output_dir
        images_output_dir = os.path.join(output_dir, "images")
        if not os.path.exists(images_output_dir):
            os.makedirs(images_output_dir)
        hip_roi_visualizer(
            medical_volume,
            left_head_roi,
            left_head_centroid,
            left_head_hu,
            images_output_dir,
            "left_head",
        )
        hip_roi_visualizer(
            medical_volume,
            left_intertrochanter_roi,
            left_intertrochanter_centroid,
            left_intertrochanter_hu,
            images_output_dir,
            "left_intertrochanter",
        )
        hip_roi_visualizer(
            medical_volume,
            left_neck_roi,
            left_neck_centroid,
            left_neck_hu,
            images_output_dir,
            "left_neck",
        )
        hip_roi_visualizer(
            medical_volume,
            right_head_roi,
            right_head_centroid,
            right_head_hu,
            images_output_dir,
            "right_head",
        )
        hip_roi_visualizer(
            medical_volume,
            right_intertrochanter_roi,
            right_intertrochanter_centroid,
            right_intertrochanter_hu,
            images_output_dir,
            "right_intertrochanter",
        )
        hip_roi_visualizer(
            medical_volume,
            right_neck_roi,
            right_neck_centroid,
            right_neck_hu,
            images_output_dir,
            "right_neck",
        )
        return {}
