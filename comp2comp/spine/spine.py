import os
from pathlib import Path
from time import time
from typing import Union

import dosma
import numpy as np
from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.inference_class_base import InferenceClass
from comp2comp.models.models import Models
from comp2comp.spine import spine_utils


class SpineSegmentation(InferenceClass):
    """Spine segmentation."""

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path

    def __call__(self, inference_pipeline):
        inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        seg, mv = self.spine_seg(self.input_path, self.output_dir_segmentations + "spine.nii.gz")
        return {"segmentation": seg, "medical_volume": mv}

    def spine_seg(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ):
        """Run spine segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting spine...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        task_id = [252]
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None

        setup_nnunet()
        download_pretrained_weights(task_id[0])

        from totalsegmentator.nnunet import nnUNet_predict_image

        with nostdout():

            seg, mvs = nnUNet_predict_image(
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
        print(f"Total time for spine segmentation: {end-st:.2f}s.")

        return seg, mvs


class SpineReorient(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, segmentation, medical_volume):
        mv_ndarray = medical_volume.volume
        pixel_spacing = medical_volume.pixel_spacing

        # Flip / transpose the axes if necessary
        transpose_idxs = dosma.core.get_transpose_inds(
            medical_volume.orientation, ("AP", "RL", "SI")
        )
        mv_ndarray = np.transpose(mv_ndarray, transpose_idxs)
        seg = np.transpose(segmentation, transpose_idxs)
        # apply same transformation to pixel spacing
        pixel_spacing_list = []
        for i in range(3):
            pixel_spacing_list.append(pixel_spacing[transpose_idxs[i]])

        flip_idxs = dosma.core.get_flip_inds(medical_volume.orientation, ("AP", "RL", "SI"))
        mv_ndarray = np.flip(mv_ndarray, flip_idxs)
        seg = np.flip(seg, flip_idxs)

        flip_si = False
        if 2 in flip_idxs:
            flip_si = True

        inference_pipeline.segmentation = seg
        inference_pipeline.medical_volume = medical_volume
        inference_pipeline.mv_ndarray = mv_ndarray
        inference_pipeline.pixel_spacing_list = pixel_spacing_list
        inference_pipeline.flip_si = flip_si

        return {}


class SpineComputeROIs(InferenceClass):
    def __init__(self, spine_model):
        super().__init__()
        self.spine_model_name = spine_model
        self.spine_model_type = Models.model_from_name(self.spine_model_name)

    def __call__(self, inference_pipeline):
        # Compute ROIs
        inference_pipeline.spine_model_type = self.spine_model_type

        (spine_hus, rois, centroids_3d) = spine_utils.compute_rois(
            inference_pipeline.segmentation,
            inference_pipeline.mv_ndarray,
            inference_pipeline.medical_volume.get_metadata("RescaleSlope"),
            inference_pipeline.medical_volume.get_metadata("RescaleIntercept"),
            self.spine_model_type,
            inference_pipeline.pixel_spacing_list,
        )

        spine_hus = spine_hus[::-1]

        inference_pipeline.spine_hus = spine_hus
        inference_pipeline.rois = rois
        inference_pipeline.centroids_3d = centroids_3d

        return {}


class SpineFindDicoms(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        dicom_files, names, centroids = spine_utils.find_spine_dicoms(
            inference_pipeline.segmentation,
            inference_pipeline.dicom_series_path,
            inference_pipeline.spine_model_type,
            inference_pipeline.flip_si,
        )

        inference_pipeline.dicom_files = dicom_files
        inference_pipeline.names = names
        inference_pipeline.dicom_file_names = names
        inference_pipeline.centroids = centroids

        return {}


class SpineCoronalSagittalVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        output_path = inference_pipeline.output_dir
        spine_model_type = inference_pipeline.spine_model_type

        spine_utils.visualize_coronal_sagittal_spine(
            inference_pipeline.segmentation,
            inference_pipeline.rois,
            inference_pipeline.mv_ndarray,
            inference_pipeline.centroids,
            inference_pipeline.centroids_3d,
            inference_pipeline.names,
            output_path,
            spine_hus=inference_pipeline.spine_hus,
            model_type=spine_model_type,
            pixel_spacing=inference_pipeline.pixel_spacing_list,
        )

        dicom_files = inference_pipeline.dicom_files
        # convert to list of paths
        dicom_files = [Path(d) for d in dicom_files]
        inference_pipeline.spine = True

        return {"dicom_file_paths": dicom_files}
