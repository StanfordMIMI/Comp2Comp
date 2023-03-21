import os
import zipfile
from pathlib import Path
from time import time
from typing import Union

import nibabel as nib
import numpy as np
import wget
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

    def __init__(self, input_path, model_name):
        super().__init__()
        self.input_path = input_path
        self.model_name = model_name

    def __call__(self, inference_pipeline):
        inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        seg, mv = self.spine_seg(
            self.input_path,
            self.output_dir_segmentations + "spine.nii.gz",
            inference_pipeline.model_dir,
        )
        inference_pipeline.segmentation = seg
        inference_pipeline.medical_volume = mv
        return {}

    def setup_nnunet_c2c(self, model_dir: Union[str, Path]):
        """Adapted from TotalSegmentator."""

        model_dir = Path(model_dir)
        config_dir = model_dir / Path("." + self.model_name)
        (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(exist_ok=True, parents=True)
        (config_dir / "nnunet/results/nnUNet/2d").mkdir(exist_ok=True, parents=True)
        weights_dir = config_dir / "nnunet/results"
        self.weights_dir = weights_dir

        os.environ["nnUNet_raw_data_base"] = str(
            weights_dir
        )  # not needed, just needs to be an existing directory
        os.environ["nnUNet_preprocessed"] = str(
            weights_dir
        )  # not needed, just needs to be an existing directory
        os.environ["RESULTS_FOLDER"] = str(weights_dir)

    def download_spine_model(self, model_dir: Union[str, Path]):
        download_dir = Path(
            os.path.join(
                self.weights_dir,
                "nnUNet/3d_fullres/Task252_Spine/nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1",
            )
        )
        fold_0_path = download_dir / "fold_0"
        if not os.path.exists(fold_0_path):
            download_dir.mkdir(parents=True, exist_ok=True)
            wget.download(
                "https://huggingface.co/louisblankemeier/spine_v1/resolve/main/fold_0.zip",
                out=os.path.join(download_dir, "fold_0.zip"),
            )
            with zipfile.ZipFile(os.path.join(download_dir, "fold_0.zip"), "r") as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(os.path.join(download_dir, "fold_0.zip"))
            wget.download(
                "https://huggingface.co/louisblankemeier/spine_v1/resolve/main/plans.pkl",
                out=os.path.join(download_dir, "plans.pkl"),
            )
            print("Spine model downloaded.")
        else:
            print("Spine model already downloaded.")

    def spine_seg(self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir):
        """Run spine segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting spine...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [252]

        if self.model_name == "ts_spine":
            setup_nnunet()
            download_pretrained_weights(task_id[0])
        elif self.model_name == "stanford_spine_v0.0.1":
            self.setup_nnunet_c2c(model_dir)
            self.download_spine_model(model_dir)
        else:
            raise ValueError("Invalid model name.")

        from totalsegmentator.nnunet import nnUNet_predict_image

        # with nostdout():

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
        print(f"Total time for spine segmentation: {end-st:.2f}s.")

        if self.model_name == "stanford_spine_v0.0.1":
            seg_data = seg.get_fdata()
            # subtract 17 from seg values except for 0
            seg_data = np.where(seg_data == 0, 0, seg_data - 17)
            seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)

        return seg, img


class SpineReorient(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        inference_pipeline.flip_si = False  # necessary for finding dicoms in correct order
        if "I" in nib.aff2axcodes(inference_pipeline.medical_volume.affine):
            inference_pipeline.flip_si = True

        canonical_segmentation = nib.as_closest_canonical(inference_pipeline.segmentation)
        canonical_medical_volume = nib.as_closest_canonical(inference_pipeline.medical_volume)

        inference_pipeline.segmentation = canonical_segmentation
        inference_pipeline.medical_volume = canonical_medical_volume
        inference_pipeline.pixel_spacing_list = canonical_medical_volume.header.get_zooms()
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
            inference_pipeline.medical_volume,
            self.spine_model_type,
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
            inference_pipeline.segmentation.get_fdata(),
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
            inference_pipeline.segmentation.get_fdata(),
            inference_pipeline.rois,
            inference_pipeline.medical_volume.get_fdata(),
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
