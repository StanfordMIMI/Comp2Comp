"""
@author: louisblankemeier
"""

import math
import os
import shutil
import zipfile
from pathlib import Path
from time import time
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd
import wget
from PIL import Image
from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)
from totalsegmentatorv2.python_api import totalsegmentator

from comp2comp.inference_class_base import InferenceClass
from comp2comp.models.models import Models
from comp2comp.spine import spine_utils
from comp2comp.visualization.dicom import to_dicom


class SpineSegmentation(InferenceClass):
    """Spine segmentation."""

    def __init__(self, model_name, save=True):
        super().__init__()
        self.model_name = model_name
        self.save_segmentations = save

    def __call__(self, inference_pipeline):
        # inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        inference_pipeline.output_dir_masks = os.path.join(self.output_dir, "masks/")

        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)
        if not os.path.exists(inference_pipeline.output_dir_masks):
            os.makedirs(inference_pipeline.output_dir_masks)

        self.model_dir = inference_pipeline.model_dir

        inference_pipeline.spine_model_name = self.model_name

        seg, mv = self.spine_seg(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            self.output_dir_segmentations + "spine.nii.gz",
            inference_pipeline.model_dir,
        )

        os.environ["TOTALSEG_WEIGHTS_PATH"] = self.model_dir

        mv = nib.load(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz")
        )

        # save the seg
        nib.save(
            seg,
            os.path.join(self.output_dir_segmentations, "spine_seg.nii.gz"),
            # os.path.join(inference_pipeline.output_dir_masks, "spine_seg.nii.gz"),
        )

        nib.save(
            mv,
            # os.path.join(self.output_dir_segmentations, "spine_seg.nii.gz"),
            os.path.join(inference_pipeline.output_dir_masks, "ct.nii.gz"),
        )

        # inference_pipeline.segmentation = nib.load(
        #     os.path.join(self.output_dir_segmentations, "segmentation.nii")
        # )
        inference_pipeline.segmentation = seg
        inference_pipeline.medical_volume = mv
        inference_pipeline.save_segmentations = self.save_segmentations

        return {}

    def setup_nnunet_c2c(self, model_dir: Union[str, Path]):
        """Adapted from TotalSegmentator."""

        model_dir = Path(model_dir)
        config_dir = model_dir / Path("." + self.model_name)
        (config_dir / "nnunet/results/nnUNet/3d_fullres").mkdir(
            exist_ok=True, parents=True
        )
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
            with zipfile.ZipFile(
                os.path.join(download_dir, "fold_0.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(os.path.join(download_dir, "fold_0.zip"))
            wget.download(
                "https://huggingface.co/louisblankemeier/spine_v1/resolve/main/plans.pkl",
                out=os.path.join(download_dir, "plans.pkl"),
            )
            print("Spine model downloaded.")
        else:
            print("Spine model already downloaded.")

    def spine_seg(
        self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir
    ):
        """Run spine segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting spine...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir
        os.environ["TOTALSEG_WEIGHTS_PATH"] = self.model_dir

        if self.model_name == "ts_spine":
            seg = totalsegmentator(
                input=os.path.join(
                    self.output_dir_segmentations, "converted_dcm.nii.gz"
                ),
                output=os.path.join(self.output_dir_segmentations, "segmentation.nii"),
                task_ids=[292],
                ml=True,
                nr_thr_resamp=1,
                nr_thr_saving=6,
                fast=False,
                nora_tag="None",
                preview=False,
                task="total",
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

            img = None

        elif self.model_name == "stanford_spine_v0.0.1":
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

            if not self.save_segmentations:
                output_path = None

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
                    nora_tag="None",
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


class AxialCropper(InferenceClass):
    """Crop the CT image (medical_volume) and segmentation based on user-specified
    lower and upper levels of the spine.
    """

    def __init__(self, lower_level: str = "L5", upper_level: str = "L1", save=True):
        """
        Args:
            lower_level (str, optional): Lower level of the spine. Defaults to "L5".
            upper_level (str, optional): Upper level of the spine. Defaults to "L1".
            save (bool, optional): Save cropped image and segmentation. Defaults to True.

        Raises:
            ValueError: If lower_level or upper_level is not a valid spine level.
        """
        super().__init__()
        self.lower_level = lower_level
        self.upper_level = upper_level
        ts_spine_full_model = Models.model_from_name("ts_spine")
        categories = ts_spine_full_model.categories
        try:
            self.lower_level_index = categories[self.lower_level]
            self.upper_level_index = categories[self.upper_level]
        except KeyError:
            raise ValueError("Invalid spine level.") from None
        self.save = save

    def __call__(self, inference_pipeline):
        """
        First dim goes from L to R.
        Second dim goes from P to A.
        Third dim goes from I to S.
        """
        segmentation = inference_pipeline.segmentation
        segmentation_data = segmentation.get_fdata()
        try:
            upper_level_index = np.where(segmentation_data == self.upper_level_index)[
                2
            ].max()
        except:
            upper_level_index = segmentation_data.shape[2]
        try:
            lower_level_index = np.where(segmentation_data == self.lower_level_index)[
                2
            ].min()
        except:
            lower_level_index = 0
        segmentation = segmentation.slicer[:, :, lower_level_index:upper_level_index]
        inference_pipeline.segmentation = segmentation

        medical_volume = inference_pipeline.medical_volume
        medical_volume = medical_volume.slicer[
            :, :, lower_level_index:upper_level_index
        ]
        inference_pipeline.medical_volume = medical_volume

        if self.save:
            nib.save(
                segmentation,
                os.path.join(
                    inference_pipeline.output_dir, "segmentations", "spine.nii.gz"
                ),
            )
            nib.save(
                medical_volume,
                os.path.join(
                    inference_pipeline.output_dir,
                    "segmentations",
                    "converted_dcm.nii.gz",
                ),
            )
        return {}


class SpineComputeROIs(InferenceClass):
    def __init__(self, spine_model):
        super().__init__()
        self.spine_model_name = spine_model
        self.spine_model_type = Models.model_from_name(self.spine_model_name)

    def __call__(self, inference_pipeline):
        # Compute ROIs
        inference_pipeline.spine_model_type = self.spine_model_type

        (spine_hus, rois, segmentation_hus, centroids_3d, spine_masks) = (
            spine_utils.compute_rois(
                inference_pipeline.segmentation,
                inference_pipeline.medical_volume,
                self.spine_model_type,
            )
        )

        inference_pipeline.spine_hus = spine_hus
        inference_pipeline.segmentation_hus = segmentation_hus
        inference_pipeline.rois = rois
        inference_pipeline.centroids_3d = centroids_3d
        inference_pipeline.spine_masks = spine_masks

        # save the ROIs and spine masks as a single array
        full_rois_array = np.zeros(
            inference_pipeline.medical_volume.shape, dtype=np.int8
        )
        full_spine_masks_array = np.zeros(
            inference_pipeline.medical_volume.shape, dtype=np.int8
        )

        for i, level in enumerate(rois.keys()):
            full_rois_array[rois[level] > 0] = i + 1
            # full_spine_masks_array[spine_masks[level] > 0] = i + 1

        inference_pipeline.saveArrToNifti(
            full_rois_array,
            os.path.join(
                inference_pipeline.output_dir_masks, "central_rois_mask.nii.gz"
            ),
        )

        nib.save(
            inference_pipeline.segmentation,
            os.path.join(inference_pipeline.output_dir_masks, "spine_seg.nii.gz"),
        )

        # inference_pipeline.saveArrToNifti(
        #     full_spine_masks_array,
        #     os.path.join(
        #         inference_pipeline.output_dir_masks,
        #         "vertebrae_body_mask.nii.gz",
        #     ),
        # )

        return {}


class SpineMetricsSaver(InferenceClass):
    """Save metrics to a CSV file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        """Save metrics to a CSV file."""
        self.spine_hus = inference_pipeline.spine_hus
        self.seg_hus = inference_pipeline.segmentation_hus
        self.output_dir = inference_pipeline.output_dir
        self.csv_output_dir = os.path.join(self.output_dir, "metrics")
        if not os.path.exists(self.csv_output_dir):
            os.makedirs(self.csv_output_dir, exist_ok=True)
        self.save_results()
        # if hasattr(inference_pipeline, "dicom_ds"):
        #     if not os.path.exists(os.path.join(self.output_dir, "dicom_metadata.csv")):
        #         io_utils.write_dicom_metadata_to_csv(
        #             inference_pipeline.dicom_ds,
        #             os.path.join(self.output_dir, "dicom_metadata.csv"),
        #         )

        return {}

    def save_results(self):
        """Save results to a CSV file."""
        df = pd.DataFrame(columns=["Level", "ROI HU", "Seg HU"])
        for i, level in enumerate(self.spine_hus):
            hu = self.spine_hus[level]
            seg_hu = self.seg_hus[level]
            row = [level, hu, seg_hu]
            df.loc[i] = row
        df = df.iloc[::-1]
        df.to_csv(os.path.join(self.csv_output_dir, "spine_metrics.csv"), index=False)


class SpineFindDicoms(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        inferior_superior_centers = spine_utils.find_spine_dicoms(
            inference_pipeline.centroids_3d,
        )

        spine_utils.save_nifti_select_slices(
            inference_pipeline.output_dir, inferior_superior_centers
        )
        inference_pipeline.dicom_file_paths = [
            str(center) for center in inferior_superior_centers
        ]
        inference_pipeline.names = list(inference_pipeline.rois.keys())
        inference_pipeline.dicom_file_names = list(inference_pipeline.rois.keys())
        inference_pipeline.inferior_superior_centers = inferior_superior_centers

        return {}


class SpineCoronalSagittalVisualizer(InferenceClass):
    def __init__(self, format="png"):
        super().__init__()
        self.format = format

    def __call__(self, inference_pipeline):
        output_path = inference_pipeline.output_dir
        spine_model_type = inference_pipeline.spine_model_type

        img_sagittal, img_coronal = spine_utils.visualize_coronal_sagittal_spine(
            inference_pipeline.segmentation.get_fdata(),
            list(inference_pipeline.rois.values()),
            inference_pipeline.medical_volume.get_fdata(),
            list(inference_pipeline.centroids_3d.values()),
            output_path,
            spine_hus=inference_pipeline.spine_hus,
            seg_hus=inference_pipeline.segmentation_hus,
            model_type=spine_model_type,
            pixel_spacing=inference_pipeline.pixel_spacing_list,
            format=self.format,
        )
        inference_pipeline.spine_vis_sagittal = img_sagittal
        inference_pipeline.spine_vis_coronal = img_coronal
        inference_pipeline.spine = True
        if not inference_pipeline.save_segmentations:
            shutil.rmtree(os.path.join(output_path, "segmentations"))
        return {}


class SpineReport(InferenceClass):
    def __init__(self, format="png"):
        super().__init__()
        self.format = format

    def __call__(self, inference_pipeline):
        sagittal_image = inference_pipeline.spine_vis_sagittal
        coronal_image = inference_pipeline.spine_vis_coronal
        # concatenate these numpy arrays laterally
        img = np.concatenate((coronal_image, sagittal_image), axis=1)
        output_path = os.path.join(
            inference_pipeline.output_dir, "images", "spine_report"
        )
        if self.format == "png":
            im = Image.fromarray(img)
            im.save(output_path + ".png")
        elif self.format == "dcm":
            to_dicom(img, output_path + ".dcm")
        return {}


class SpineMuscleAdiposeTissueReport(InferenceClass):
    """Spine muscle adipose tissue report class."""

    def __init__(self):
        super().__init__()
        self.image_files = [
            "spine_coronal.png",
            "spine_sagittal.png",
            "T12.png",
            "L1.png",
            "L2.png",
            "L3.png",
            "L4.png",
            "L5.png",
        ]

    def __call__(self, inference_pipeline):
        image_dir = Path(inference_pipeline.output_dir) / "images"
        self.generate_panel(image_dir)
        return {}

    def generate_panel(self, image_dir: Union[str, Path]):
        """Generate panel.
        Args:
            image_dir (Union[str, Path]): Path to the image directory.
        """
        image_files = [os.path.join(image_dir, path) for path in self.image_files]
        # construct a list which includes only the images that exist
        image_files = [path for path in image_files if os.path.exists(path)]

        im_cor = Image.open(image_files[0])
        im_sag = Image.open(image_files[1])
        im_cor_width = int(im_cor.width / im_cor.height * 512)
        num_muscle_fat_cols = math.ceil((len(image_files) - 2) / 2)
        width = (8 + im_cor_width + 8) + ((512 + 8) * num_muscle_fat_cols)
        height = 1048
        new_im = Image.new("RGB", (width, height))

        index = 2
        for j in range(8, height, 520):
            for i in range(8 + im_cor_width + 8, width, 520):
                try:
                    im = Image.open(image_files[index])
                    im.thumbnail((512, 512))
                    new_im.paste(im, (i, j))
                    index += 1
                    im.close()
                except Exception:
                    continue

        im_cor.thumbnail((im_cor_width, 512))
        new_im.paste(im_cor, (8, 8))
        im_sag.thumbnail((im_cor_width, 512))
        new_im.paste(im_sag, (8, 528))
        new_im.save(os.path.join(image_dir, "spine_muscle_adipose_tissue_report.png"))
        im_cor.close()
        im_sag.close()
        new_im.close()
