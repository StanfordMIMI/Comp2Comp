import os
import re
import zipfile
from pathlib import Path
from time import perf_counter
from typing import List, Union

import cv2
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import wget
from keras import backend as K
from tqdm import tqdm

from comp2comp.inference_class_base import InferenceClass
from comp2comp.metrics.metrics import CrossSectionalArea, HounsfieldUnits
from comp2comp.models.models import Models
from comp2comp.muscle_adipose_tissue.data import Dataset, predict


class MuscleAdiposeTissueSegmentation(InferenceClass):
    """Muscle adipose tissue segmentation class."""

    def __init__(self, batch_size: int, model_name: str, model_dir: str = None):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_type = Models.model_from_name(model_name)

    def forward_pass_2d(self, files):
        dataset = Dataset(files, windows=self.model_type.windows)
        num_workers = 1

        print("Computing segmentation masks using {}...".format(self.model_name))
        start_time = perf_counter()
        _, preds, results = predict(
            self.model,
            dataset,
            num_workers=num_workers,
            use_multiprocessing=num_workers > 1,
            batch_size=self.batch_size,
        )
        K.clear_session()
        print(
            f"Completed {len(files)} segmentations in {(perf_counter() - start_time):.2f} seconds."
        )
        for i in range(len(results)):
            results[i]["preds"] = preds[i]
        return results

    def download_muscle_adipose_tissue_model(self, model_dir: Union[str, Path]):
        download_dir = Path(
            os.path.join(
                model_dir,
                ".totalsegmentator/nnunet/results/nnUNet/2d/Task927_FatMuscle/nnUNetTrainerV2__nnUNetPlansv2.1",
            )
        )
        all_path = download_dir / "all"
        if not os.path.exists(all_path):
            download_dir.mkdir(parents=True, exist_ok=True)
            wget.download(
                "https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/resolve/main/all.zip",
                out=os.path.join(download_dir, "all.zip"),
            )
            with zipfile.ZipFile(os.path.join(download_dir, "all.zip"), "r") as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(os.path.join(download_dir, "all.zip"))
            wget.download(
                "https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/resolve/main/plans.pkl",
                out=os.path.join(download_dir, "plans.pkl"),
            )
            print("Muscle and adipose tissue model downloaded.")
        else:
            print("Muscle and adipose tissue model already downloaded.")

    def __call__(self, inference_pipeline):
        inference_pipeline.muscle_adipose_tissue_model_type = self.model_type
        inference_pipeline.muscle_adipose_tissue_model_name = self.model_name

        if self.model_name == "stanford_v0.0.2":
            self.download_muscle_adipose_tissue_model(inference_pipeline.model_dir)
            nifti_path = os.path.join(
                inference_pipeline.output_dir,
                "segmentations",
                "converted_dcm_multilevel.nii.gz",
            )
            output_path = os.path.join(
                inference_pipeline.output_dir,
                "segmentations",
                "multilevel_muscle_fat_seg.nii.gz",
            )

            from nnunet.inference import predict

            predict.predict_cases(
                model=os.path.join(
                    inference_pipeline.model_dir,
                    ".totalsegmentator/nnunet/results/nnUNet/2d/Task927_FatMuscle/nnUNetTrainerV2__nnUNetPlansv2.1",
                ),
                list_of_lists=[[nifti_path]],
                output_filenames=[output_path],
                folds="all",
                save_npz=False,
                num_threads_preprocessing=8,
                num_threads_nifti_save=8,
                segs_from_prev_stage=None,
                do_tta=False,
                mixed_precision=True,
                overwrite_existing=False,
                all_in_gpu=False,
                step_size=0.5,
                checkpoint_name="model_final_checkpoint",
                segmentation_export_kwargs=None,
            )

            image_nib = nib.load(nifti_path)
            image_nib = nib.as_closest_canonical(image_nib)
            image = image_nib.get_fdata()
            pred = nib.load(output_path)
            pred = nib.as_closest_canonical(pred)
            pred = pred.get_fdata()

            images = [image[:, :, i] for i in range(image.shape[-1])]
            preds = [pred[:, :, i] for i in range(pred.shape[-1])]

            # flip both axes and transpose
            images = [np.flip(np.flip(image, axis=0), axis=1).T for image in images]
            preds = [np.flip(np.flip(pred, axis=0), axis=1).T for pred in preds]

            spacings = [
                image_nib.header.get_zooms()[0:2] for i in range(image.shape[-1])
            ]

            categories = self.model_type.categories

            # for each image in images, convert to one hot encoding
            masks = []
            for pred in preds:
                mask = np.zeros((pred.shape[0], pred.shape[1], 4))
                for i, category in enumerate(categories):
                    mask[:, :, i] = pred == categories[category]
                mask = mask.astype(np.uint8)
                masks.append(mask)
            return {"images": images, "preds": masks, "spacings": spacings}

        else:
            dicom_file_paths = inference_pipeline.dicom_file_paths
            # if dicom_file_names not an attribute of inference_pipeline, add it
            if not hasattr(inference_pipeline, "dicom_file_names"):
                inference_pipeline.dicom_file_names = [
                    dicom_file_path.stem for dicom_file_path in dicom_file_paths
                ]
            self.model = self.model_type.load_model(inference_pipeline.model_dir)

            results = self.forward_pass_2d(dicom_file_paths)
            images = []
            for result in results:
                images.append(result["image"])
            preds = []
            for result in results:
                preds.append(result["preds"])
            spacings = []
            for result in results:
                spacings.append(result["spacing"])

            return {"images": images, "preds": preds, "spacings": spacings}


class MuscleAdiposeTissuePostProcessing(InferenceClass):
    """Post-process muscle and adipose tissue segmentation."""

    def __init__(self):
        super().__init__()

    def preds_to_mask(self, preds):
        """Convert model predictions to a mask.

        Args:
            preds (np.ndarray): Model predictions.

        Returns:
            np.ndarray: Mask.
        """
        if self.use_softmax:
            # softmax
            labels = np.zeros_like(preds, dtype=np.uint8)
            l_argmax = np.argmax(preds, axis=-1)
            for c in range(labels.shape[-1]):
                labels[l_argmax == c, c] = 1
            return labels.astype(np.bool)
        else:
            # sigmoid
            return preds >= 0.5

    def __call__(self, inference_pipeline, images, preds, spacings):
        """Post-process muscle and adipose tissue segmentation."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.use_softmax = self.model_type.use_softmax
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        return self.post_process(images, preds, spacings)

    def remove_small_objects(self, mask, min_size=10):
        mask = mask.astype(np.uint8)
        components, output, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        sizes = stats[1:, -1]
        mask = np.zeros((output.shape))
        for i in range(0, components - 1):
            if sizes[i] >= min_size:
                mask[output == i + 1] = 1
        return mask

    def post_process(
        self,
        images,
        preds,
        spacings,
    ):
        categories = self.model_type.categories

        start_time = perf_counter()

        if self.model_name == "stanford_v0.0.2":
            masks = preds
        else:
            masks = [self.preds_to_mask(p) for p in preds]

        for i, _ in enumerate(masks):
            # Keep only channels from the model_type categories dict
            masks[i] = np.squeeze(masks[i])

        masks = self.fill_holes(masks)

        cats = list(categories.keys())

        file_idx = 0
        for mask, image in tqdm(zip(masks, images), total=len(masks)):
            muscle_mask = mask[..., cats.index("muscle")]
            imat_mask = mask[..., cats.index("imat")]
            imat_mask = (
                np.logical_and(
                    (image * muscle_mask) <= -30, (image * muscle_mask) >= -190
                )
            ).astype(int)
            imat_mask = self.remove_small_objects(imat_mask)
            mask[..., cats.index("imat")] += imat_mask
            mask[..., cats.index("muscle")][imat_mask == 1] = 0
            masks[file_idx] = mask
            images[file_idx] = image
            file_idx += 1

        print(
            f"Completed post-processing in {(perf_counter() - start_time):.2f} seconds."
        )

        return {"images": images, "masks": masks, "spacings": spacings}

    # function that fills in holes in a segmentation mask
    def _fill_holes(self, mask: np.ndarray, mask_id: int):
        """Fill in holes in a segmentation mask.

        Args:
            mask (ndarray): NxHxW
            mask_id (int): Label of the mask.

        Returns:
            ndarray: Filled mask.
        """
        int_mask = ((1 - mask) > 0.5).astype(np.int8)
        components, output, stats, _ = cv2.connectedComponentsWithStats(
            int_mask, connectivity=8
        )
        sizes = stats[1:, -1]
        components = components - 1
        # Larger threshold for SAT
        # TODO make this configurable / parameter
        if mask_id == 2:
            min_size = 50
            # min_size = 0
        else:
            min_size = 5
            # min_size = 0
        img_out = np.ones_like(mask)
        for i in range(0, components):
            if sizes[i] > min_size:
                img_out[output == i + 1] = 0
        return img_out

    def fill_holes(self, ys: List):
        """Take an array of size NxHxWxC and for each channel fill in holes.

        Args:
            ys (list): List of segmentation masks.
        """
        segs = []
        for n in range(len(ys)):
            ys_out = [
                self._fill_holes(ys[n][..., i], i) for i in range(ys[n].shape[-1])
            ]
            segs.append(np.stack(ys_out, axis=2).astype(float))

        return segs


class MuscleAdiposeTissueComputeMetrics(InferenceClass):
    """Compute muscle and adipose tissue metrics."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, images, masks, spacings):
        """Compute muscle and adipose tissue metrics."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        metrics = self.compute_metrics_all(images, masks, spacings)
        return metrics

    def compute_metrics_all(self, images, masks, spacings):
        """Compute metrics for all images and masks.

        Args:
            images (List[np.ndarray]): Images.
            masks (List[np.ndarray]): Masks.

        Returns:
            Dict: Results.
        """
        results = []
        for image, mask, spacing in zip(images, masks, spacings):
            results.append(self.compute_metrics(image, mask, spacing))
        return {"images": images, "results": results}

    def compute_metrics(self, x, mask, spacing):
        """Compute results for a given segmentation."""
        categories = self.model_type.categories

        hu = HounsfieldUnits()
        csa_units = "cm^2" if spacing else ""
        csa = CrossSectionalArea(csa_units)

        hu_vals = hu(mask, x, category_dim=-1)
        csa_vals = csa(mask=mask, spacing=spacing, category_dim=-1)

        # check if any values are nan and replace with 0
        hu_vals = np.nan_to_num(hu_vals)
        csa_vals = np.nan_to_num(csa_vals)

        if mask.shape[-1] != len(categories):
            # TODO: Handle this properly. This is a hard fix removing the BG class, 
            # which is added by the abCT_v0.0.1 model in the end. 
            mask = mask[..., :-1]

        assert mask.shape[-1] == len(
            categories
        ), "{} categories found in mask, " "but only {} categories specified".format(
            mask.shape[-1], len(categories)
        )

        results = {
            cat: {
                "mask": mask[..., idx],
                hu.name(): hu_vals[idx],
                csa.name(): csa_vals[idx],
            }
            for idx, cat in enumerate(categories.keys())
        }
        return results


class MuscleAdiposeTissueH5Saver(InferenceClass):
    """Save results to an HDF5 file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, results):
        """Save results to an HDF5 file."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        self.output_dir = inference_pipeline.output_dir
        self.h5_output_dir = os.path.join(self.output_dir, "segmentations")
        os.makedirs(self.h5_output_dir, exist_ok=True)
        self.dicom_file_paths = inference_pipeline.dicom_file_paths
        self.dicom_file_names = inference_pipeline.dicom_file_names
        self.save_results(results)
        return {"results": results}

    def save_results(self, results):
        """Save results to an HDF5 file."""
        categories = self.model_type.categories
        cats = list(categories.keys())

        for i, result in enumerate(results):
            file_name = self.dicom_file_names[i]
            with h5py.File(
                os.path.join(self.h5_output_dir, file_name + ".h5"), "w"
            ) as f:
                for cat in cats:
                    mask = result[cat]["mask"]
                    f.create_dataset(name=cat, data=np.array(mask, dtype=np.uint8))


def natural_sort_key(s):
    """
    Create a key for sorting strings in a 'natural' order. e.g., 'slice10' comes after 'slice2'.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class MuscleAdiposeTissueNiftiSaver(InferenceClass):
    """
    Saves the multi-class muscle and adipose tissue segmentations as a single multi-labeled NIfTI file.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, images, results):
        """Orchestrates the entire saving and assembly process."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.output_dir = inference_pipeline.output_dir
        self.nifti_output_dir = os.path.join(self.output_dir, "segmentations")
        self.dicom_file_names = inference_pipeline.dicom_file_names
        os.makedirs(self.nifti_output_dir, exist_ok=True)
        self.spacings = getattr(inference_pipeline, 'spacings', None)
        self.save_results(results)
        return {"results": results}

    def save_results(self, results):
        """Saves NIfTI file."""
        categories = self.model_type.categories

        slices = {}
        for i, result in enumerate(results):
            file_name = self.dicom_file_names[i]
            first_cat_name = list(categories.keys())[0]
            if first_cat_name not in result:
                continue
            mask_shape = result[first_cat_name]["mask"].shape
            multi_label_slice = np.zeros(mask_shape, dtype=np.uint8)
            for class_name, label in categories.items():
                if class_name in result:
                    class_mask = result[class_name]["mask"]
                    multi_label_slice[class_mask > 0] = label + 1
            slices[file_name] = multi_label_slice
        slices = [slices[fname] for fname in sorted(slices.keys(), key=natural_sort_key)]
        
        final_image = sitk.GetImageFromArray(np.stack(slices, axis=0)[::-1, :, :])
        if self.spacings and len(self.spacings) > 0:
            # Assumes spacing is (x, y, z)
            final_spacing = tuple(float(s) for s in self.spacings[0])
            final_image.SetSpacing(final_spacing)
        else:
            final_image.SetSpacing((1.0, 1.0, 1.0))
        
        sitk.WriteImage(final_image, os.path.join(self.nifti_output_dir, "muscle_adipose_tissue_seg.nii.gz"))


class MuscleAdiposeTissueMetricsSaver(InferenceClass):
    """Save metrics to a CSV file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline, results):
        """Save metrics to a CSV file."""
        self.model_type = inference_pipeline.muscle_adipose_tissue_model_type
        self.model_name = inference_pipeline.muscle_adipose_tissue_model_name
        self.output_dir = inference_pipeline.output_dir
        self.csv_output_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(self.csv_output_dir, exist_ok=True)
        self.dicom_file_paths = inference_pipeline.dicom_file_paths
        self.dicom_file_names = inference_pipeline.dicom_file_names
        self.save_results(results)
        return {}

    def save_results(self, results):
        """Save results to a CSV file."""
        self.model_type.categories
        df = pd.DataFrame(
            columns=[
                "Level",
                "Index",
                "Muscle HU",
                "Muscle CSA (cm^2)",
                "SAT HU",
                "SAT CSA (cm^2)",
                "VAT HU",
                "VAT CSA (cm^2)",
                "IMAT HU",
                "IMAT CSA (cm^2)",
            ]
        )

        for i, result in enumerate(results):
            row = []
            row.append(self.dicom_file_names[i])
            row.append(self.dicom_file_paths[i])
            for cat in result:
                row.append(result[cat]["Hounsfield Unit"])
                row.append(result[cat]["Cross-sectional Area (cm^2)"])
            df.loc[i] = row
        df = df.iloc[::-1]
        df.to_csv(
            os.path.join(self.csv_output_dir, "muscle_adipose_tissue_metrics.csv"),
            index=False,
        )
