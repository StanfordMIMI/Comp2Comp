import os
from pathlib import Path
from time import perf_counter
from typing import List

import cv2
import h5py
import numpy as np
import pandas as pd
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

    def __call__(self, inference_pipeline):
        inference_pipeline.muscle_adipose_tissue_model_type = self.model_type
        inference_pipeline.muscle_adipose_tissue_model_name = self.model_name
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

        masks = [self.preds_to_mask(p) for p in preds]

        for i, _ in enumerate(masks):
            # Keep only channels from the model_type categories dict
            masks[i] = masks[i][..., [categories[cat] for cat in categories]]

        masks = self.fill_holes(masks)

        cats = list(categories.keys())

        file_idx = 0
        for mask, image in tqdm(zip(masks, images), total=len(masks)):
            muscle_mask = mask[..., cats.index("muscle")]
            imat_mask = mask[..., cats.index("imat")]
            imat_mask = (
                np.logical_and((image * muscle_mask) <= -30, (image * muscle_mask) >= -190)
            ).astype(int)
            imat_mask = self.remove_small_objects(imat_mask)
            mask[..., cats.index("imat")] += imat_mask
            mask[..., cats.index("muscle")][imat_mask == 1] = 0
            masks[file_idx] = mask
            images[file_idx] = image
            file_idx += 1

        print(f"Completed post-processing in {(perf_counter() - start_time):.2f} seconds.")

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
        components, output, stats, _ = cv2.connectedComponentsWithStats(int_mask, connectivity=8)
        sizes = stats[1:, -1]
        components = components - 1
        # Larger threshold for SAT
        # TODO make this configurable / parameter
        if mask_id == 2:
            min_size = 200
        else:
            # min_size = 50  # Smaller threshold for everything else
            min_size = 20
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
            ys_out = [self._fill_holes(ys[n][..., i], i) for i in range(ys[n].shape[-1])]
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
            with h5py.File(os.path.join(self.h5_output_dir, file_name + ".h5"), "w") as f:
                for cat in cats:
                    mask = result[cat]["mask"]
                    f.create_dataset(name=cat, data=np.array(mask, dtype=np.uint8))


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
        categories = self.model_type.categories
        cats = list(categories.keys())
        df = pd.DataFrame(
            columns=[
                "File Name",
                "File Path",
                "Muscle HU",
                "Muscle CSA (cm^2)",
                "IMAT HU",
                "IMAT CSA (cm^2)",
                "SAT HU",
                "SAT CSA (cm^2)",
                "VAT HU",
                "VAT CSA (cm^2)",
            ]
        )

        for i, result in enumerate(results):
            row = []
            row.append(self.dicom_file_names[i])
            row.append(self.dicom_file_paths[i])
            for cat in cats:
                row.append(result[cat]["Hounsfield Unit"])
                row.append(result[cat]["Cross-sectional Area (cm^2)"])
            df.loc[i] = row
        df.to_csv(
            os.path.join(self.csv_output_dir, "muscle_adipose_tissue_metrics.csv"), index=False
        )
