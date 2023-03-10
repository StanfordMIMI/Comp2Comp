import argparse
import logging
import os
from time import perf_counter
from typing import List
from pathlib import Path

import silx.io.dictdump as sio
from keras import backend as K
from tqdm import tqdm
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

from comp2comp.muscle_adipose_tissue.data import Dataset, fill_holes, predict
from comp2comp.models.models import Models
from comp2comp.utils import dl_utils
from comp2comp.metrics.metrics import CrossSectionalArea, HounsfieldUnits

from comp2comp.inference_class_base import InferenceClass


class MuscleAdiposeTissueSegmentation(InferenceClass):
    """Muscle adipose tissue segmentation class."""
    def __init__(self, batch_size: int, model_name: str, model_dir: str = None):
        super().__init__()
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_type = Models.model_from_name(model_name)

    def forward_pass_2d(
        self,
        files
    ):
        dataset = Dataset(files, windows=self.model_type.windows)
        categories = self.model_type.categories
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
        print(f"Completed {len(files)} segmentations in {(perf_counter() - start_time):.2f} seconds.")
        for i in range(len(results)):
            results[i]["preds"] = preds[i]
        return results

    def __call__(
        self,
        inference_pipeline,
        dicom_file_paths: List[Path]
    ):
        inference_pipeline.muscle_adipose_tissue_model_type = self.model_type
        inference_pipeline.muscle_adipose_tissue_model_name = self.model_name
        self.model = self.model_type.load_model(inference_pipeline.model_dir)

        results = self.forward_pass_2d(
            dicom_file_paths
        )
        images = []
        for result in results:
            images.append(result["image"])
        preds = []
        for result in results:
            preds.append(result["preds"])
        spacings = []
        for result in results:
            spacings.append(result["spacing"])

        return {
            "images": images,
            "preds": preds,
            "spacings": spacings
        }


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
        components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
        m_name = self.model_name
        categories = self.model_type.categories

        start_time = perf_counter()

        masks = [self.preds_to_mask(p) for p in preds]

        for i, mask in enumerate(masks):
            # Keep only channels from the model_type categories dict
            masks[i] = masks[i][
                ..., [categories[cat] for cat in categories]
            ]

        masks = fill_holes(masks)
        cats = list(categories.keys())
        
        if "muscle" in categories and "imat" in categories:
            muscle_idx = cats.index("muscle")

        file_idx = 0
        for mask, image in tqdm(zip(masks, images), total=len(masks)):
            muscle_mask = mask[..., cats.index("muscle")]
            imat_mask = mask[..., cats.index("imat")]
            imat_mask = (np.logical_and((image * muscle_mask) <= -30, (image * muscle_mask) >= -190)).astype(int)
            imat_mask = self.remove_small_objects(imat_mask)
            mask[..., cats.index("imat")] += imat_mask
            mask[..., cats.index("muscle")][imat_mask == 1] = 0
            masks[file_idx] = mask
            images[file_idx] = image
            file_idx += 1

        print(f"Completed post-processing in {(perf_counter() - start_time):.2f} seconds.")

        return {"images": images, "masks": masks, "spacings": spacings} 


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
        return {"results": results}

    def compute_metrics(self, x, mask, spacing):
        """Compute results for a given segmentation.
        """
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
                "image": x,
                "mask": mask[..., idx],
                hu.name(): hu_vals[idx],
                csa.name(): csa_vals[idx],
            }
            for idx, cat in enumerate(categories.keys())
        }
        return results



    
