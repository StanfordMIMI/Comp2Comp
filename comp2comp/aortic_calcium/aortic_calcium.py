#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:36:05 2023

@author: maltejensen
"""
import os
import time
from pathlib import Path
from typing import Union

import numpy as np
from scipy import ndimage
from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.inference_class_base import InferenceClass


class AortaSegmentation(InferenceClass):
    """Aorta segmentation."""

    def __init__(self):
        super().__init__()
        # self.input_path = input_path

    def __call__(self, inference_pipeline):
        # inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        inference_pipeline.output_dir_segmentations = os.path.join(
            self.output_dir, "segmentations/"
        )

        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        mv, seg = self.aorta_seg(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            self.output_dir_segmentations + "organs.nii.gz",
            inference_pipeline.model_dir,
        )
        # the medical volume is already set by the spine segmentation model
        # the toCanonical methods looks for "segmentation", so it's overridden
        inference_pipeline.spine_segmentation = inference_pipeline.segmentation
        inference_pipeline.segmentation = seg
        # inference_pipeline.medical_volume = mv

        return {}

    def aorta_seg(self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir):
        """Run organ segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting aorta...")
        st = time.time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [251]

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
                nora_tag="None",
                preview=False,
                nr_threads_resampling=1,
                nr_threads_saving=6,
                quiet=False,
                verbose=True,
                test=0,
            )
        end = time.time()

        # Log total time for spine segmentation
        print(f"Total time for aorta segmentation: {end-st:.2f}s.")

        return seg, mvs


class AorticCalciumSegmentation(InferenceClass):
    """Segmentaiton of aortic calcium"""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        ct = inference_pipeline.medical_volume.get_fdata()
        aorta_mask = inference_pipeline.segmentation.get_fdata() == 7
        spine_mask = inference_pipeline.spine_segmentation.get_fdata() > 0

        inference_pipeline.calc_mask = self.detectCalcifications(
            ct, aorta_mask, exclude_mask=spine_mask, remove_size=3
        )

        self.output_dir = inference_pipeline.output_dir
        self.output_dir_images_organs = os.path.join(self.output_dir, "images/")
        inference_pipeline.output_dir_images_organs = self.output_dir_images_organs

        if not os.path.exists(self.output_dir_images_organs):
            os.makedirs(self.output_dir_images_organs)

        # np.save(os.path.join(self.output_dir_images_organs, 'ct.npy'), ct)
        np.save(os.path.join(self.output_dir_images_organs, "aorta_mask.npy"), aorta_mask)
        np.save(os.path.join(self.output_dir_images_organs, "spine_mask.npy"), spine_mask)

        return {}

    def detectCalcifications(
        self,
        ct,
        aorta_mask,
        exclude_mask=None,
        return_dilated_mask=False,
        dilation=(3, 1),
        dilation_iteration=4,
        return_dilated_exclude=False,
        dilation_exclude_mask=(3, 1),
        dilation_iteration_exclude=3,
        show_time=False,
        num_std=3,
        remove_size=None,
        verbose=False,
        exclude_center_aorta=True,
        return_eroded_aorta=False,
        aorta_erode_iteration=6,
    ):
        """
        Function that takes in a CT image and aorta segmentation (and optionally volumes to use for exclusion of the segmentations),
        And returns a mask of the segmented calcifications (and optionally other volumes).
        The calcium threshold is adapative and uses the median of the CT points inside the aorta together with one
        standard devidation to the left, as this is more stable. The num_std is multiplied with the distance between the median
        and the one standard deviation mark, and can be used to control the threshold.

        Args:
            ct (array): CT image.
            aorta_mask (array): Mask of the aorta.
            exclude_mask (array, optional): Mask for structures to exclude e.g. spine. Defaults to None.
            return_dilated_mask (bool, optional): Return the dilated aorta mask. Defaults to False.
            dilation (list, optional): Structuring element for aorta dilation. Defaults to (3,1).
            dilation_iteration (int, optional): Number of iterations for the strcturing element. Defaults to 4.
            return_dilated_exclude (bool, optional): Return the dilated exclusio mask. Defaults to False.
            dilation_exclude_mask (list, optional): Structering element for the exclusio mask. Defaults to (3,1).
            dilation_iteration_exclude (int, optional): Number of iterations for the strcturing element. Defaults to 3.
            show_time (bool, optional): Show time for each operation. Defaults to False.
            num_std (float, optional): How many standard deviations out the threshold will be set at. Defaults to 3.
            remove_size (int, optional): Remove foci under a certain size. Warning: quite slow. Defaults to None.
            verbose (bool, optional): Give verbose feedback on operations. Defaults to False.
            exclude_center_aorta (bool, optional): Use eroded aorta to exclude center of the aorta. Defaults to True.
            return_eroded_aorta (bool, optional): Return the eroded center aorta. Defaults to False.
            aorta_erode_iteration (int, optional): Number of iterations for the strcturing element. Defaults to 6.

        Returns:
            results: array of only the mask is returned, or dict if other volumes are also returned.

        """

        def slicedDilationOrErosion(input_mask, struct, num_iteration, operation):
            """
            Perform the dilation on the smallest slice that will fit the
            segmentation
            """
            margin = 2 if num_iteration is None else num_iteration + 1

            x_idx = np.where(input_mask.sum(axis=(1, 2)))[0]
            x_start, x_end = x_idx[0] - margin, x_idx[-1] + margin
            y_idx = np.where(input_mask.sum(axis=(0, 2)))[0]
            y_start, y_end = y_idx[0] - margin, y_idx[-1] + margin

            if operation == "dilate":
                mask_slice = ndimage.binary_dilation(
                    input_mask[x_start:x_end, y_start:y_end, :], structure=struct
                ).astype(np.int8)
            elif operation == "erode":
                mask_slice = ndimage.binary_erosion(
                    input_mask[x_start:x_end, y_start:y_end, :], structure=struct
                ).astype(np.int8)

            output_mask = input_mask.copy()

            output_mask[x_start:x_end, y_start:y_end, :] = mask_slice

            return output_mask

        # remove parts that are not the abdominal aorta
        labelled_aorta, num_classes = ndimage.label(aorta_mask)
        if num_classes > 1:
            if verbose:
                print("Removing {} parts".format(num_classes - 1))

            aorta_vols = []

            for i in range(1, num_classes + 1):
                aorta_vols.append((labelled_aorta == i).sum())

            biggest_idx = np.argmax(aorta_vols) + 1
            aorta_mask[labelled_aorta != biggest_idx] = 0

        # Get aortic CT point to set adaptive threshold
        aorta_ct_points = ct[aorta_mask == 1]

        # equal to one standard deviation to the left of the curve
        quant = 0.158
        quantile_median_dist = np.median(aorta_ct_points) - np.quantile(aorta_ct_points, q=quant)
        calc_thres = np.median(aorta_ct_points) + quantile_median_dist * num_std

        t0 = time.time()

        if dilation is not None:
            struct = ndimage.generate_binary_structure(*dilation)
            if dilation_iteration is not None:
                struct = ndimage.iterate_structure(struct, dilation_iteration)
            aorta_dilated = slicedDilationOrErosion(
                aorta_mask, struct=struct, num_iteration=dilation_iteration, operation="dilate"
            )

            if show_time:
                print("dilation mask time: {:.2f}".format(time.time() - t0))

        t0 = time.time()
        calc_mask = np.logical_and(aorta_dilated == 1, ct >= calc_thres)
        if show_time:
            print("find calc time: {:.2f}".format(time.time() - t0))

        if exclude_center_aorta:
            t0 = time.time()

            struct = ndimage.generate_binary_structure(3, 1)
            struct = ndimage.iterate_structure(struct, aorta_erode_iteration)

            aorta_eroded = slicedDilationOrErosion(
                aorta_mask, struct=struct, num_iteration=aorta_erode_iteration, operation="erode"
            )
            calc_mask = calc_mask * (aorta_eroded == 0)
            if show_time:
                print("exclude center aorta time: {:.2f} sec".format(time.time() - t0))

        t0 = time.time()
        if exclude_mask is not None:
            if dilation_exclude_mask is not None:
                struct_exclude = ndimage.generate_binary_structure(*dilation_exclude_mask)
                if dilation_iteration_exclude is not None:
                    struct_exclude = ndimage.iterate_structure(
                        struct_exclude, dilation_iteration_exclude
                    )

                exclude_mask = slicedDilationOrErosion(
                    exclude_mask,
                    struct=struct_exclude,
                    num_iteration=dilation_iteration_exclude,
                    operation="dilate",
                )

            if show_time:
                print("exclude dilation time: {:.2f}".format(time.time() - t0))

            t0 = time.time()
            calc_mask = calc_mask * (exclude_mask == 0)
            if show_time:
                print("exclude time: {:.2f}".format(time.time() - t0))

        if remove_size is not None:
            t0 = time.time()

            labels, num_features = ndimage.label(calc_mask)

            counter = 0
            for n in range(1, num_features + 1):
                idx_tmp = labels == n
                if idx_tmp.sum() <= remove_size:
                    calc_mask[idx_tmp] = 0
                    counter += 1

            if show_time:
                print("Size exclusion time: {:.1f} sec".format(time.time() - t0))
            if verbose:
                print("Excluded {} foci under {}".format(counter, remove_size))

        if not all([return_dilated_mask, return_dilated_exclude]):
            return calc_mask.astype(np.int8)
        else:
            results = {}
            results["calc_mask"] = calc_mask.astype(np.int8)
            if return_dilated_mask:
                results["dilated_mask"] = aorta_dilated
            if return_dilated_exclude:
                results["dilated_exclude"] = exclude_mask
            if return_eroded_aorta:
                results["aorta_eroded"] = aorta_eroded

            results["threshold"] = calc_thres

            return results


class AorticCalciumMetrics(InferenceClass):
    """Calculate metrics for the aortic calcifications"""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        calc_mask = inference_pipeline.calc_mask

        inference_pipeline.pix_dims = inference_pipeline.medical_volume.header["pixdim"][1:4]
        # divided with 10 to get in cm
        inference_pipeline.vol_per_pixel = np.prod(inference_pipeline.pix_dims / 10)

        # count statistics for individual calcifications
        labelled_calc, num_lesions = ndimage.label(calc_mask)

        metrics = {
            "volume": [],
            "mean_hu": [],
            "median_hu": [],
            "max_hu": [],
        }

        ct = inference_pipeline.medical_volume.get_fdata()

        for i in range(1, num_lesions + 1):
            tmp_mask = labelled_calc == i

            tmp_ct_vals = ct[tmp_mask]

            metrics["volume"].append(len(tmp_ct_vals) * inference_pipeline.vol_per_pixel)
            metrics["mean_hu"].append(np.mean(tmp_ct_vals))
            metrics["median_hu"].append(np.median(tmp_ct_vals))
            metrics["max_hu"].append(np.max(tmp_ct_vals))

        # Volume of calcificaitons
        calc_vol = np.sum(metrics["volume"])
        metrics["volume_total"] = calc_vol

        metrics["num_calc"] = len(metrics["volume"])

        inference_pipeline.metrics = metrics

        return {}
