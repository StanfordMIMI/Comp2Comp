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

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pydicom 

# from totalsegmentator.libs import (
#     download_pretrained_weights,
#     nostdout,
#     setup_nnunet,
# )
from totalsegmentatorv2.python_api import totalsegmentator

from comp2comp.inference_class_base import InferenceClass


class AortaSegmentation(InferenceClass):
    """Aorta segmentation."""

    def __init__(self):
        super().__init__()
        # self.input_path = input_path

    def __call__(self, inference_pipeline):
        # check if kernels are allowed if agatston is used
        if inference_pipeline.args.threshold == 'agatston':
            self.reconKernelChecker(inference_pipeline.dcm)
            
        # inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        inference_pipeline.output_dir_segmentations = os.path.join(
            self.output_dir, "segmentations/"
        )

        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        seg = self.aorta_seg(
            os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            self.output_dir_segmentations + "organs.nii.gz",
            inference_pipeline.model_dir,
        )
        # the medical volume is already set by the spine segmentation model
        # the toCanonical methods looks for "segmentation", so it's overridden
        inference_pipeline.spine_segmentation = inference_pipeline.segmentation
        inference_pipeline.segmentation = seg

        return {}

    def aorta_seg(
        self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir
    ):
        """Run organ segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting aorta...")
        st = time.time()
        os.environ["SCRATCH"] = self.model_dir

        seg = totalsegmentator(
            input=os.path.join(self.output_dir_segmentations, "converted_dcm.nii.gz"),
            output=os.path.join(self.output_dir_segmentations, "segmentation.nii"),
            task_ids=[293],
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

        end = time.time()

        # Log total time for spine segmentation
        print(f"Total time for aorta segmentation: {end-st:.2f}s.")

        return seg

    def reconKernelChecker(self, dcm):
        ge_kernels = ["standard", "md stnd"]
        philips_kernels = ["a", "b", "c", "sa", "sb"]
        canon_kernels = ["fc08", "fc18"]
        siemens_kernels = ["b20s", "b20f", "b30f", "b31s", "b31f", "br34f", "b35f", "bf37f", "br38f", "b41f", 
                        "qr40", "qr40d", "br36f", "br40", "b40f", "br40d", "i30f", "i31f", "i26f", "i31s", 
                        "i40f", "b30s", "br36d", "bf39f", "b41s", "br40f"]
        toshiba_kernels = ["fc01", "fc02", "fc07", "fc08", "fc13", "fc18"]

        all_kernels = ge_kernels+philips_kernels+canon_kernels+siemens_kernels+toshiba_kernels
        
        conv_kernel_raw = dcm['ConvolutionKernel'].value
        
        if isinstance(conv_kernel_raw, pydicom.multival.MultiValue):
            conv_kernel = conv_kernel_raw[0].lower()
            recon_kernel_extra = str(conv_kernel_raw)
        else:
            conv_kernel = conv_kernel_raw.lower()
            recon_kernel_extra = 'n/a'
            
        if conv_kernel in all_kernels:
            return True
        else:          
            raise ValueError('Reconstruction kernel not allowed, found: ' + conv_kernel +'\n'
                             + 'Allowed kernels are: ' + str(all_kernels))

class AorticCalciumSegmentation(InferenceClass):
    """Segmentaiton of aortic calcium"""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        # Set output dirs
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_images_organs = os.path.join(self.output_dir, "images/")
        inference_pipeline.output_dir_images_organs = self.output_dir_images_organs
        self.output_dir_segmentation_masks = os.path.join(
            self.output_dir, "segmentation_masks/"
        )
        inference_pipeline.output_dir_segmentation_masks = (
            self.output_dir_segmentation_masks
        )

        if not os.path.exists(self.output_dir_images_organs):
            os.makedirs(self.output_dir_images_organs)
        if not os.path.exists(self.output_dir_segmentation_masks):
            os.makedirs(self.output_dir_segmentation_masks)
        if not os.path.exists(os.path.join(self.output_dir, "metrics/")):
            os.makedirs(os.path.join(self.output_dir, "metrics/"))

        ct = inference_pipeline.medical_volume.get_fdata()
        aorta_mask = inference_pipeline.segmentation.get_fdata().astype(np.int8) == 52
        spine_mask = inference_pipeline.spine_segmentation.get_fdata() > 0

        # Determine the target number of pixels
        pix_size = np.array(inference_pipeline.medical_volume.header.get_zooms())
        # target: 1 mm
        target_aorta_dil = round(1 / pix_size[0])
        # target: 3 mm
        target_exclude_dil = round(3 / pix_size[0])
        # target: 7 mm
        target_aorta_erode = round(7 / pix_size[0])

        # Run calcification detection pipeline
        calcification_results = self.detectCalcifications(
            ct,
            aorta_mask,
            exclude_mask=spine_mask,
            remove_size=3,
            return_dilated_mask=True,
            return_eroded_aorta=True,
            threshold=inference_pipeline.args.threshold,
            dilation_iteration=target_aorta_dil,
            dilation_iteration_exclude=target_exclude_dil,
            aorta_erode_iteration=target_aorta_erode,
        )

        inference_pipeline.calc_mask = calcification_results["calc_mask"]
        inference_pipeline.calcium_threshold = calcification_results["threshold"]

        # save masks
        inference_pipeline.saveArrToNifti(
            inference_pipeline.calc_mask,
            os.path.join(
                inference_pipeline.output_dir_segmentation_masks,
                "calcium_segmentations.nii.gz",
            ),
        )
        
        inference_pipeline.saveArrToNifti(
            calcification_results["dilated_mask"],
            os.path.join(
                inference_pipeline.output_dir_segmentation_masks,
                "dilated_aorta_mask.nii.gz",
            ),
        )
        
        inference_pipeline.saveArrToNifti(
            calcification_results["aorta_eroded"],
            os.path.join(
                inference_pipeline.output_dir_segmentation_masks,
                "eroded_aorta_mask.nii.gz",
            ),
        )

        inference_pipeline.saveArrToNifti(
            spine_mask,
            os.path.join(
                inference_pipeline.output_dir_segmentation_masks, "spine_mask.nii.gz"
            ),
        )
        
        inference_pipeline.saveArrToNifti(
            aorta_mask,
            os.path.join(
                inference_pipeline.output_dir_segmentation_masks, "aorta_mask.nii.gz"
            ),
        )

        inference_pipeline.saveArrToNifti(
            ct,
            os.path.join(inference_pipeline.output_dir_segmentation_masks, "ct.nii.gz"),
        )

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
        threshold="adaptive",
        agatston_failsafe=100,
        generate_plots=True,
    ):
        """
        Function that takes in a CT image and aorta segmentation (and optionally volumes to use
        for exclusion of the segmentations), And returns a mask of the segmented calcifications
        (and optionally other volumes). The calcium threshold is adapative and uses the median
        of the CT points inside the aorta together with one standard devidation to the left, as
        this is more stable. The num_std is multiplied with the distance between the median
        and the one standard deviation mark, and can be used to control the threshold.

        Args:
            ct (array): CT image.
            aorta_mask (array): Mask of the aorta.
            exclude_mask (array, optional):
                Mask for structures to exclude e.g. spine. Defaults to None.
            return_dilated_mask (bool, optional):
                Return the dilated aorta mask. Defaults to False.
            dilation (list, optional):
                Structuring element for aorta dilation. Defaults to (3,1).
            dilation_iteration (int, optional):
                Number of iterations for the strcturing element. Defaults to 4.
            return_dilated_exclude (bool, optional):
                Return the dilated exclusio mask. Defaults to False.
            dilation_exclude_mask (list, optional):
                Structering element for the exclusio mask. Defaults to (3,1).
            dilation_iteration_exclude (int, optional):
                Number of iterations for the strcturing element. Defaults to 3.
            show_time (bool, optional):
                Show time for each operation. Defaults to False.
            num_std (float, optional):
                How many standard deviations out the threshold will be set at. Defaults to 3.
            remove_size (int, optional):
                Remove foci under a certain size. Warning: quite slow. Defaults to None.
            verbose (bool, optional):
                Give verbose feedback on operations. Defaults to False.
            exclude_center_aorta (bool, optional):
                Use eroded aorta to exclude center of the aorta. Defaults to True.
            return_eroded_aorta (bool, optional):
                Return the eroded center aorta. Defaults to False.
            aorta_erode_iteration (int, optional):
                Number of iterations for the strcturing element. Defaults to 6.
            threshold: (str, int):
                Can either be 'adaptive', 'agatston', or int. Choosing 'agatston'
                Will mean a threshold of 130 HU.
            agatston_failsafe: (int):
                A fail-safe raising an error if the mean HU of the aorta is too high
                to reliably be using the agatston threshold of 130. Defaults to 100 HU.

        Returns:
            results: array of only the mask is returned, or dict if other volumes are also returned.

        """

        """
        Remove the ascending aorta if present
        """
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

        """
        Erode the center aorta to get statistics from the blood pool
        """
        t0 = time.time()

        struct = ndimage.generate_binary_structure(3, 1)
        struct = ndimage.iterate_structure(struct, aorta_erode_iteration)

        aorta_eroded = self.slicedDilationOrErosion(
            aorta_mask,
            struct=struct,
            num_iteration=aorta_erode_iteration,
            operation="erode",
        )

        eroded_ct_points = ct[aorta_eroded == 1]
        eroded_ct_points_mean = eroded_ct_points.mean()
        eroded_ct_points_std = eroded_ct_points.std()

        if generate_plots:
            # save the statistics of the eroded aorta for reference
            with open(
                os.path.join(self.output_dir, "metrics/eroded_aorta_statistics.csv"),
                "w",
            ) as f:
                f.write("metric,value\n")
                f.write("mean,{:.1f}\n".format(eroded_ct_points_mean))
                f.write("std,{:.1f}\n".format(eroded_ct_points_std))

            # save a histogram:
            fig, axx = plt.subplots(1)
            axx.hist(eroded_ct_points, bins=100)
            axx.set_ylabel("Counts")
            axx.set_xlabel("HU")
            axx.set_title("Histogram of eroded aorta")
            axx.grid()
            plt.tight_layout()
            fig.savefig(
                os.path.join(self.output_dir, "images/histogram_eroded_aorta.png")
            )

        # Perform the fail-safe check if the method is agatston
        if threshold == "agatston" and eroded_ct_points_mean > agatston_failsafe:
            raise ValueError(
                "The mean HU in the center aorta is {:.0f}, and the agatston method will provide unreliable results (fail-safe threshold is {})".format(
                    eroded_ct_points_mean, agatston_failsafe
                )
            )

        # calc_mask = calc_mask * (aorta_eroded == 0)
        if show_time:
            print("exclude center aorta time: {:.2f} sec".format(time.time() - t0))

        """
        Choose threshold
        """

        if threshold == "adaptive":
            # calc_thres = eroded_ct_points.max()

            # Get aortic CT point to set adaptive threshold
            aorta_ct_points = ct[aorta_mask == 1]

            # equal to one standard deviation to the left of the curve
            quant = 0.158
            quantile_median_dist = np.median(aorta_ct_points) - np.quantile(
                aorta_ct_points, q=quant
            )
            calc_thres = np.median(aorta_ct_points) + quantile_median_dist * num_std

        elif threshold == "agatston":
            calc_thres = 130

            counter = self.slicedSizeCount(aorta_eroded, ct, remove_size, calc_thres)

            # if num_features >= 10:
            #     raise ValueError('Too many pixels above 130 in blood pool, found: {}'.format(num_features))

            if verbose:
                print("{} calc over threshold of {}".format(counter, remove_size))

            if generate_plots:
                # save the statistics of the eroded aorta for reference
                with open(
                    os.path.join(
                        self.output_dir, "metrics/eroded_aorta_statistics.csv"
                    ),
                    "a",
                ) as f:
                    f.write("num calcification blood pool,{}\n".format(counter))
        else:
            try:
                calc_thres = int(threshold)
            except:
                raise ValueError(
                    "Error in threshold value for aortic calcium segmentaiton. \
                    Should be 'adaptive', 'agatston' or int, but got: "
                    + str(threshold)
                )

        """
        Dilate aorta before using threshold to segment calcifications
        """
        t0 = time.time()
        if dilation is not None:
            struct = ndimage.generate_binary_structure(*dilation)
            if dilation_iteration is not None:
                struct = ndimage.iterate_structure(struct, dilation_iteration)
            aorta_dilated = self.slicedDilationOrErosion(
                aorta_mask,
                struct=struct,
                num_iteration=dilation_iteration,
                operation="dilate",
            )

            if show_time:
                print("dilation mask time: {:.2f}".format(time.time() - t0))

        t0 = time.time()
        # make threshold
        calc_mask = np.logical_and(aorta_dilated == 1, ct >= calc_thres)

        if show_time:
            print("find calc time: {:.2f}".format(time.time() - t0))

        t0 = time.time()
        if exclude_mask is not None:
            if dilation_exclude_mask is not None:
                struct_exclude = ndimage.generate_binary_structure(
                    *dilation_exclude_mask
                )
                if dilation_iteration_exclude is not None:
                    struct_exclude = ndimage.iterate_structure(
                        struct_exclude, dilation_iteration_exclude
                    )

                exclude_mask = self.slicedDilationOrErosion(
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
            if verbose:
                print("Excluding calcifications under {} pixels".format(remove_size))

            t0 = time.time()

            if calc_mask.sum() != 0:
                # perform the exclusion on a slice for speed
                arr_slices = self.getSmallestArraySlice(calc_mask, margin=1)
                labels, num_features = ndimage.label(calc_mask[arr_slices])

                counter = 0
                for n in range(1, num_features + 1):
                    idx_tmp = labels == n
                    if idx_tmp.sum() <= remove_size:
                        labels[idx_tmp] = 0
                        counter += 1

                calc_mask[arr_slices] = labels > 0

            if show_time:
                print("Size exclusion time: {:.1f} sec".format(time.time() - t0))

        if not any([return_dilated_mask, return_dilated_exclude]):
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

    def slicedDilationOrErosion(self, input_mask, struct, num_iteration, operation):
        """
        Perform the dilation on the smallest slice that will fit the
        segmentation
        """

        if num_iteration < 1:
            return input_mask

        margin = 2 if num_iteration is None else num_iteration + 1

        x_idx = np.where(input_mask.sum(axis=(1, 2)))[0]
        if len(x_idx) > 0:
            x_start, x_end = max(x_idx[0] - margin, 0), min(
                x_idx[-1] + margin, input_mask.shape[0]
            )

        y_idx = np.where(input_mask.sum(axis=(0, 2)))[0]
        if len(y_idx) > 0:
            y_start, y_end = max(y_idx[0] - margin, 0), min(
                y_idx[-1] + margin, input_mask.shape[1]
            )

        # Don't dilate the aorta at the bifurcation
        z_idx = np.where(input_mask.sum(axis=(0, 1)))[0]
        z_start, z_end = z_idx[0], z_idx[-1]

        if operation == "dilate":
            mask_slice = ndimage.binary_dilation(
                input_mask[x_start:x_end, y_start:y_end, :], structure=struct
            ).astype(np.int8)
        elif operation == "erode":
            mask_slice = ndimage.binary_erosion(
                input_mask[x_start:x_end, y_start:y_end, :], structure=struct
            ).astype(np.int8)

        # copy to not change the originial mask
        output_mask = input_mask.copy()

        # insert dilated mask, but restrain to undilated z_start
        output_mask[x_start:x_end, y_start:y_end, z_start:] = mask_slice[:, :, z_start:]

        return output_mask

    def slicedSizeCount(self, aorta_eroded, ct, remove_size, calc_thres):
        """
        Counts the number of calcifications over the size threshold in the eroded
        aorta on the smallest slice that fits the aorta.
        """
        eroded_calc_mask = np.logical_and(aorta_eroded == 1, ct >= calc_thres)

        if eroded_calc_mask.sum() != 0:
            # Perfom the counts on a slice of the aorta for speed
            arr_slices = self.getSmallestArraySlice(eroded_calc_mask, margin=1)
            labels, num_features = ndimage.label(eroded_calc_mask[arr_slices])
            counter = 0
            for n in range(1, num_features + 1):
                idx_tmp = labels == n
                if idx_tmp.sum() > remove_size:
                    counter += 1

            return counter
        else:
            return 0

    def getSmallestArraySlice(self, input_mask, margin=0):
        """
        Generated the smallest slice that will fit the mask plus the given margin
        and return a touple of slice objects
        """

        x_idx = np.where(input_mask.sum(axis=(1, 2)))[0]
        if len(x_idx) > 0:
            x_start, x_end = max(x_idx[0] - margin, 0), min(
                x_idx[-1] + margin, input_mask.shape[0]
            )

        y_idx = np.where(input_mask.sum(axis=(0, 2)))[0]
        if len(y_idx) > 0:
            y_start, y_end = max(y_idx[0] - margin, 0), min(
                y_idx[-1] + margin, input_mask.shape[1]
            )

        z_idx = np.where(input_mask.sum(axis=(0, 1)))[0]
        if len(z_idx) > 0:
            z_start, z_end = max(z_idx[0] - margin, 0), min(
                z_idx[-1] + margin, input_mask.shape[2]
            )

        return (slice(x_start, x_end), slice(y_start, y_end), slice(z_start, z_end))
    

class AorticCalciumMetrics(InferenceClass):
    """Calculate metrics for the aortic calcifications"""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        calc_mask = inference_pipeline.calc_mask
        spine_mask = inference_pipeline.spine_segmentation.get_fdata().astype(np.int8)
        """     26: "vertebrae_S1",
                27: "vertebrae_L5",
                28: "vertebrae_L4",
                29: "vertebrae_L3",
                30: "vertebrae_L2",
                31: "vertebrae_L1",
                32: "vertebrae_T12",
                33: "vertebrae_T11",
                34: "vertebrae_T10",
                35: "vertebrae_T9",
                36: "vertebrae_T8",
                37: "vertebrae_T7",
                38: "vertebrae_T6",
                39: "vertebrae_T5",
                40: "vertebrae_T4",
                41: "vertebrae_T3",
                42: "vertebrae_T2",
                43: "vertebrae_T1",
                44: "vertebrae_C7",
                45: "vertebrae_C6",
                46: "vertebrae_C5",
                47: "vertebrae_C4",
                48: "vertebrae_C3",
                49: "vertebrae_C2",
                50: "vertebrae_C1","""

        t12_level = np.where((spine_mask == 32).sum(axis=(0, 1)))[0]
        l1_level = np.where((spine_mask == 31).sum(axis=(0, 1)))[0]

        if len(t12_level) != 0 and len(l1_level) != 0:
            sep_plane = round(np.mean([t12_level[0], l1_level[-1]]))
        elif len(t12_level) == 0 and len(l1_level) != 0:
            print("WARNNG: could not locate T12, using L1 only..")
            sep_plane = l1_level[-1]
        elif len(t12_level) != 0 and len(l1_level) == 0:
            print("WARNNG: could not locate L1, using T12 only..")
            sep_plane = t12_level[0]
        else:
            raise ValueError("Could not locate either T12 or L1, aborting..")

        planes = np.zeros_like(spine_mask, dtype=np.int8)
        planes[:, :, sep_plane] = 1
        planes[spine_mask == 32] = 2
        planes[spine_mask == 31] = 3

        inference_pipeline.saveArrToNifti(
            planes,
            os.path.join(
                inference_pipeline.output_dir_segmentation_masks, "t12_plane.nii.gz"
            ),
        )

        inference_pipeline.pix_dims = inference_pipeline.medical_volume.header[
            "pixdim"
        ][1:4]
        # divided with 10 to get in cm
        inference_pipeline.vol_per_pixel = np.prod(inference_pipeline.pix_dims / 10)

        all_regions = {}
        region_names = ["Abdominal", "Thoracic"]

        ct_full = inference_pipeline.medical_volume.get_fdata()

        for i in range(len(region_names)):
            # count statistics for individual calcifications
            if i == 0:
                calc_mask_region = calc_mask[:, :, :sep_plane]
                ct = ct_full[:, :, :sep_plane]
            elif i == 1:
                calc_mask_region = calc_mask[:, :, sep_plane:]
                ct = ct_full[:, :, sep_plane:]

            labelled_calc, num_lesions = ndimage.label(calc_mask_region)

            metrics = {
                "volume": [],
                "mean_hu": [],
                "median_hu": [],
                "max_hu": [],
            }

            if num_lesions == 0:
                metrics["volume"].append(0)
                metrics["mean_hu"].append(0)
                metrics["median_hu"].append(0)
                metrics["max_hu"].append(0)
            else:
                for j in range(1, num_lesions + 1):
                    tmp_mask = labelled_calc == j

                    tmp_ct_vals = ct[tmp_mask]

                    metrics["volume"].append(
                        len(tmp_ct_vals) * inference_pipeline.vol_per_pixel
                    )
                    metrics["mean_hu"].append(np.mean(tmp_ct_vals))
                    metrics["median_hu"].append(np.median(tmp_ct_vals))
                    metrics["max_hu"].append(np.max(tmp_ct_vals))

            # Volume of calcificaitons
            calc_vol = np.sum(metrics["volume"])
            metrics["volume_total"] = calc_vol

            metrics["num_calc"] = num_lesions

            if inference_pipeline.args.threshold == "agatston":
                if num_lesions == 0:
                    metrics["agatston_score"] = 0
                else:
                    metrics["agatston_score"] = self.CalculateAgatstonScore(
                        calc_mask_region, ct, inference_pipeline.pix_dims
                    )

            all_regions[region_names[i]] = metrics

        inference_pipeline.metrics = all_regions

        return {}

    def CalculateAgatstonScore(self, calc_mask_region, ct, pix_dims):
        """
        Original Agatston papers says need to be >= 1mm^2, other papers
        use at least 3 face-linked pixels.
        """

        def get_hu_factor(max_hu):
            # if max_hu ><
            if max_hu < 200:
                factor = 1
            elif 200 <= max_hu < 300:
                factor = 2
            elif 300 <= max_hu < 400:
                factor = 3
            elif max_hu >= 400:
                factor = 4
            else:
                raise ValueError("Could determine factor, got: " + str(max_hu))

            return factor

        # dims are in mm here
        area_per_pixel = pix_dims[0] * pix_dims[1]
        agatston = 0

        for i in range(calc_mask_region.shape[2]):
            tmp_slice = calc_mask_region[:, :, i]
            tmp_ct_slice = ct[:, :, i]

            labelled_calc, num_lesions = ndimage.label(tmp_slice)

            for j in range(1, num_lesions + 1):
                tmp_mask = labelled_calc == j

                tmp_area = tmp_mask.sum() * area_per_pixel
                # exclude if less than 1 mm^2
                if tmp_area <= 1:
                    continue
                else:
                    agatston += tmp_area * get_hu_factor(
                        int(tmp_ct_slice[tmp_mask].max())
                    )

        return agatston
