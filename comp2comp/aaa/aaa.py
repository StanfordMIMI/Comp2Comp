import operator
import os
import zipfile
from pathlib import Path
from time import time
from tkinter import Tcl
from typing import Union

import cv2
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import wget
from totalsegmentator.libs import nostdout

from comp2comp.inference_class_base import InferenceClass


class AortaSegmentation(InferenceClass):
    """Spine segmentation."""

    def __init__(self, save=True):
        super().__init__()
        self.model_name = "totalsegmentator"
        self.save_segmentations = save

    def _infer_raw_geometry(self, input_path: Union[str, Path]):
        """
        Return (raw_shape, raw_affine) for the *raw* CT space.
        - If input_path is a DICOM folder: build affine from DICOM headers.
        - If input_path is a NIfTI file (.nii/.nii.gz): load with nibabel.
        """
        if os.path.isdir(input_path):
            # --- DICOM series case ---
            files = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if not f.startswith(".")
            ]
            ds_list = []
            for fp in files:
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True)
                    ds_list.append(ds)
                except Exception:
                    pass
            if not ds_list:
                raise RuntimeError(f"No readable DICOM slices found in: {input_path}")

            ds0 = ds_list[0]
            rows = int(ds0.Rows)
            cols = int(ds0.Columns)

            # Direction cosines
            iop = [float(x) for x in ds0.ImageOrientationPatient]  # 6 values
            row_cos = np.array(iop[:3], dtype=float)
            col_cos = np.array(iop[3:], dtype=float)
            normal = np.cross(row_cos, col_cos)

            # Pixel spacing (mm)
            ps = [float(x) for x in ds0.PixelSpacing]  # [row_spacing, col_spacing]
            row_spacing = ps[0]
            col_spacing = ps[1]

            # Sort slices along the normal using ImagePositionPatient or InstanceNumber
            def ipp_dot(d):
                if hasattr(d, "ImagePositionPatient"):
                    ipp = np.array(
                        [float(x) for x in d.ImagePositionPatient], dtype=float
                    )
                    return float(np.dot(ipp, normal))
                return 0.0

            if all(hasattr(d, "ImagePositionPatient") for d in ds_list):
                ds_list.sort(key=ipp_dot)
            elif all(hasattr(d, "InstanceNumber") for d in ds_list):
                ds_list.sort(key=lambda d: int(d.InstanceNumber))
            else:
                # fallback: filename order
                ds_list.sort(key=lambda d: getattr(d, "SOPInstanceUID", ""))

            # Slice spacing (use IPP difference if possible)
            if (
                len(ds_list) > 1
                and hasattr(ds_list[0], "ImagePositionPatient")
                and hasattr(ds_list[1], "ImagePositionPatient")
            ):
                ipp0 = np.array(
                    [float(x) for x in ds_list[0].ImagePositionPatient], dtype=float
                )
                ipp1 = np.array(
                    [float(x) for x in ds_list[1].ImagePositionPatient], dtype=float
                )
                slice_step = abs(np.dot((ipp1 - ipp0), normal))
                if slice_step == 0:
                    slice_step = float(
                        getattr(
                            ds0,
                            "SpacingBetweenSlices",
                            getattr(ds0, "SliceThickness", 1.0),
                        )
                    )
            else:
                slice_step = float(
                    getattr(
                        ds0, "SpacingBetweenSlices", getattr(ds0, "SliceThickness", 1.0)
                    )
                )

            # Origin = IPP of first slice
            if hasattr(ds_list[0], "ImagePositionPatient"):
                origin = np.array(
                    [float(x) for x in ds_list[0].ImagePositionPatient], dtype=float
                )
            else:
                origin = np.zeros(3, dtype=float)

            # Build affine (voxel indices i,j,k -> world x,y,z)
            # i: columns (x), j: rows (y), k: slices (z)
            aff = np.eye(4, dtype=float)
            aff[0:3, 0] = col_cos * col_spacing
            aff[0:3, 1] = row_cos * row_spacing
            aff[0:3, 2] = normal * slice_step
            aff[0:3, 3] = origin

            raw_shape = (rows, cols, len(ds_list))
            raw_affine = aff
            return raw_shape, raw_affine

        else:
            # --- NIfTI case ---
            nii = nib.load(str(input_path))
            return tuple(nii.shape), np.array(nii.affine, dtype=float)

    def __call__(self, inference_pipeline):
        print("TRY NEW: DICOM series path is:", inference_pipeline.input_path)
        inference_pipeline.dicom_series_path = inference_pipeline.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        # keep the path to the *raw* NIfTI you feed into the model
        raw_nii_path = os.path.join(
            self.output_dir_segmentations, "converted_dcm.nii.gz"
        )

        seg, mv = self.spine_seg(
            raw_nii_path,
            self.output_dir_segmentations + "spine.nii.gz",
            inference_pipeline.model_dir,
        )

        # NEW: derive raw geometry from the *input* CT (DICOM folder or NIfTI)
        raw_shape, raw_affine = self._infer_raw_geometry(inference_pipeline.input_path)
        inference_pipeline.raw_shape = raw_shape
        inference_pipeline.raw_affine = raw_affine

        # Keep processed-space metadata for pixel→mm conversion & mapping
        inference_pipeline.proc_affine = mv.affine
        inference_pipeline.proc_zooms = mv.header.get_zooms()
        inference_pipeline.proc_shape = mv.shape

        print("DEBUG raw_shape:", inference_pipeline.raw_shape)
        print("DEBUG proc_shape:", inference_pipeline.proc_shape)

        seg = seg.get_fdata()
        medical_volume = mv.get_fdata()

        axial_masks = []
        ct_image = []

        for i in range(seg.shape[2]):
            axial_masks.append(seg[:, :, i])

        for i in range(medical_volume.shape[2]):
            ct_image.append(medical_volume[:, :, i])

        # Save input axial slices to pipeline
        inference_pipeline.ct_image = ct_image

        # Save aorta masks to pipeline
        inference_pipeline.axial_masks = axial_masks

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
                "nnUNet/3d_fullres/Task253_Aorta/nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1",
            )
        )
        print(download_dir)
        fold_0_path = download_dir / "fold_0"
        if not os.path.exists(fold_0_path):
            download_dir.mkdir(parents=True, exist_ok=True)
            wget.download(
                "https://huggingface.co/AdritRao/aaa_test/resolve/main/fold_0.zip",
                out=os.path.join(download_dir, "fold_0.zip"),
            )
            with zipfile.ZipFile(
                os.path.join(download_dir, "fold_0.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(download_dir)
            os.remove(os.path.join(download_dir, "fold_0.zip"))
            wget.download(
                "https://huggingface.co/AdritRao/aaa_test/resolve/main/plans.pkl",
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

        print(self.model_dir)

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [253]

        self.setup_nnunet_c2c(model_dir)
        self.download_spine_model(model_dir)

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

        seg_data = seg.get_fdata()
        seg = nib.Nifti1Image(seg_data, seg.affine, seg.header)

        return seg, img


class AortaDiameter(InferenceClass):
    def __init__(self):
        super().__init__()

    def normalize_img(self, img: np.ndarray) -> np.ndarray:
        return (img - img.min()) / (img.max() - img.min())

    def __call__(self, inference_pipeline):
        axial_masks = (
            inference_pipeline.axial_masks
        )  # list of 2D masks (processed space)
        ct_img = inference_pipeline.ct_image  # list/array of slices (processed space)

        # output dirs
        output_dir = inference_pipeline.output_dir
        output_dir_slices = os.path.join(output_dir, "images/slices/")
        output_dir_summary = os.path.join(output_dir, "images/summary/")
        os.makedirs(output_dir_slices, exist_ok=True)
        os.makedirs(output_dir_summary, exist_ok=True)

        # --- Affine-based mapping: processed -> raw ---
        # processed image metadata (where we actually measure pixels)
        proc_affine = np.array(inference_pipeline.proc_affine)
        proc_zooms = tuple(inference_pipeline.proc_zooms)
        proc_shape = tuple(inference_pipeline.proc_shape)  # (H, W, Z_proc)
        Hp, Wp, Zp = proc_shape

        # raw CT reference space (target indexing for CSV)
        raw_shape = tuple(inference_pipeline.raw_shape)  # e.g., (512, 512, 480)
        raw_aff = np.array(inference_pipeline.raw_affine)  # 4x4
        Z_raw = raw_shape[2]

        print(f"[DEBUG] raw_shape={raw_shape} -> Z_raw={Z_raw}")
        print(f"[DEBUG] proc_shape={proc_shape} -> Zp={Zp}")

        # precompute inverse affine for raw (do this once)
        inv_raw_aff = np.linalg.inv(raw_aff)

        # use in-plane mm-per-pixel from the processed image (safer if x!=y)
        RATIO_PIXEL_TO_MM = float((proc_zooms[0] + proc_zooms[1]) / 2.0)

        # pick the center of the processed image in XY to define the slice plane
        cx = (Wp - 1) / 2.0
        cy = (Hp - 1) / 2.0

        print("Processed shape (H,W,Z):", proc_shape)
        print("Raw shape (H,W,Z):", raw_shape)

        # maps RAW index -> diameter (cm)
        diameter_cm_by_raw_index = {}

        for i in range(Zp):  # i is processed-space slice index
            # ensure binary mask
            mask = (axial_masks[i] > 0).astype("uint8")
            if mask.max() == 0:
                continue  # no aorta on this processed slice

            # ---- map processed slice i to raw slice index k_raw ----
            proc_voxel = np.array([cx, cy, i, 1.0])  # center of slice
            world_xyz1 = proc_affine @ proc_voxel  # world coord
            raw_ijk1 = inv_raw_aff @ world_xyz1  # raw voxel coord
            k_raw = int(round(raw_ijk1[2]))  # raw z-index

            # clamp into valid range
            if k_raw < 0 or k_raw >= Z_raw:
                continue

            # ---- measure diameter on the processed slice ----
            img = ct_img[i]
            img = np.clip(img, -300, 1800)
            img = self.normalize_img(img) * 255.0
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img = np.tile(img, (1, 1, 3))

            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            # largest component
            areas = [cv2.contourArea(c) for c in contours]
            contours = contours[int(np.argmax(areas))]

            # overlay (cosmetic)
            back = img.copy()
            cv2.drawContours(back, [contours], 0, (0, 255, 0), -1)
            img = cv2.addWeighted(img, 0.75, back, 0.25, 0)

            ellipse = cv2.fitEllipse(contours)
            (xc, yc), (d1, d2), angle = ellipse
            cv2.ellipse(img, ellipse, (0, 255, 0), 1)
            cv2.circle(img, (int(xc), int(yc)), 5, (0, 0, 255), -1)

            max(d1, d2) / 2.0
            rminor = min(d1, d2) / 2.0

            # diameter from minor axis
            pixel_length = rminor * 2.0
            diameter_mm = round(pixel_length * RATIO_PIXEL_TO_MM)
            diameter_cm = diameter_mm / 10.0

            # if multiple processed slices map to the same raw index (rare), keep the max
            prev = diameter_cm_by_raw_index.get(k_raw)
            if (prev is None) or (diameter_cm > prev):
                diameter_cm_by_raw_index[k_raw] = diameter_cm

            # Save an image for quick QC (tagged with RAW index)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w, _ = img.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = min(w, h) / (25 / 0.03)
            cv2.putText(
                img,
                f"CT raw slice index: {k_raw}",
                (10, 40),
                font,
                fontScale,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"Diameter (cm): {diameter_cm}",
                (10, 70),
                font,
                fontScale,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(os.path.join(output_dir_slices, f"slice_raw_{k_raw}.png"), img)

        # --- Summary plot vs RAW index ---
        if diameter_cm_by_raw_index:
            xs = sorted(diameter_cm_by_raw_index.keys())
            ys = [diameter_cm_by_raw_index[k] for k in xs]
            plt.bar(xs, ys)
            plt.title(r"$\bf{Diameter}$" + " " + r"$\bf{Progression}$")
            plt.xlabel("CT raw slice index (0-based)")
            plt.ylabel("Diameter (cm)")
            plt.savefig(os.path.join(output_dir_summary, "diameter_graph.png"), dpi=500)
            plt.close()

        # --- Max diameter (by RAW index) ---
        if diameter_cm_by_raw_index:
            max_raw_idx = max(
                diameter_cm_by_raw_index.items(), key=operator.itemgetter(1)
            )[0]
            print(
                "Max raw index:",
                max_raw_idx,
                "diameter_cm:",
                diameter_cm_by_raw_index[max_raw_idx],
            )
            inference_pipeline.max_diameter = diameter_cm_by_raw_index[max_raw_idx]
        else:
            max_raw_idx = None
            inference_pipeline.max_diameter = float("nan")

        # --- MP4 (frames may be fewer than Z_raw; only saved slices with aorta) ---
        image_files = [
            os.path.join(output_dir_slices, f)
            for f in Tcl().call("lsort", "-dict", os.listdir(output_dir_slices))
            if f.endswith(".png")
        ]
        if image_files:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                image_files, fps=20
            )
            clip.write_videofile(os.path.join(output_dir_summary, "aaa.mp4"))

        # --- CSV over ALL raw slices: 0..Z_raw-1 (NaN where no aorta) ---
        metrics_dir = os.path.join(inference_pipeline.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        rows = []
        for k in range(Z_raw):
            d_cm = diameter_cm_by_raw_index.get(k, np.nan)
            rows.append(
                {
                    "ct_slice": k,
                    "diameter_cm": d_cm,
                    "diameter_mm": (d_cm * 10.0) if not np.isnan(d_cm) else np.nan,
                }
            )
        df = pd.DataFrame(rows, columns=["ct_slice", "diameter_cm", "diameter_mm"])
        df.to_csv(os.path.join(metrics_dir, "aorta_diameters.csv"), index=False)

        return {}


class AortaMetricsSaver(InferenceClass):
    """Save metrics to a CSV file."""

    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        """Save metrics to a CSV file."""
        self.max_diameter = inference_pipeline.max_diameter
        self.dicom_series_path = inference_pipeline.dicom_series_path
        self.output_dir = inference_pipeline.output_dir
        self.csv_output_dir = os.path.join(self.output_dir, "metrics")
        if not os.path.exists(self.csv_output_dir):
            os.makedirs(self.csv_output_dir, exist_ok=True)
        self.save_results()
        return {}

    def save_results(self):
        """Save results to a CSV file."""
        _, filename = os.path.split(self.dicom_series_path)
        data = [[filename, str(self.max_diameter)]]
        df = pd.DataFrame(data, columns=["Filename", "Max Diameter"])
        df.to_csv(os.path.join(self.csv_output_dir, "aorta_metrics.csv"), index=False)
