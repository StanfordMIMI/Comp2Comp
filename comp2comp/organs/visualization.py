#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from comp2comp.inference_class_base import InferenceClass
from comp2comp.organs.visualization_utils import (
    generate_liver_spleen_pancreas_report,
    generate_slice_images,
)


class LiverSpleenPancreasVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

        self.unit_dict = {
            "Volume": r"$\mathregular{cm^3}$",
            "Mean": "HU",
            "Median": "HU",
        }

        self.class_nums = [1, 5, 10]
        self.organ_names = ["liver", "spleen", "pancreas"]

    def __call__(self, inference_pipeline):

        self.output_dir = inference_pipeline.output_dir
        self.output_dir_images_organs = os.path.join(self.output_dir, "images/")
        inference_pipeline.output_dir_images_organs_organs_organs = self.output_dir_images_organs

        if not os.path.exists(self.output_dir_images_organs):
            os.makedirs(self.output_dir_images_organs)

        inference_pipeline.medical_volume_arr = np.flip(inference_pipeline.medical_volume.get_fdata(), axis=1)
        inference_pipeline.segmentation_arr = np.flip(inference_pipeline.segmentation.get_fdata(), axis=1)

    
        inference_pipeline.pix_dims = inference_pipeline.medical_volume.header["pixdim"][1:4]
        inference_pipeline.vol_per_pixel = np.prod(
            inference_pipeline.pix_dims / 10
        )  # mm to cm for having ml/pixel.

        self.organ_metrics = generate_slice_images(
            inference_pipeline.medical_volume_arr,
            inference_pipeline.segmentation_arr,
            self.class_nums,
            self.unit_dict,
            inference_pipeline.vol_per_pixel,
            inference_pipeline.pix_dims,
            self.output_dir_images_organs,
            fontsize=24,
        )

        inference_pipeline.organ_metrics = self.organ_metrics

        generate_liver_spleen_pancreas_report(self.output_dir_images_organs, self.organ_names)

        return {}


class LiverSpleenPancreasMetricsPrinter(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        results = inference_pipeline.organ_metrics
        organs = list(results.keys())

        name_dist = max([len(o) for o in organs])
        metrics = []
        for k in results[list(results.keys())[0]].keys():
            if k != "Organ":
                metrics.append(k)

        units = ["cm^3", "HU", "HU"]

        header = "{:<" + str(name_dist + 4) + "}" + ("{:<" + str(15) + "}") * len(metrics)
        header = header.format("Organ", *[m + "(" + u + ")" for m, u in zip(metrics, units)])

        base_print = "{:<" + str(name_dist + 4) + "}" + ("{:<" + str(15) + ".0f}") * len(metrics)

        print("\n")
        print(header)

        for organ in results.values():
            line = base_print.format(*organ.values())
            print(line)

        print("\n")

        output_dir = inference_pipeline.output_dir
        self.output_dir_metrics_organs = os.path.join(output_dir, "metrics/")

        if not os.path.exists(self.output_dir_metrics_organs):
            os.makedirs(self.output_dir_metrics_organs)

        header = ",".join(["Organ"] + [m + "(" + u + ")" for m, u in zip(metrics, units)]) + "\n"
        with open(
            os.path.join(self.output_dir_metrics_organs, "liver_spleen_pancreas_metrics.csv"), "w"
        ) as f:
            f.write(header)

            for organ in results.values():
                line = ",".join([str(v) for v in organ.values()]) + "\n"
                f.write(line)

        return {}
