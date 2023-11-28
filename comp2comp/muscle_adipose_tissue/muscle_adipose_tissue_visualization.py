"""
@author: louisblankemeier
"""

import os
from pathlib import Path

import numpy as np

from comp2comp.inference_class_base import InferenceClass
from comp2comp.visualization.detectron_visualizer import Visualizer


class MuscleAdiposeTissueVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

        self._spine_colors = {
            "L5": [255, 0, 0],
            "L4": [0, 255, 0],
            "L3": [255, 255, 0],
            "L2": [255, 128, 0],
            "L1": [0, 255, 255],
            "T12": [255, 0, 255],
        }

        self._muscle_fat_colors = {
            "muscle": [255, 136, 133],
            "imat": [154, 135, 224],
            "vat": [140, 197, 135],
            "sat": [246, 190, 129],
        }

        self._SPINE_TEXT_OFFSET_FROM_TOP = 10.0
        self._SPINE_TEXT_OFFSET_FROM_RIGHT = 63.0
        self._SPINE_TEXT_VERTICAL_SPACING = 14.0

        self._MUSCLE_FAT_TEXT_HORIZONTAL_SPACING = 40.0
        self._MUSCLE_FAT_TEXT_VERTICAL_SPACING = 14.0
        self._MUSCLE_FAT_TEXT_OFFSET_FROM_TOP = 22.0
        self._MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT = 181.0

    def __call__(self, inference_pipeline, images, results):
        self.output_dir = inference_pipeline.output_dir
        self.dicom_file_names = inference_pipeline.dicom_file_names
        # if spine is an attribute of the inference pipeline, use it
        if not hasattr(inference_pipeline, "spine"):
            spine = False
        else:
            spine = True
            self.spine_masks = inference_pipeline.spine_masks

        for i, (image, result) in enumerate(zip(images, results)):
            # now, result is a dict with keys for each tissue
            dicom_file_name = self.dicom_file_names[i]
            self.save_binary_segmentation_overlay(image, result, dicom_file_name, spine)
        # pass along for next class in pipeline
        return {"results": results}

    def save_binary_segmentation_overlay(self, image, result, dicom_file_name, spine):
        file_name = dicom_file_name + ".png"
        img_in = image
        assert img_in.shape == (512, 512), "Image shape is not 512 x 512"

        img_in = np.clip(img_in, -300, 1800)
        img_in = self.normalize_img(img_in) * 255.0

        # Create the folder to save the images
        images_base_path = Path(self.output_dir) / "images"
        images_base_path.mkdir(exist_ok=True)

        text_start_vertical_offset = self._MUSCLE_FAT_TEXT_OFFSET_FROM_TOP

        img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
        img_rgb = np.tile(img_in, (1, 1, 3))

        vis = Visualizer(img_rgb)
        vis.draw_text(
            text="Density (HU)",
            position=(
                img_in.shape[1] - self._MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT - 63,
                text_start_vertical_offset,
            ),
            color=[1, 1, 1],
            font_size=9,
            horizontal_alignment="left",
        )
        vis.draw_text(
            text="Area (CM²)",
            position=(
                img_in.shape[1] - self._MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT - 63,
                text_start_vertical_offset + self._MUSCLE_FAT_TEXT_VERTICAL_SPACING,
            ),
            color=[1, 1, 1],
            font_size=9,
            horizontal_alignment="left",
        )

        if spine:
            spine_color = np.array(self._spine_colors[dicom_file_name]) / 255.0
            vis.draw_box(
                box_coord=(1, 1, img_in.shape[0] - 1, img_in.shape[1] - 1),
                alpha=1,
                edge_color=spine_color,
            )
            # draw the level T12 - L5 in the upper left corner
            if dicom_file_name == "T12":
                position = (40, 15)
            else:
                position = (30, 15)
            vis.draw_text(
                text=dicom_file_name, position=position, color=spine_color, font_size=24
            )
            vis.draw_binary_mask(
                self.spine_masks[dicom_file_name],
                color=spine_color,
                alpha=0.9,
                area_threshold=0,
            )

        for idx, tissue in enumerate(result.keys()):
            alpha_val = 0.9
            color = np.array(self._muscle_fat_colors[tissue]) / 255.0
            edge_color = color
            mask = result[tissue]["mask"]

            vis.draw_binary_mask(
                mask,
                color=color,
                edge_color=edge_color,
                alpha=alpha_val,
                area_threshold=0,
            )

            hu_val = round(result[tissue]["Hounsfield Unit"])
            area_val = round(result[tissue]["Cross-sectional Area (cm^2)"])

            vis.draw_text(
                text=tissue,
                position=(
                    mask.shape[1]
                    - self._MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT
                    + self._MUSCLE_FAT_TEXT_HORIZONTAL_SPACING * (idx + 1),
                    text_start_vertical_offset - self._MUSCLE_FAT_TEXT_VERTICAL_SPACING,
                ),
                color=color,
                font_size=9,
                horizontal_alignment="center",
            )

            vis.draw_text(
                text=hu_val,
                position=(
                    mask.shape[1]
                    - self._MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT
                    + self._MUSCLE_FAT_TEXT_HORIZONTAL_SPACING * (idx + 1),
                    text_start_vertical_offset,
                ),
                color=color,
                font_size=9,
                horizontal_alignment="center",
            )
            vis.draw_text(
                text=area_val,
                position=(
                    mask.shape[1]
                    - self._MUSCLE_FAT_TEXT_OFFSET_FROM_RIGHT
                    + self._MUSCLE_FAT_TEXT_HORIZONTAL_SPACING * (idx + 1),
                    text_start_vertical_offset + self._MUSCLE_FAT_TEXT_VERTICAL_SPACING,
                ),
                color=color,
                font_size=9,
                horizontal_alignment="center",
            )

        vis_obj = vis.get_output()
        vis_obj.save(os.path.join(images_base_path, file_name))

    def normalize_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize the image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Normalized image.
        """
        return (img - img.min()) / (img.max() - img.min())
