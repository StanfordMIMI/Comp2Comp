import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

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

        for i, (image, result) in enumerate(zip(images, results, strict=True)):
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
            vis.draw_text(text=dicom_file_name, position=position, color=spine_color, font_size=24)

        for idx, tissue in enumerate(result.keys()):
            alpha_val = 0.9
            color = np.array(self._muscle_fat_colors[tissue]) / 255.0
            edge_color = color
            mask = result[tissue]["mask"]

            vis.draw_binary_mask(
                mask, color=color, edge_color=edge_color, alpha=alpha_val, area_threshold=0
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


class SpineMuscleAdiposeTissueReport(InferenceClass):
    """Spine muscle adipose tissue report class."""

    def __init__(self):
        super().__init__()
        self.image_files = [
            "spine_coronal.png",
            "spine_sagittal.png",
            "T12.png",
            "L3.png",
            "L1.png",
            "L4.png",
            "L2.png",
            "L5.png",
        ]

    def __call__(self, inference_pipeline):
        image_dir = Path(inference_pipeline.output_dir) / "images"
        self.generate_panel(image_dir)

    def generate_panel(self, image_dir: Union[str, Path]):
        """Generate panel.
        Args:
            image_dir (Union[str, Path]): Path to the image directory.
        """
        image_files = [os.path.join(image_dir, path) for path in self.image_files]
        im_cor = Image.open(image_files[0])
        im_sag = Image.open(image_files[1])
        im_cor_width = int(im_cor.width / im_cor.height * 512)
        width = (8 + im_cor_width + 8) + ((512 + 8) * 3)
        height = 1048
        new_im = Image.new("RGB", (width, height))

        index = 2
        for i in range(8 + im_cor_width + 8, width, 520):
            for j in range(8, height, 520):
                im = Image.open(image_files[index])
                im.thumbnail((512, 512))
                new_im.paste(im, (i, j))
                index += 1
                im.close()

        im_cor.thumbnail((im_cor_width, 512))
        new_im.paste(im_cor, (8, 8))
        im_sag.thumbnail((im_cor_width, 512))
        new_im.paste(im_sag, (8, 528))
        new_im.save(os.path.join(image_dir, "report.png"))
        im_cor.close()
        im_sag.close()
        new_im.close()
