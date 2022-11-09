from detectron2.utils.visualizer import Visualizer, VisImage
from pathlib import Path
import numpy as np
import os

_COLORS = np.array(
    [
        1.000, 0.000, 0.000,
        0.000, 0.000, 1.000, 
        0.000, 1.000, 0.000, 
        1.000, 1.000, 0.000
    ]
).astype(np.float32).reshape(-1, 3)

def save_binary_segmentation_overlay(img_in, mask, base_path, file_name):
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)
    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))
    vis = Visualizer(img_rgb)
    for num_bin_masks in range(1, mask.shape[2]):
        vis.draw_binary_mask(mask[:, :, num_bin_masks].astype(int), color = _COLORS[num_bin_masks - 1])
    vis.get_output().save(os.path.join(images_base_path, file_name))