from detectron2.utils.visualizer import Visualizer, VisImage
from pathlib import Path
import numpy as np
import os

_COLORS = np.array(
    [
        1.000, 0.000, 0.000,
        0.000, 0.000, 1.000, 
        0.000, 1.000, 0.000, 
        1.000, 1.000, 0.000,
        0.000, 1.000, 1.000,
        1.000, 0.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

_COLOR_MAP = {
    "L5": 0,
    "L4": 1,
    "L3": 2,
    "L2": 3,
    "L1": 4,
    "T12": 5
}

def save_binary_segmentation_overlay(img_in, mask, base_path, file_name, centroids = None, figure_text_key = None):
    # Window image to retain most information
    img_in = np.clip(img_in, -300, 1800)

    # Normalize the image
    img_in = normalize_img(img_in) * 255.0

    # Create the folder to save the images
    images_base_path = Path(base_path) / "images"
    images_base_path.mkdir(exist_ok=True)
    
    text_spacing = 25.0
    text_start_offset = 10.0

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))
    vis = Visualizer(img_rgb)
    for num_bin_masks in range(1, mask.shape[2]):
        vis.draw_binary_mask(mask[:, :, num_bin_masks].astype(int), color = _COLORS[num_bin_masks - 1], alpha=0.3)
        if centroids:
            #print(centroids)
            vis.draw_line(x_data = (0, mask.shape[1] - 1), y_data = (centroids[num_bin_masks - 1], centroids[num_bin_masks - 1]), color = _COLORS[num_bin_masks - 1], linestyle="dashed", linewidth = 0.5)
        else:
            #print(figure_text_key)
            vis.draw_box(box_coord = (1, 1, mask.shape[0] - 1, mask.shape[1] - 1), alpha = 1, edge_color = _COLORS[_COLOR_MAP[file_name.split('_')[0]]])
            if figure_text_key:
                vis.draw_text(text="HU: " + figure_text_key[file_name.split('_')[0]][(num_bin_masks - 1) * 2], position=(mask.shape[1] - 48, text_spacing * (num_bin_masks - 1) + text_start_offset), color=_COLORS[num_bin_masks - 1], font_size = 7)
                vis.draw_text(text="AREA: " + figure_text_key[file_name.split('_')[0]][((num_bin_masks - 1) * 2) + 1], position=(mask.shape[1] - 48, text_spacing * (num_bin_masks - 1) + text_start_offset + (text_spacing / 2)), color=_COLORS[num_bin_masks - 1], font_size = 7)
    vis_obj = vis.get_output()
    vis_obj.save(os.path.join(images_base_path, file_name))

def normalize_img(img):
    return (img - img.min()) / (img.max() - img.min())