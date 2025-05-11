import io
import os
import shutil

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.transforms import Bbox
import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from PIL import Image

# color map used for segmnetations
color_array = plt.get_cmap('Set1')(range(10))
color_array = np.concatenate( (np.array([[0,0,0,0]]),color_array[:-1,:]), axis=0)
map_object_seg = ListedColormap(name='segmentation_cmap',colors=color_array)

def createMipPlot(
    ct: NDArray,
    calc_mask: NDArray,
    aorta_mask: NDArray,
    plane_mask: NDArray,
    HU_val: int,
    pix_size: NDArray,
    metrics: dict,
    save_root: str
    ) -> None:
    '''
    Create a MIP projection in the frontal and side plane with
    the calcication overlayed. The text box is generated seperately 
    and then resampled to the MIP
    '''

    
    '''
    Generate MIP image 
    '''
    # Create transparent hot cmap
    hot = plt.get_cmap('hot', 256)
    hot_colors = hot(np.linspace(0, 1, 256))
    hot_colors[0, -1] = 0  
    hot_transparent = ListedColormap(hot_colors)

    fig, axx = plt.subplots(figsize=(12,12), dpi=300)
    fig.patch.set_facecolor('black')
    axx.set_facecolor('black')

    # Create the frontal projection
    thres = 300

    ct_proj = np.flip(np.transpose(ct.max(axis=1)), axis=0)
    ct_proj[ct_proj < thres] = thres

    calc_mask_proj = np.flip(np.transpose(calc_mask.sum(axis=1)), axis=0)
    # normalize both views for the heat map
    if not calc_mask_proj.max() == 0:
        calc_mask_proj = calc_mask_proj/calc_mask_proj.max()

    aorta_mask_proj = np.flip(np.transpose(aorta_mask.max(axis=1)), axis=0)*2
    plane_mask_proj = np.where(np.flip(np.transpose( (plane_mask == 1).max(axis=(0,1))), axis=0))[0][0]

    # Create the side projection
    ct_proj_side = np.flip(np.transpose(ct.max(axis=0)), axis=0)
    ct_proj_side[ct_proj_side < thres] = thres

    calc_mask_proj_side = np.flip(np.transpose(calc_mask.sum(axis=0)), axis=0)
    # normalize both views for the heat map
    if not calc_mask_proj_side.max() == 0:
        calc_mask_proj_side = calc_mask_proj_side/calc_mask_proj_side.max()
        
    aorta_mask_proj_side = np.flip(np.transpose(aorta_mask.max(axis=0)), axis=0)*2

    # Concatenate together
    ct_proj_all = np.hstack([ct_proj, ct_proj_side])
    calc_mask_proj_all = np.hstack([calc_mask_proj, calc_mask_proj_side])
    aorta_mask_proj_all = np.hstack([aorta_mask_proj, aorta_mask_proj_side])

    # Plot the results
    axx.imshow(ct_proj_all, cmap='gray', vmin=thres, vmax=1600, aspect=pix_size[2]/pix_size[0], alpha=1)

    # Aorta mask and calcification
    aorta_im = axx.imshow(aorta_mask_proj_all, cmap=map_object_seg, zorder=1, 
                        vmin=0, vmax=10, aspect=pix_size[2]/pix_size[0], alpha=0.6, interpolation='nearest')
    calc_im = axx.imshow(calc_mask_proj_all, cmap=hot_transparent, aspect=pix_size[2]/pix_size[0], alpha=1, 
                        interpolation='nearest', zorder=2)

    # Ab and Th separating plane
    axx.plot([0, ct_proj_all.shape[1]], [plane_mask_proj, plane_mask_proj], 
            color=map_object_seg(3), lw=0.8, alpha=0.8, zorder=0)
    axx.text(30, plane_mask_proj - 8, 'Thoracic', color=map_object_seg(3), 
            va='center', ha='left', alpha=0.8, fontsize=10)
    axx.text(30, plane_mask_proj + 8, 'Abdominal', color=map_object_seg(3), 
            va='center', ha='left', alpha=0.8, fontsize=10)

    axx.set_xticks([])
    axx.set_ylabel('Slice number', color='white', fontsize=10)
    axx.tick_params(axis='y', colors='white', labelsize=10)

    # extend black background 
    axx.set_xlim(0, ct_proj_all.shape[1])

    # wrap plot in Image
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    # Create a new bounding box with padding only on the left
    # The bbox coordinates are [left, bottom, right, top]
    custom_bbox = Bbox([[tight_bbox.x0 - 0.05, tight_bbox.y0-0.07],  # Add 0.5 inches to the left only
                        [tight_bbox.x1, tight_bbox.y1]])
    buf_mip = io.BytesIO()
    fig.savefig(buf_mip, bbox_inches=custom_bbox, pad_inches=0, dpi=300, format='png')
    plt.close(fig)
    buf_mip.seek(0)
    image_mip = Image.open(buf_mip)

    '''
    Generate the text box 
    '''
    spacing = 23
    indent = 1
    text_box_x_offset = 20
    report_text = []
    report_text.append(r'$\bf{Calcification\ Report}$')
    
    for i, (region, region_metrics) in enumerate(metrics.items()):
        report_text.append(region)    
        report_text.append('{:<{}}{:<{}}{}'.format('',indent, 'Total number:', spacing, region_metrics['num_calc']))
        report_text.append('{:<{}}{:<{}}{:.3f}'.format('',indent,'Total volume (cm³):', spacing, region_metrics['volume_total']))
        report_text.append('{:<{}}{:<{}}{:.3f}{}{:.3f}'.format('',indent,'Mean volume (cm³):', spacing, 
                                                    np.mean(region_metrics["volume"]),r'$\pm$',np.std(region_metrics["volume"])))
        report_text.append('{:<{}}{:<{}}{:.1f}{}{:.1f}'.format('',indent,'Median HU:', spacing, 
                                                    np.mean(region_metrics["median_hu"]),r'$\pm$',np.std(region_metrics["median_hu"])))
        report_text.append('{:<{}}{:<{}}{:.3f}'.format('',indent,'% Volume calcified:', spacing, 
                                                    np.mean(region_metrics["perc_calcified"])))
        
        if 'agatston_score' in region_metrics:
            report_text.append('{:<{}}{:<{}}{:.0f}'.format('',indent,'Agatston:', spacing, region_metrics['agatston_score']))
        
        report_text.append('\n')
        
    report_text.append('{:<{}}{:<{}}{}'.format('',indent, 'Threshold (HU):', spacing, HU_val))

    fig_t, axx_t = plt.subplots(figsize=(5.85,5.85), dpi=300)
    fig_t.patch.set_facecolor('black')
    axx_t.set_facecolor('black')

    axx_t.imshow(np.ones((100,65)), cmap='gray')
    bbox_props = dict(boxstyle="round", facecolor="gray", alpha=0.5)

    text_obj = axx_t.text(
        x=0.3,
        y=1,
        s='\n'.join(report_text),
        color="white",
        # fontsize=12.12 * font_scale,
        fontsize=15.7,
        family="monospace",
        bbox=bbox_props,
        va="top",
        ha="left",
    )

    axx_t.set_aspect(0.69) 
    axx_t.axis('off')
    fig.canvas.draw_idle()
    plt.tight_layout()
    
    # wrap text box in Image
    tight_bbox_text = fig_t.get_tightbbox(fig_t.canvas.get_renderer())
    custom_bbox_text = Bbox([[tight_bbox_text.x0 - 0.05, tight_bbox_text.y0-0.14],  # Add 0.5 inches to the left only
                        [tight_bbox_text.x1+0.15, tight_bbox_text.y1+0.05]])
    buf_text = io.BytesIO()
    fig_t.savefig(buf_text, bbox_inches=custom_bbox_text, pad_inches=0, dpi=300, format='png')
    plt.close(fig_t)
    buf_text.seek(0)
    image_text = Image.open(buf_text)
    # Ensure the same boarder on the text box
    image_text = crop_and_pad_image(image_text, pad_percent=0.015)

    # Match the width to the projection image
    aspect_ratio = image_text.height / image_text.width
    adjusted_width = int(image_mip.height / aspect_ratio)

    image_text_resample = image_text.resize([adjusted_width, image_mip.height], Image.LANCZOS)
    
    # Merge into one image
    result = Image.new("RGB", (image_mip.width + image_text_resample.width, image_mip.height))
    result.paste(im=image_mip, box=(0, 0))
    result.paste(im=image_text_resample, box=(image_mip.width, 0))

    # create path and save
    path = os.path.join(save_root, 'sub_figures')
    os.makedirs(path, exist_ok=True)
    result.save(os.path.join(path,'projection.png'), dpi=(300, 300))
    

def crop_and_pad_image(image, pad_percent=0.025, pad_color=(0, 0, 0, 255)):
    # Ensure image has alpha channel
    image = image.convert('RGBA')
    
    # Create a binary mask: 255 for non-black, 0 for black
    mask = image.convert('RGB').point(lambda p: 255 if p != 0 else 0).convert('L')
    bbox = mask.getbbox()

    if not bbox:
        return image  # No content found; return original
    
    # Crop the image
    cropped = image.crop(bbox)

    # Calculate padding
    width, height = cropped.size
    pad = int(width * pad_percent)

    # Add padding
    padded = Image.new('RGBA', (width + pad * 2, height + pad * 2), pad_color)
    padded.paste(cropped, (pad, pad))

    return padded


def createCalciumMosaic(
    ct: NDArray,
    calc_mask: NDArray,
    aorta_mask: NDArray,
    spine_mask: NDArray,
    pix_size: NDArray,
    save_root: str,
    mosaic_type: str = 'all'
    ) -> None:
    '''
    Wrapper function that calls different functions for creating the mosaic 
    depending on the "mosaic_type".
    '''
    if mosaic_type == 'all':
        createCalciumMosaicAll(
            ct,
            calc_mask,
            aorta_mask,
            pix_size,
            save_root,
        )
    elif mosaic_type == 'vertebrae':
        createCalciumMosaicVertebrae(
            ct,
            calc_mask,
            aorta_mask,
            spine_mask,
            pix_size,
            save_root,
        )
    else:
        raise ValueError('mosaic_type not recognized, got: ' + str(mosaic_type))

def createCalciumMosaicAll(
    ct: NDArray,
    calc_mask: NDArray,
    aorta_mask_dil: NDArray,
    pix_size: NDArray,
    save_root: str,
    ) -> None:
    
    calc_idx = np.where( calc_mask.sum(axis=(0,1)) )[0]

    per_row = 15

    if len(calc_idx) < per_row:
        per_row = len(calc_idx)

    # target size of 60 mm crop size rounded to nearest multiple of 2
    crop_size = round(60 / pix_size[0])
    crop_size = 2*round(crop_size/2)

    ct_crops = []
    mask_crops = []
    aorta_dil_crops = []

    ct_tmp = []
    mask_tmp = []
    aorta_dil_tmp = []

    for i, idx in enumerate(calc_idx[::-1]):
        ct_slice = np.flip(np.transpose(ct[:,:,idx]), axis=(0,1))
        mask_slice = np.flip(np.transpose(calc_mask[:,:,idx]), axis=(0,1))
        aorta_slice = np.flip(np.transpose(aorta_mask_dil[:,:,idx]), axis=(0,1))
                
        x_center = np.where(aorta_slice.sum(axis=1))[0]
        x_center = x_center[len(x_center)//2] 
        
        y_center = np.where(aorta_slice.sum(axis=0))[0]
        y_center = y_center[len(y_center)//2] 

        ct_tmp.append(ct_slice[
            x_center-crop_size//2:x_center+crop_size//2,
            y_center-crop_size//2:y_center+crop_size//2,
                                ])
        
        mask_tmp.append(mask_slice[
            x_center-crop_size//2:x_center+crop_size//2,
            y_center-crop_size//2:y_center+crop_size//2,
                                ])
        
        aorta_dil_tmp.append(aorta_slice[
            x_center-crop_size//2:x_center+crop_size//2,
            y_center-crop_size//2:y_center+crop_size//2,
                                ])
        
        if (i+1) % per_row == 0:
            # print('got here')
            ct_crops.append(np.hstack(ct_tmp))
            mask_crops.append(np.hstack(mask_tmp))
            aorta_dil_crops.append(np.hstack(aorta_dil_tmp))

            ct_tmp = []
            mask_tmp = []
            aorta_dil_tmp = []

    if len(ct_tmp) > 0:
        pad_len = per_row*crop_size - len(ct_tmp)*crop_size
        
        ct_crops.append(np.pad(np.hstack(ct_tmp), ((0,0), (0,pad_len)), mode='constant', constant_values=-400))
        mask_crops.append(np.pad(np.hstack(mask_tmp), ((0,0), (0,pad_len)), mode='constant', constant_values=0))
        aorta_dil_crops.append(np.pad(np.hstack(aorta_dil_tmp), ((0,0), (0,pad_len)), mode='constant', constant_values=0))

        
    ct_crops = np.vstack(ct_crops)
    mask_crops = np.vstack(mask_crops)
    aorta_dil_crops = np.vstack(aorta_dil_crops)
    # combine into one crop where calc has precedence
    combined_crops = np.zeros_like(mask_crops)
    combined_crops[aorta_dil_crops == 1] = 2
    combined_crops[mask_crops == 1] = 1

    fig, axx = plt.subplots(1, figsize=(20,20))

    axx.imshow(ct_crops, cmap='gray', vmin=-150, vmax=250)
    axx.imshow(combined_crops, vmin=0, vmax=10, cmap=map_object_seg, interpolation='nearest', alpha=0.4)
    
    for i, idx_ in enumerate(calc_idx[::-1]):
        # account for how the MIP is displayed
        idx = calc_mask.shape[2] - idx_
        x_ = crop_size*(i % per_row) + 4 #  crop_size//2
        y_ = crop_size*(i // per_row) + 4
        axx.text(x_, y_, s=str(idx), color = 'white', fontsize=10, va='top', ha='left',
                bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.1'))

    axx.set_xticks([])
    axx.set_yticks([])
    
    path = os.path.join(save_root, 'sub_figures')
    os.makedirs(path, exist_ok=True)
    # Save with the custom bounding box
    fig.savefig(os.path.join(path,'mosaic.png'),bbox_inches="tight", pad_inches=0, dpi=300)

    plt.close(fig)

def createCalciumMosaicVertebrae(
    ct: NDArray,
    calc_mask: NDArray,
    aorta_mask_dil: NDArray,
    spine_mask: NDArray,
    pix_size: NDArray,
    save_root: str,
    ) -> None:
    
    vertebrae_num = {
        26: "vertebrae_S1",
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
        50: "vertebrae_C1"}

    vertebrae_name = {v: k for k, v in vertebrae_num.items()}

    calc_idx = np.where( calc_mask.sum(axis=(0,1)) )[0]
    per_row = 5
    found_vertebras = np.unique(spine_mask)

    if len(found_vertebras) < per_row:
        per_row = len(found_vertebras)

    # target size of 120 mm crop size rounded to nearest multiple of 2
    crop_size = round(120 / pix_size[0])
    crop_size = 2*round(crop_size/2)

    ct_crops = []
    mask_crops = []
    aorta_dil_crops = []

    ct_tmp = []
    mask_tmp = []
    aorta_dil_tmp = []

    vertebra_names = []
    i = 0

    for vert_num in found_vertebras[::-1]:
        # only vertebraes
        if vert_num < 26:
            continue 
        
        # Find middle of the vertebra
        tmp_spine_mask = spine_mask == vert_num
        tmp_spine_idx = np.where(tmp_spine_mask.sum(axis=(0,1)))[0]
        idx = tmp_spine_idx[len(tmp_spine_idx)//2]

        ct_slice = np.flip(np.transpose(ct[:,:,idx]), axis=(0,1))
        mask_slice = np.flip(np.transpose(calc_mask[:,:,idx]), axis=(0,1))
        aorta_slice = np.flip(np.transpose(aorta_mask_dil[:,:,idx]), axis=(0,1))
        
        x_center = np.where(aorta_slice.sum(axis=1))[0]
        # skip if no aorta is present
        if len(x_center) == 0:
            continue
        x_center = x_center[len(x_center)//2] 
        
        y_center = np.where(aorta_slice.sum(axis=0))[0]
        y_center = y_center[len(y_center)//2] 

        ct_tmp.append(ct_slice[
            x_center-crop_size//2:x_center+crop_size//2,
            y_center-crop_size//2:y_center+crop_size//2,
                                ]
                        )
        mask_tmp.append(mask_slice[
            x_center-crop_size//2:x_center+crop_size//2,
            y_center-crop_size//2:y_center+crop_size//2,
                                ])
        
        aorta_dil_tmp.append(aorta_slice[
            x_center-crop_size//2:x_center+crop_size//2,
            y_center-crop_size//2:y_center+crop_size//2,
                                ])
        
        if (i+1) % per_row == 0:
            # print('got here')
            ct_crops.append(np.hstack(ct_tmp))
            mask_crops.append(np.hstack(mask_tmp))
            aorta_dil_crops.append(np.hstack(aorta_dil_tmp))

            ct_tmp = []
            mask_tmp = []
            aorta_dil_tmp = []
        
        vertebra_names.append(vertebrae_num[vert_num].split('_')[1])
        i += 1

    if len(ct_tmp) > 0:
        pad_len = per_row*crop_size - len(ct_tmp)*crop_size
        
        ct_crops.append(np.pad(np.hstack(ct_tmp), ((0,0), (0,pad_len)), mode='constant', constant_values=-400))
        mask_crops.append(np.pad(np.hstack(mask_tmp), ((0,0), (0,pad_len)), mode='constant', constant_values=0))
        aorta_dil_crops.append(np.pad(np.hstack(aorta_dil_tmp), ((0,0), (0,pad_len)), mode='constant', constant_values=0))

        
    ct_crops = np.vstack(ct_crops)
    mask_crops = np.vstack(mask_crops)
    aorta_dil_crops = np.vstack(aorta_dil_crops)
    # combine into one crop where calc has precedence
    combined_crops = np.zeros_like(mask_crops)
    combined_crops[aorta_dil_crops == 1] = 2
    combined_crops[mask_crops == 1] = 1

    fig, axx = plt.subplots(1, figsize=(20,20))

    axx.imshow(ct_crops, cmap='gray', vmin=-150, vmax=250)
    axx.imshow(combined_crops, vmin=0, vmax=10, cmap=map_object_seg, interpolation='nearest', alpha=0.4)
    
    # for i, idx_ in enumerate(calc_idx[::-1]):
    for i, name in enumerate(vertebra_names):
        x_ = crop_size*(i % per_row) + 4 #  crop_size//2
        y_ = crop_size*(i // per_row) + 4
        axx.text(x_, y_, s=name, color = 'white', fontsize=18, va='top', ha='left',
                bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.1'))

    axx.set_xticks([])
    axx.set_yticks([])
    
    path = os.path.join(save_root, 'sub_figures')
    os.makedirs(path, exist_ok=True)
    # Save with the custom bounding box
    fig.savefig(os.path.join(path,'mosaic.png'),bbox_inches="tight", pad_inches=0, dpi=300)

    plt.close(fig)
    
    
def mergeMipAndMosaic(save_root: str) -> None:
    '''
    This function loads the MIP and mosaic images and merges them into
    a single final image.
    '''
    
    if os.path.isfile(os.path.join(save_root, 'sub_figures/mosaic.png')):
        img_proj = Image.open(os.path.join(save_root, 'sub_figures/projection.png'))
        img_mosaic = Image.open(os.path.join(save_root, 'sub_figures/mosaic.png'))
        
        # Match the width to the projection image
        aspect_ratio = img_mosaic.height / img_mosaic.width
        new_height = int(img_proj.width * aspect_ratio)

        img_mosaic_resample = img_mosaic.resize([img_proj.width, new_height], Image.LANCZOS)

        result = Image.new("RGB", (img_proj.width, img_proj.height + img_mosaic_resample.height))

        result.paste(im=img_proj, box=(0, 0))
        result.paste(im=img_mosaic_resample, box=(0, img_proj.height))

        result.save(os.path.join(save_root,'overview.png'), dpi=(300, 300))
    else:
        shutil.copy2(os.path.join(save_root, 'sub_figures/projection.png'), os.path.join(save_root,'overview.png'))
        