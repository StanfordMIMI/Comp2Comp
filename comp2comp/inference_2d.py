import argparse
import logging
import os
from time import perf_counter
from typing import List

import silx.io.dictdump as sio
from keras import backend as K
from tqdm import tqdm
import numpy as np
import cv2

from comp2comp.data import Dataset, fill_holes, predict
from comp2comp.models import Models
from comp2comp.run import compute_results
from comp2comp.utils import dl_utils


def forward_pass_2d(
    args: argparse.Namespace,
    batch_size: int,
    use_pp: bool,
    num_workers: int,
    files: list,
    num_gpus: int,
    logger: logging.Logger,
    model_type: Models,
):
    """Run inference on 2D images.

    Args:
        args (argparse.Namespace): Arguments.
        batch_size (int): Batch size.
        use_pp (bool): Use post-processing.
        num_workers (int): Number of workers.
        files (list): List of files.
        num_gpus (int): Number of GPUs.
        logger (logging.Logger): Logger object.
        model_type (Models): Model type.

    Returns:
        preds (list): Predictions.
        params_dicts (list): Parameters dictionaries.
    """

    m_name = args.muscle_fat_model
    logger.info("Computing masks with model {}".format(m_name))

    dataset = Dataset(files, windows=model_type.windows)
    categories = model_type.categories
    model = model_type.load_model(logger)

    if num_gpus > 1:
        model = dl_utils.ModelMGPU(model, gpus=num_gpus)

    logger.info("<MODEL>: {}".format(m_name))
    logger.info("Computing segmentation masks using {}...".format(m_name))
    start_time = perf_counter()
    _, preds, params_dicts = predict(
        model,
        dataset,
        num_workers=num_workers,
        use_multiprocessing=num_workers > 1,
        batch_size=batch_size,
        use_postprocessing=use_pp,
        postprocessing_params={"categories": categories},
    )
    K.clear_session()
    logger.info(
        "<TIME>: Segmentation - count: {} - {:.4f} seconds".format(
            len(files), perf_counter() - start_time
        )
    )
    return preds, params_dicts


def compute_and_save_results(
    args: argparse.Namespace,
    preds: list,
    params_dicts: list,
    files: list,
    label_text: List[str],
    output_dir: str,
    logger: logging.Logger,
    model_type: Models,
):
    """Compute and save results.

    Args:
        args (argparse.Namespace): Arguments.
        preds (list): Predictions.
        params_dicts (list): Parameters dictionaries.
        files (list): List of files.
        label_text (List[str]): Label text.
        output_dir (str): Output directory.
        logger (logging.Logger): Logger object.
        model_type (Models): Model type.

    Returns:
        inputs (list): Inputs.
        masks (list): Masks.
        file_names (list): File names.
        results (list): Results.
    """
    inputs = []
    masks = []
    file_names = []
    results_dict = {}
    logger.info("Computing metrics...")
    m_name = args.muscle_fat_model
    categories = model_type.categories
    start_time = perf_counter()
    masks = [model_type.preds_to_mask(p) for p in preds]
    for i, mask in enumerate(masks):
        # Keep only channels from the model_type categories dict
        masks[i] = masks[i][
            ..., [model_type.categories[cat] for cat in model_type.categories]
        ]
    if args.pp:
        masks = fill_holes(masks)
        if "muscle" in categories and "imat" in categories:
            cats = list(categories.keys())
            muscle_idx = cats.index("muscle")

    assert len(masks) == len(params_dicts)
    assert len(masks) == len(files)
    m_save_name = "/{}".format(m_name)
    file_idx = 0
    cats = list(categories.keys())
    for _, _, mask, params in tqdm(  # noqa: B007
        zip(files, preds, masks, params_dicts), total=len(files)
    ):

        x = params["image"]
        muscle_mask = mask[..., cats.index("muscle")]
        imat_mask = mask[..., cats.index("imat")]
        imat_mask = (np.logical_and((x * muscle_mask) <= -30, (x * muscle_mask) >= -190)).astype(int)
        # get rid of small connected components as these are likely noise
        imat_mask = remove_small_objects(imat_mask)
        mask[..., cats.index("imat")] += imat_mask
        mask[..., cats.index("muscle")][imat_mask == 1] = 0
        results = compute_results(x, mask, categories, params)
        results_dict[label_text[file_idx]] = results

        if label_text:
            file_name = label_text[file_idx]
            x = params["image"]
            inputs.append(x)
            masks.append(mask)
            file_names.append(file_name)
        else:
            file_name = None
        output_file = os.path.join(output_dir, file_name + ".h5")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sio.dicttoh5(results, output_file, m_save_name, mode="a", overwrite_data=True)
        file_idx += 1
    logger.info(
        "<TIME>: Metrics - count: {} - {:.4f} seconds".format(
            len(files), perf_counter() - start_time
        )
    )
    return (inputs, masks, file_names, results_dict)    

def remove_small_objects(mask, min_size=10):
    mask = mask.astype(np.uint8)
    components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    mask = np.zeros((output.shape))
    for i in range(0, components - 1):
        if sizes[i] >= min_size:
            mask[output == i + 1] = 1
    return mask

def inference_2d(
    args: argparse.Namespace,
    batch_size: int,
    use_pp: bool,
    num_workers: int,
    files: list,
    num_gpus: int,
    logger: logging.Logger,
    model_type: Models,
    label_text: List[str] = None,
    output_dir: str = None,
):
    """Run inference on 2D images.

    Args:
        args (argparse.Namespace): Arguments.
        batch_size (int): Batch size.
        use_pp (bool): Use post-processing.
        num_workers (int): Number of workers.
        files (list): List of files.
        num_gpus (int): Number of GPUs.
        logger (logging.Logger): Logger object.
        model_type (Models): Model type.
        label_text (List[str], optional): Label text. Defaults to None.
        output_dir (str, optional): Output directory. Defaults to None.

    Returns:
        inputs (list): Inputs.
        masks (list): Masks.
        file_names (list): File names.
        results (list): Results.
    """

    (preds, params_dict) = forward_pass_2d(
        args,
        batch_size,
        use_pp,
        num_workers,
        files,
        num_gpus,
        logger,
        model_type,
    )
    (inputs, masks, file_names, results) = compute_and_save_results(
        args,
        preds,
        params_dict,
        files,
        label_text,
        output_dir,
        logger,
        model_type,
    )
    return (inputs, masks, file_names, results)
