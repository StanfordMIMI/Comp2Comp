import math
from typing import List, Sequence
import matplotlib.pyplot as plt

import cv2
import keras.utils as k_utils
import numpy as np
import pydicom
from keras.utils.data_utils import OrderedEnqueuer
from tqdm import tqdm


def parse_windows(windows):
    """Parse windows provided by the user.

    These windows can either be strings corresponding to popular windowing
    thresholds for CT or tuples of (upper, lower) bounds.

    Args:
        windows (list): List of strings or tuples.

    Returns:
        list: List of tuples of (upper, lower) bounds.
    """
    windowing = {
        "soft": (400, 50),
        "bone": (1800, 400),
        "liver": (150, 30),
        "spine": (250, 50),
        "custom": (500, 50),
    }
    vals = []
    for w in windows:
        if isinstance(w, Sequence) and len(w) == 2:
            assert_msg = "Expected tuple of (lower, upper) bound"
            assert len(w) == 2, assert_msg
            assert isinstance(w[0], (float, int)), assert_msg
            assert isinstance(w[1], (float, int)), assert_msg
            assert w[0] < w[1], assert_msg
            vals.append(w)
            continue

        if w not in windowing:
            raise KeyError("Window {} not found".format(w))
        window_width = windowing[w][0]
        window_level = windowing[w][1]
        upper = window_level + window_width / 2
        lower = window_level - window_width / 2

        vals.append((lower, upper))

    return tuple(vals)


def _window(xs, bounds):
    """Apply windowing to an array of CT images.

    Args:
        xs (ndarray): NxHxW
        bounds (tuple): (lower, upper) bounds

    Returns:
        ndarray: Windowed images.
    """

    imgs = []
    for lb, ub in bounds:
        imgs.append(np.clip(xs, a_min=lb, a_max=ub))

    if len(imgs) == 1:
        return imgs[0]
    elif xs.shape[-1] == 1:
        return np.concatenate(imgs, axis=-1)
    else:
        return np.stack(imgs, axis=-1)


class Dataset(k_utils.Sequence):
    def __init__(self, files: List[str], batch_size: int = 16, windows=None):
        self._files = files
        self._batch_size = batch_size
        self.windows = windows

    def __len__(self):
        return math.ceil(len(self._files) / self._batch_size)

    def __getitem__(self, idx):
        files = self._files[idx * self._batch_size : (idx + 1) * self._batch_size]
        dcms = [pydicom.read_file(f, force=True) for f in files]

        xs = [(x.pixel_array + int(x.RescaleIntercept)).astype("float32") for x in dcms]

        params = [{"spacing": header.PixelSpacing, "image": x} for header, x in zip(dcms, xs)]

        # Preprocess xs via windowing.
        xs = np.stack(xs, axis=0)
        if self.windows:
            xs = _window(xs, parse_windows(self.windows))
        else:
            xs = xs[..., np.newaxis]

        return xs, params


# function that fills in holes in a segmentation mask
def _fill_holes(mask: np.ndarray, mask_id: int):
    """Fill in holes in a segmentation mask.

    Args:
        mask (ndarray): NxHxW
        mask_id (int): Label of the mask.

    Returns:
        ndarray: Filled mask.
    """
    int_mask = ((1 - mask) > 0.5).astype(np.int8)
    components, output, stats, _ = cv2.connectedComponentsWithStats(int_mask, connectivity=8)
    sizes = stats[1:, -1]
    components = components - 1
    # Larger threshold for SAT
    # TODO make this configurable / parameter
    if mask_id == 2:
        min_size = 200
    else:
        #min_size = 50  # Smaller threshold for everything else
        min_size = 20
    img_out = np.ones_like(mask)
    for i in range(0, components):
        if sizes[i] > min_size:
            img_out[output == i + 1] = 0
    return img_out


def fill_holes(ys: List):
    """Take an array of size NxHxWxC and for each channel fill in holes.

    Args:
        ys (list): List of segmentation masks.
    """
    segs = []
    for n in range(len(ys)):
        ys_out = [_fill_holes(ys[n][..., i], i) for i in range(ys[n].shape[-1])]
        segs.append(np.stack(ys_out, axis=2).astype(float))

    return segs


def _swap_muscle_imap(xs, ys, muscle_idx: int, imat_idx: int, threshold=-30.0):
    """
    If pixel labeled as muscle but has HU < threshold, change label to imat.

    Args:
        xs (ndarray): NxHxWxC
        ys (ndarray): NxHxWxC
        muscle_idx (int): Index of the muscle label.
        imat_idx (int): Index of the imat label.
        threshold (float): Threshold for HU value.

    Returns:
        ndarray: Segmentation mask with swapped labels.
    """
    labels = ys.copy()

    muscle_mask = (labels[..., muscle_idx] > 0.5).astype(int)
    imat_mask = labels[..., imat_idx]

    imat_mask[muscle_mask.astype(np.bool) & (xs < threshold)] = 1
    muscle_mask[xs < threshold] = 0

    labels[..., muscle_idx] = muscle_mask
    labels[..., imat_idx] = imat_mask

    return labels


def postprocess(xs: np.ndarray, ys: np.ndarray, params: dict):
    """Built-in post-processing.

    TODO: Make this configurable.

    Args:
        xs (ndarray): NxHxW
        ys (ndarray): NxHxWxC
        params (dictionary): Post-processing parameters. Must contain
            "categories".

    Returns:
        ndarray: Post-processed labels.
    """
    assert params
    categories = params["categories"]

    # Add another channel full of zeros to ys
    ys = np.concatenate([ys, np.zeros_like(ys[..., :1])], axis=-1)

    # If muscle hu is < -30, assume it is imat.

    '''
    if "muscle" in categories and "imat" in categories:
        ys = _swap_muscle_imap(
            xs,
            ys,
            muscle_idx=categories["muscle"],
            imat_idx=categories["imat"],
        )
    '''

    return ys


def predict(
    model,
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 1,
    max_queue_size: int = 10,
    use_multiprocessing: bool = False,
    use_postprocessing: bool = False,
    postprocessing_params: dict = None,
):
    """Predict segmentation masks for a dataset.

    Args:
        model (keras.Model): Model to use for prediction.
        dataset (Dataset): Dataset to predict on.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        max_queue_size (int): Maximum queue size.
        use_multiprocessing (bool): Use multiprocessing.
        use_postprocessing (bool): Use built-in post-processing.
        postprocessing_params (dict): Post-processing parameters.

    Returns:
        List: List of segmentation masks.
    """

    if num_workers > 0:
        enqueuer = OrderedEnqueuer(dataset, use_multiprocessing=use_multiprocessing, shuffle=False)
        enqueuer.start(workers=num_workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
    else:
        output_generator = iter(dataset)

    num_scans = len(dataset)
    xs = []
    ys = []
    params = []
    for _ in tqdm(range(num_scans)):
        x, p_dicts = next(output_generator)
        y = model.predict(x, batch_size=batch_size)

        if use_postprocessing:
            image = np.stack([out["image"] for out in p_dicts], axis=0)
            y = postprocess(image, y, postprocessing_params)

        params.extend(p_dicts)
        xs.extend([x[i, ...] for i in range(len(x))])
        ys.extend([y[i, ...] for i in range(len(y))])

    return xs, ys, params
