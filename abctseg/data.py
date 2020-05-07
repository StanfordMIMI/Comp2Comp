import math
from typing import List, Sequence

import keras.utils as k_utils
import numpy as np
import pydicom
from keras.utils.data_utils import OrderedEnqueuer
from tqdm import tqdm


def parse_windows(windows):
    """Parse windows provided by the user.

    These windows can either be strings corresponding to popular windowing
    thresholds for CT or tuples of (upper, lower) bounds.
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
    imgs = []
    for l, u in bounds:
        imgs.append(np.clip(xs, a_min=l, a_max=u))

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
        files = self._files[
            idx * self._batch_size : (idx + 1) * self._batch_size
        ]
        dcms = [pydicom.read_file(f, force=True) for f in files]

        xs = [
            (x.pixel_array + int(x.RescaleIntercept)).astype("float32")
            for x in dcms
        ]

        params = [
            {"spacing": header.PixelSpacing, "image": x}
            for header, x in zip(dcms, xs)
        ]

        # Preprocess xs via windowing.
        xs = np.stack(xs, axis=0)
        if self.windows:
            xs = _window(xs, parse_windows(self.windows))
        else:
            xs = xs[..., np.newaxis]

        return xs, params


def predict(
    model,
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 1,
    max_queue_size: int = 10,
    use_multiprocessing: bool = False,
):
    if num_workers > 0:
        enqueuer = OrderedEnqueuer(
            dataset, use_multiprocessing=use_multiprocessing, shuffle=False
        )
        enqueuer.start(workers=num_workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
    else:
        output_generator = iter(dataset)

    num_scans = len(dataset)
    xs = []
    ys = []
    params = []
    for _ in tqdm(range(num_scans)):
        x, p = next(output_generator)
        y = model.predict(x, batch_size=batch_size)

        params.extend(p)
        xs.extend([x[i, ...] for i in range(len(x))])
        ys.extend([y[i, ...] for i in range(len(y))])

    return xs, ys, params
