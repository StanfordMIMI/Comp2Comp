import argparse
import logging
import os
from time import perf_counter

from keras.models import K
from tqdm import tqdm

from ihd_pipeline.utils.logger import setup_logger
from ihd_pipeline.utils import dl_utils
from ihd_pipeline.models import Models
from ihd_pipeline.preferences import PREFERENCES
from ihd_pipeline.run import find_files, compute_results, format_output_path
from ihd_pipeline.data import Dataset, predict
from ihd_pipeline.metrics import HounsfieldUnits, CrossSectionalArea

import silx.io.dictdump as sio


def argument_parser():
    parser = argparse.ArgumentParser("segment abdominal CT")
    parser.add_argument(
        "--dicoms", nargs="+", type=str,
        help="path to dicom files(s)/directories to segment."
    )
    parser.add_argument(
        "--max-depth", nargs="?", type=str,
        default=None,
        help="max depth to search directory. Default: None (recursive search)"
    )
    parser.add_argument(
        "--pattern", nargs="?", type=str,
        default=None,
        help="regex pattern for file names. Default: None"
    )
    parser.add_argument(
        "--num-gpus", default=1,
        help="number of GPU(s) to use. Defaults to cpu if no gpu found."
    )
    parser.add_argument(
        "--models", nargs="+", type=str,
        choices=[x.model_name for x in Models],
        help="models to use for inference",
    )
    return parser


def setup(args):
    """Load preferences and perform basic setups.
    """
    config_file = args.config_file
    if config_file:
        PREFERENCES.merge_from_file(args.config_file)
    PREFERENCES.merge_from_list(args.opts)
    setup_logger(PREFERENCES.CACHE_DIR)


def main():
    args = argument_parser().parse_args()
    setup(args)
    logger = logging.getLogger("ihd_pipeline.cli.__main__")
    logger.info("\n\n======================================================")
    gpus = dl_utils.get_available_gpus(args.num_gpus)
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = 0  # cpu
    num_gpus = len(gpus) if gpus is not None else 0

    # Find files.
    files = []
    dirs = []
    for f in args.dicoms:
        if os.path.isfile(f):
            files.append(f)
        elif os.path.isdir(f):
            dirs.append(f)
    files.extend(
        find_files(
            dirs,
            max_depth=args.max_depth,
            exist_ok=args.exist_ok,
            pattern=args.pattern
        )
    )
    logger.info("{} scans found".format(len(files)))

    dataset = Dataset(files)
    batch_size = PREFERENCES.BATCH_SIZE

    for m_name in args.models:
        logger.info("Computing masks with model {}".format(m_name))

        model_type: Models = None
        for m_type in Models:
            if m_type.model_name == m_name:
                model_type = m_type
                break
        assert model_type is not None
        categories = model_type.categories
        model = model_type.load_model()
        if num_gpus > 1:
            model = dl_utils.ModelMGPU(model, gpus=num_gpus)

        logger.info("Computing segmentation masks...")
        start_time = perf_counter()
        _, preds, params_dicts = predict(
            model,
            dataset,
            batch_size=batch_size,
        )
        K.clear_session()
        logger.info("<TIME>: Segmentation - count: {} - {:.4f} seconds".format(
            len(files), perf_counter() - start_time
        ))

        logger.info("Computing metrics...")
        start_time = perf_counter()
        masks = [model_type.preds_to_mask(p) for p in preds]
        assert len(masks) == len(params_dicts)
        assert len(masks) == len(files)
        for f, pred, mask, params in tqdm(zip(files, preds, masks, params_dicts)):  # noqa
            x = params["image"]
            results = compute_results(
                x, mask, categories, params
            )
            output_file = format_output_path(f)
            sio.dicttoh5(results, output_file, "/{}".format(m_name), mode="a")
        logger.info("<TIME>: Metrics - count: {} - {:.4f} seconds".format(
            len(files), perf_counter() - start_time
        ))
