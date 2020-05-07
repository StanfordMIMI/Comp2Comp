import argparse
import logging
import os
import sys
from time import perf_counter

import silx.io.dictdump as sio
from keras.models import K
from tqdm import tqdm

from abctseg.data import Dataset, predict
from abctseg.models import Models
from abctseg.preferences import PREFERENCES
from abctseg.run import compute_results, find_files, format_output_path
from abctseg.utils import dl_utils
from abctseg.utils.logger import setup_logger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def argument_parser():
    parser = argparse.ArgumentParser("segment abdominal CT")
    subparsers = parser.add_subparsers(dest="action")

    seg_parser = subparsers.add_parser("segment", help="segment abCT scans")
    seg_parser.add_argument(
        "--dicoms",
        nargs="+",
        type=str,
        required=True,
        help="path to dicom files(s)/directories to segment.",
    )
    seg_parser.add_argument(
        "--max-depth",
        nargs="?",
        type=str,
        default=None,
        help="max depth to search directory. Default: None (recursive search)",
    )
    seg_parser.add_argument(
        "--pattern",
        nargs="?",
        type=str,
        default=None,
        help="regex pattern for file names. Default: None",
    )
    seg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite results for computed files. Default: False",
    )
    seg_parser.add_argument(
        "--num-gpus",
        default=1,
        type=int,
        help="number of GPU(s) to use. Defaults to cpu if no gpu found.",
    )
    seg_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        choices=[x.model_name for x in Models],
        help="models to use for inference",
    )
    seg_parser.add_argument(
        "opts",
        help="Modify preferences options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    pref_parser = subparsers.add_parser(
        "configure", help="configure preferences"
    )
    pref_parser.add_argument(
        "opts",
        help="Modify preferences options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def setup(args):
    """Load preferences and perform basic setups.
    """
    args_dict = vars(args)
    config_file = args.config_file if "config_file" in args_dict else None
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
            exist_ok=args.overwrite,
            pattern=args.pattern,
        )
    )
    logger.info("{} scans found".format(len(files)))
    if len(files) == 0:
        sys.exit(0)

    batch_size = PREFERENCES.BATCH_SIZE

    for m_name in args.models:
        logger.info("Computing masks with model {}".format(m_name))

        model_type: Models = None
        for m_type in Models:
            if m_type.model_name == m_name:
                model_type = m_type
                break
        assert model_type is not None

        dataset = Dataset(files, windows=model_type.windows)
        categories = model_type.categories
        model = model_type.load_model()
        if num_gpus > 1:
            model = dl_utils.ModelMGPU(model, gpus=num_gpus)

        logger.info("Computing segmentation masks...")
        start_time = perf_counter()
        _, preds, params_dicts = predict(model, dataset, batch_size=batch_size)
        K.clear_session()
        logger.info(
            "<TIME>: Segmentation - count: {} - {:.4f} seconds".format(
                len(files), perf_counter() - start_time
            )
        )

        logger.info("Computing metrics...")
        start_time = perf_counter()
        masks = [model_type.preds_to_mask(p) for p in preds]
        assert len(masks) == len(params_dicts)
        assert len(masks) == len(files)
        for f, pred, mask, params in tqdm(
            zip(files, preds, masks, params_dicts), total=len(files)
        ):  # noqa
            x = params["image"]
            results = compute_results(x, mask, categories, params)
            output_file = format_output_path(f)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            sio.dicttoh5(
                results,
                output_file,
                "/{}".format(m_name),
                mode="w",
                overwrite_data=True,
            )
        logger.info(
            "<TIME>: Metrics - count: {} - {:.4f} seconds".format(
                len(files), perf_counter() - start_time
            )
        )


if __name__ == "__main__":
    main()
