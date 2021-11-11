from keras.models import K
from tqdm import tqdm

import argparse
import logging
import os
import silx.io.dictdump as sio
import sys
from abctseg.data import Dataset, predict
from abctseg.models import Models
from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences
from abctseg.run import compute_results, find_files, format_output_path
from abctseg.utils import dl_utils
from abctseg.utils.logger import setup_logger
from time import perf_counter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup(args):
    """Load preferences and perform basic setups."""
    args_dict = vars(args)
    config_file = args.config_file if "config_file" in args_dict else None
    if config_file:
        PREFERENCES.merge_from_file(args.config_file)
    opts = args.opts
    if len(opts) and opts[0] == "--":
        opts = opts[1:]
    PREFERENCES.merge_from_list(opts)
    setup_logger(PREFERENCES.CACHE_DIR)


def add_config_file_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--config-file",
        type=str,
        required=False,
        help="Preferences config file",
    )


def add_opts_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "opts",
        help="Modify preferences options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


def argument_parser():
    parser = argparse.ArgumentParser("abCTSeg command line interface")
    subparsers = parser.add_subparsers(dest="action")

    # Processing parser
    process_parser = subparsers.add_parser("process", help="process abCT scans")
    process_parser.add_argument(
        "--dicoms",
        nargs="+",
        type=str,
        required=True,
        help="path to dicom files(s)/directories to segment.",
    )
    process_parser.add_argument(
        "--max-depth",
        nargs="?",
        type=str,
        default=None,
        help="max depth to search directory. Default: None (recursive search)",
    )
    process_parser.add_argument(
        "--pattern",
        nargs="?",
        type=str,
        default=None,
        help="regex pattern for file names. Default: None",
    )
    process_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite results for computed files. Default: False",
    )
    process_parser.add_argument(
        "--num-gpus",
        default=1,
        type=int,
        help="number of GPU(s) to use. Defaults to cpu if no gpu found.",
    )
    process_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        type=str,
        choices=[x.model_name for x in Models],
        help="models to use for inference",
    )
    process_parser.add_argument(
        "--pp",
        action="store_true",
        help="use post-processing. Will be used for all specified models.",
    )
    add_config_file_argument(process_parser)
    add_opts_argument(process_parser)

    # init parser.
    cfg_parser = subparsers.add_parser("config", help="init abCTSeg library")
    subparsers = cfg_parser.add_subparsers(
        title="config sub-commands", dest="cfg_action"
    )
    subparsers.add_parser("ls", help="list default preferences config")
    subparsers.add_parser("reset", help="reset to default config")
    save_cfg_parser = subparsers.add_parser("save", help="set config defaults")
    add_config_file_argument(save_cfg_parser)
    add_opts_argument(save_cfg_parser)
    return parser


def handle_init(args):
    cfg_action = args.cfg_action
    if cfg_action == "ls":
        print("\n" + PREFERENCES.dump())
    elif cfg_action == "reset":
        print("\nResetting preferences...")
        reset_preferences()
        save_preferences()
        print("\n" + PREFERENCES.dump())
    elif cfg_action == "save":
        setup(args)
        save_preferences()
        print("\nUpdated Preferences:")
        print("====================")
        print(PREFERENCES.dump())
    else:
        raise AssertionError("cfg_action {} not supported".format(cfg_action))


def handle_process(args):
    setup(args)
    if not PREFERENCES.MODELS_DIR:
        raise ValueError(
            "MODELS_DIR not initialized. "
            "Use `python -m abctseg.cli config` to set MODELS_DIR"
        )
    logger = logging.getLogger("abctseg.cli.__main__")
    logger.info("\n\n======================================================")
    gpus = dl_utils.get_available_gpus(args.num_gpus)
    num_gpus = len(gpus) if gpus is not None else 0
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # cpu

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
    use_pp = args.pp
    num_workers = PREFERENCES.NUM_WORKERS

    logger.info("Preferences:\n" + PREFERENCES.dump())
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

        logger.info("Computing metrics...")
        start_time = perf_counter()
        masks = [model_type.preds_to_mask(p) for p in preds]
        assert len(masks) == len(params_dicts)
        assert len(masks) == len(files)
        m_save_name = "/{}".format(m_name)
        if use_pp:
            m_save_name += "+pp"
        for f, pred, mask, params in tqdm(  # noqa: B007
            zip(files, preds, masks, params_dicts), total=len(files)
        ):
            x = params["image"]
            results = compute_results(x, mask, categories, params)
            output_file = format_output_path(f)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            sio.dicttoh5(
                results, output_file, m_save_name, mode="a", overwrite_data=True
            )
        logger.info(
            "<TIME>: Metrics - count: {} - {:.4f} seconds".format(
                len(files), perf_counter() - start_time
            )
        )


def main():
    args = argument_parser().parse_args()
    if args.action == "config":
        handle_init(args)
    elif args.action == "process":
        handle_process(args)
    else:
        raise AssertionError("{} command not supported".format(args.action))


if __name__ == "__main__":
    main()
