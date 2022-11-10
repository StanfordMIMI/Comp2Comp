from time import perf_counter
from keras import backend as K
from tqdm import tqdm
import os
import silx.io.dictdump as sio
import matplotlib.pyplot as plt

from abctseg.models import Models
from abctseg.data import Dataset, predict
from abctseg.run import compute_results, find_files, format_output_path
from abctseg.utils.visualization import save_binary_segmentation_overlay
from abctseg.preferences import PREFERENCES, reset_preferences, save_preferences

def inference_2d(args, batch_size, use_pp, num_workers, files, num_gpus, logger, label_text):
    inputs = []
    masks = []
    file_names = []
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
        file_idx = 0
        for f, pred, mask, params in tqdm(  # noqa: B007
            zip(files, preds, masks, params_dicts), total=len(files)
        ):
            
            x = params["image"]
            results = compute_results(x, mask, categories, params)

            if label_text:
                file_name = label_text[file_idx]
                x = params["image"]
                inputs.append(x)
                masks.append(mask)
                file_names.append(file_name)
                #save_binary_segmentation_overlay(x, mask, PREFERENCES.OUTPUT_DIR, f"{file_name}.png")
            else:
                file_name = None
            output_file = format_output_path(
                f, base_dirs=dirs if args.batch else None, file_name=file_name
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            sio.dicttoh5(
                results, output_file, m_save_name, mode="a", overwrite_data=True
            )
            file_idx += 1
        logger.info(
            "<TIME>: Metrics - count: {} - {:.4f} seconds".format(
                len(files), perf_counter() - start_time
            )
        )
    return (inputs, masks, file_names)