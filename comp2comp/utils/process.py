import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import time

from comp2comp.io.io_utils import get_dicom_paths_and_num


def process_2d(args, pipeline_builder):
    output_dir = Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../outputs",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    )
    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    pipeline = pipeline_builder(args)

    pipeline(output_dir=output_dir, model_dir=model_dir)


def process_3d(args, pipeline_builder):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for path, num in get_dicom_paths_and_num(args.input_path):

        try:
            st = time()

            print("Processing: ", path, " with ", num, " slices")

            min_slices = 30
            if num < min_slices:
                print(f"Number of slices is less than {min_slices}, skipping\n")
                continue

            print("")

            try:
                sys.stdout.flush()
            except Exception:
                pass

            output_dir = Path(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "../../outputs",
                    date_time,
                    Path(os.path.basename(path)),
                )
            )

            if not os.path.exists(output_dir):
                output_dir.mkdir(parents=True)

            pipeline = pipeline_builder(path, args)

            pipeline(output_dir=output_dir, model_dir=model_dir)

            print(f"Finished processing {path} in {time() - st:.1f} seconds\n")

        except Exception:
            print(f"ERROR PROCESSING {path}\n")
            traceback.print_exc()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            continue
