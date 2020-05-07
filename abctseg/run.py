import logging
import os
import re
from typing import Dict, Sequence, Union

from abctseg.metrics import CrossSectionalArea, HounsfieldUnits
from abctseg.preferences import PREFERENCES

logger = logging.getLogger(__name__)


def format_output_path(file_path, save_dir: str = None):
    if not save_dir:
        save_dir = PREFERENCES.OUTPUT_DIR

    return os.path.join(
        os.path.dirname(file_path) if not save_dir else save_dir,
        "{}.h5".format(os.path.splitext(os.path.basename(file_path))[0]),
    )


def find_files(
    root_dirs: Union[str, Sequence[str]],
    max_depth: int = None,
    exist_ok: bool = False,
    pattern: str = None,
):
    """Recursively search for files.

    To avoid recomputing experiments with results, set `exist_ok=False`.
    Results will be searched for in `PREFERENCES.OUTPUT_DIR` (if non-empty).

    Args:
        root_dirs (`str(s)`): Root folder(s) to search.
        max_depth (int, optional): Maximum depth to search.
        exist_ok (bool, optional): If `True`, recompute results for
            scans.
        pattern (str, optional): If specified, looks for files with names
            matching the pattern.

    Return:
        List[str]: Experiment directories to test.
    """

    def _get_files(depth: int, dir_name: str):
        if dir_name is None or not os.path.isdir(dir_name):
            return []

        if max_depth is not None and depth > max_depth:
            return []

        files = os.listdir(dir_name)
        ret_files = []
        for file in files:
            possible_dir = os.path.join(dir_name, file)
            if os.path.isdir(possible_dir):
                subfiles = _get_files(depth + 1, possible_dir)
                ret_files.extend(subfiles)
            elif os.path.isfile(possible_dir):
                if pattern and not re.match(pattern, possible_dir):
                    continue
                output_path = format_output_path(possible_dir)
                if not exist_ok and os.path.isfile(output_path):
                    logger.info(
                        "Skipping {} - results exist at {}".format(
                            possible_dir, output_path
                        )
                    )
                    continue
                ret_files.append(possible_dir)

        return ret_files

    out_files = []
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    for d in root_dirs:
        out_files.extend(_get_files(0, d))

    return sorted(set(out_files))


def compute_results(x, mask, categories, params: Dict):
    hu = HounsfieldUnits()
    spacing = params.get("spacing", None)
    csa_units = "mm^2" if spacing else ""
    csa = CrossSectionalArea(csa_units)

    hu_vals = hu(mask, x, category_dim=-1)
    csa_vals = csa(mask=mask, spacing=spacing, category_dim=-1)

    assert mask.shape[-1] == len(categories), (
        "{} categories found in mask, "
        "but only {} categories specified".format(
            mask.shape[-1], len(categories)
        )
    )

    results = {
        cat: {
            "mask": mask[..., idx],
            hu.name(): hu_vals[idx],
            csa.name(): csa_vals[idx],
        }
        for idx, cat in enumerate(categories)
    }

    return results
