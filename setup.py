#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from os import path
from setuptools import find_packages, setup

import keras
import tensorflow as tf

tf_ver = [int(x) for x in tf.__version__.split(".")[:2]]
assert tf_ver >= [1, 8] and tf_ver < [2, 0], "Requires TensorFlow >=1.8,<2.0"
keras_ver = [int(x) for x in keras.__version__.split(".")[:3]]
assert keras_ver >= [2, 1, 6] and keras_ver < [2,2,0,], (
    "Requires Keras >=2.1.6, <2.2.0"
)


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "ihd_pipeline", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][
        0
    ]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("IHD_PIPELINE_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="ihd_pipeline",
    version=get_version(),
    author="Arjun Desai",
    url="https://github.com/ad12/ihd-pipeline",
    description="Abdominal CT segmentation pipeline.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
        "pydicom",
        "numpy",
        "h5py",
        "tabulate",
        "tqdm",
        "silx",
        "yacs",
    ],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": [
            "flake8",
            "isort",
            "black==19.3b0",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
)
