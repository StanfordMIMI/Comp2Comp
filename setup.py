#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from os import path

import keras
import tensorflow as tf
from setuptools import find_packages, setup

tf_ver = [int(x) for x in tf.__version__.split(".")[:2]]
assert tf_ver >= [1, 8] and tf_ver < [2, 0], "Requires TensorFlow >=1.8,<2.0"
keras_ver = [int(x) for x in keras.__version__.split(".")[:3]]
assert keras_ver >= [2, 1, 6] and keras_ver < [
    2,
    2,
    0,
], "Requires Keras >=2.1.6,<2.2.0"


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "abctseg", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [
        line.strip() for line in init_py if line.startswith("__version__")
    ][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("ABCTSEG_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [
            line for line in init_py if not line.startswith("__version__")
        ]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="abctseg",
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
        "pandas",
    ],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": [
            # Formatting
            "flake8",
            "isort",
            "black",
            "flake8-bugbear",
            "flake8-comprehensions",
            # Docs
            "mock",
            "sphinx",
            "sphinx-rtd-theme" "recommonmark",
        ],
    },
)
