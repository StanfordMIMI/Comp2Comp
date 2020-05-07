## Installation
We recommend using the Anaconda virtual environment to control package
visibility.

### Requirements
- Linux or macOS with Python ≥ 3.6
- keras >=2.1.6,<2.2.0
- tensorflow-gpu >=1.8.0,<2.0.0

Note, you should select the `tensorflow-gpu` version based on your cuda version. 
If you do not have a gpu, replace the
`tensorflow-gpu` package with `tensorflow`.

### Build abCTSeg from Source
```bash
python -m pip install 'git+https://github.com/ad12/ihd-pipeline.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone (recommended):
git clone https://github.com/ad12/ihd-pipeline.git
cd MedSegPy && python -m pip install -e .
```

You often need to rebuild abctseg after reinstalling TensorFlow and/or Keras.
