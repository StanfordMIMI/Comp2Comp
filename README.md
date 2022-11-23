# abCTSeg
abCTSeg is a library for segmenting 2D abdominal CT scans.
It streamlines deploying models in Keras/Tensorflow for segmenting
tissues like muscle, bone, intramuscular fat (IMAT), visceral fat (VAT), and
more.

## Installation
```bash
# Install via pip
python -m pip install 'git+https://github.com/ad12/abCTSeg.git'
# (add --user if you don't have permission)
# Or, to install it from a local clone (recommended):
git clone https://github.com/ad12/abCTSeg.git
cd abCTSeg && python -m pip install -e .
# Or, install via script (requires Anaconda/Miniconda).
bin/install.sh
```

See [Installation](INSTALL.md) for more step-by-step installation details.

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md) and learn more at our
[documentation](https://ad12.github.io/abCTSeg/).
