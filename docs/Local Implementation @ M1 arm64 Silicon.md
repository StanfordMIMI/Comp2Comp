# Local Implementation @ M1/arm64/AppleSilicon

Due to dependencies and differences in architecture, the direct installation of *Comp2Comp* using install.sh or setup.py did not work on an local machine with arm64 / apple silicon running MacOS. This guide is mainly based on [issue #30](https://github.com/StanfordMIMI/Comp2Comp/issues/30). Most of the problems I encountered are caused by requiring TensorFlow and PyTorch in the same environment, which (especially for TensorFlow) is tricky at some times. Thus, this guide focuses more on the setup of the environment @arm64 / AppleSilicon, than *Comp2Comp* or *TotalSegmentator* itself.

## Installation
Comp2Comp requires TensorFlow and TotalSegmentator requires PyTorch. Although (at the moment) neither *Comp2Comp* nor *TotalSegmentator* can make use of the M1 GPU. Thus, using the arm64-specific versions is necessary.

### TensorFlow
For reference:
- https://developer.apple.com/metal/tensorflow-plugin/
- https://developer.apple.com/forums/thread/683757
- https://developer.apple.com/forums/thread/686926?page=2

1. Create an environment (python 3.8 or 3.9) using miniforge: https://github.com/conda-forge/miniforge. (TensorFlow did not work for others using anaconda; maybe you can get it running using -c apple and -c conda-forge for the further steps. However, I am not sure whether just the channel (and the retrieved packages) or anaconda's python itself is the problem.)

2. Install TensorFlow and tensorflow-metal in these versions:
```
conda install -c apple tensorflow-deps=2.9.0 -y
python -m pip install tensorflow-macos==2.9
python -m pip install tensorflow-metal==0.5.0
```
If you use other methods to install tensorflow, version 2.11.0 might be the best option. Tensorflow version 2.12.0 has caused some problems.

### PyTorch
For reference https://pytorch.org. The nightly build is (at least for -c conda-forge or -c pytorch) not needed, and the default already supports GPU acceleration on arm64.

3. Install Pytorch
```
conda install pytorch torchvision torchaudio -c pytorch
```

### Other Dependencies (Numpy and scikit-learn)
4. Install other packages
```
conda install -c conda-forge numpy scikit-learn -y
```

### TotalSegmentator
Louis et al. modified the original *TotalSegmentator* (https://github.com/wasserth/TotalSegmentator) for the use with *Comp2Comp*. *Comp2Comp* does not work with the original version. With the current version of the modified *TotalSegmentator* (https://github.com/StanfordMIMI/TotalSegmentator), no adaptions are necessary.

### Comp2Comp
For *Comp2Comp* on M1 however, it is important **not** to use bin/install.sh, as some of the predefined requirements won't work. Thus:

5. Clone *Comp2Comp*
```
git clone https://github.com/StanfordMIMI/Comp2Comp.git
```

6. Modify setup.py by
- remove `"numpy==1.23.5"`
- remove `"tensorflow>=2.0.0"`

(You have installed these manually before.)

7. Install *Comp2Comp* with
```
python -m pip install -e .
```

## Performance
Using M1Max w/ 64GB RAM
- `process 2d` (Comp2Comp in predefined slices): 250 slices in 14.2sec / 361 slices in 17.9sec
- `process 3d` (segmentation of spine and identification of slices using TotalSegmentator, Comp2Comp in identified slices): high res, full body scan, 1367sec

## ToDos / Nice2Have / Future
- Integration and use `--fast` and `--body_seg` for TotalSegmentator might be preferable
- TotalSegmentator works only with CUDA compatible GPUs (!="mps"). I am not sure, about `torch.device("mps")` in the future, see also https://github.com/wasserth/TotalSegmentator/issues/39. Currently, only the CPU is used.
