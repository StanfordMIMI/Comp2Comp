# Local Implementation @ M1/arm64/AppleSilicon

Due to dependencies and differences in architecture, the direct installation of *Comp2Comp* using install.sh or setup.py did not work on an local machine with arm64 / apple silicon running MacOS. This guide is mainly based on [issue #30](https://github.com/StanfordMIMI/Comp2Comp/issues/30). Most of the problems I encountered are caused by requiring TensorFlow and PyTorch in the same environment, which (especially for TensorFlow) is tricky at some times. Thus, this guide focuses more on the setup of the environment @arm64 / AppleSilicon, than *Comp2Comp* or *TotalSegmentator* itself.

## Installation
Comp2Comp requires TensorFlow, TotalSegmentator PyTorch. Although (at the moment) neither *Comp2Comp* nor *TotalSegmentator* can make use of the M1 GPUs, using the arm64-specific versions is necessary.

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
Take care, Louis et al. modified the original *TotalSegmentator* (https://github.com/wasserth/TotalSegmentator) for the use with *Comp2Comp*. *Comp2Comp* does not work with the original version. However, the installer of the modified *TotalSegmentator* (https://github.com/StanfordMIMI/TotalSegmentator) must be adapted slightly, as you need other package-versions on your machine. Thus:

5. Clone (modified!!) *TotalSegmentator*
```
git clone https://github.com/StanfordMIMI/TotalSegmentator
```

6. Modify setup.py by
- replace `'SimpleITK==2.0.2'` with `'SimpleITK'`
- replace `'nnunet-customized'` with `'nnunet-customized==1.2'`

7. Install *TotalSegmentator* (modified) with
```
python -m pip install -e .
```

### Comp2Comp
Also for *Comp2Comp*, it is important **not** to use the installation bash, as some of the predefined requirements won't work. Thus:

8. Clone *Comp2Comp*
```
git clone https://github.com/StanfordMIMI/Comp2Comp.git
```

9. Modify setup.py by
- remove `"numpy==1.23.5"`
- remove `"tensorflow>=2.0.0"`
- remove `'totalsegmentator @ git+https://github.com/StanfordMIMI/TotalSegmentator.git'`

(You have installed all of these manually before.)

10. Install *Comp2Comp* with
```
python -m pip install -e .
```

## Common errors

### NvidiaSMI
The authors check free memory on gpus using `nvidia-smi`. This will not work on a arm64 machine w/o NVIDIA GPU. You can ignore the `/bin/sh: nvidia-smi: command not found`-message, both *Comp2Comp* and *TotalSegmentator* will just run on your CPU. Alternatively, you could avoid the `nvidia-smi`-call by setting `num_gpus = 0; gpus = None` manually and remove or comment out all `dl_utils.get_available_gpus`-calls in C2C and cli.py.

### get_dicom_paths_and_num
For `process 2d`, *Comp2Comp* searches for dicom files in the input directory. The `get_dicom_paths_and_num`-function does not return anything if any other files than `*.dcm` (e.g. hidden files such as .DS_Store) are included in the directory. Adapt the function accordingly, or make sure that only `*.dcm` are included in the directory.

## Performance
Using M1Max w/ 64GB RAM
- `process 2d` (Comp2Comp in predefined slices): 250 slices in 14.2sec / 361 slices in 17.9sec
- `process 3d` (segmentation of spine and identification of slices using TotalSegmentator, Comp2Comp in identified slices): high res, full body scan, 1367sec

## ToDos / Nice2Have / Future
- Integration and use `--fast` and `--body_seg` for TotalSegmentator might be preferable
- TotalSegmentator works only with CUDA compatible GPUs (!="mps"). I am not sure, about `torch.device("mps")` in the future, see also https://github.com/wasserth/TotalSegmentator/issues/39. Currently, only the CPU is used.
