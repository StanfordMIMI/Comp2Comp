#!/bin/bash

# ==============================================================================
# Auto-installation for abCTSeg for Linux and Mac machines.
# This setup script is adapted from DOSMA:
# https://github.com/ad12/DOSMA
# ==============================================================================

BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

ANACONDA_KEYWORD="anaconda"
ANACONDA_DOWNLOAD_URL="https://www.anaconda.com/distribution/"
MINICONDA_KEYWORD="miniconda"

# FIXME: Update the name.
ABCT_ENV_NAME="c2c_env"

hasAnaconda=0
updateEnv=0
updatePath=1
pythonVersion="3.8"
cudaVersion=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
    	-h|--help)
			echo "Batch evaluation with ss_recon"
			echo ""
			echo "Usage:"
			echo "    --python <string>          Python version"
			echo "    -f, --force                Force environment update"
			exit
			;;
        --python)
            pythonVersion=$2
            shift # past argument
            shift # past value
            ;;
        --cuda)
            cudaVersion=$2
            shift # past argument
            shift # past value
            ;;
        -f|--force)
            updateEnv=1
            shift # past argument
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Initial setup
source ~/.bashrc
currDir=`pwd`


if echo $PATH | grep -q $ANACONDA_KEYWORD; then
    hasAnaconda=1
    echo "Conda found in path"
fi

if echo $PATH | grep -q $MINICONDA_KEYWORD; then
    hasAnaconda=1
    echo "Miniconda found in path"
fi

if [[ $hasAnaconda -eq 0 ]]; then
    echo "Anaconda/Miniconda not installed - install from $ANACONDA_DOWNLOAD_URL"
    openURL $ANACONDA_DOWNLOAD_URL
    exit 125
fi

# Hacky way of finding the conda base directory
condaPath=`which conda`
condaPath=`dirname ${condaPath}`
condaPath=`dirname ${condaPath}`
# Source conda
source $condaPath/etc/profile.d/conda.sh 

# Check if OS is supported
if [[ "$OSTYPE" != "linux-gnu" && "$OSTYPE" != "darwin"* ]]; then
    echo "Only Linux and MacOS are supported"
    exit 125
fi

# Create Anaconda environment (dosma_env)
if [[ `conda env list | grep $ABCT_ENV_NAME` ]]; then
    if [[ ${updateEnv} -eq 0 ]]; then
        echo "Environment '${ABCT_ENV_NAME}' is installed. Run 'conda activate ${ABCT_ENV_NAME}' to get started."
        exit 0
    else
        conda env remove -n $ABCT_ENV_NAME
        conda create -y -n $ABCT_ENV_NAME python=3.8
    fi
else
    conda create -y -n $ABCT_ENV_NAME python=3.8
fi

conda activate $ABCT_ENV_NAME

# Install tensorflow and keras
# https://www.tensorflow.org/install/source#gpu
pip install tensorflow

# Install pytorch
# FIXME: PyTorch has to be installed with pip to respect setup.py files from nn UNet
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     # Mac
#     if [[ $cudaVersion != "" ]]; then
#         # CPU
#         echo "Cannot install PyTorch with CUDA support on Mac"
#         exit 1
#     fi
#     conda install -y pytorch torchvision torchaudio -c pytorch
# else
#     # Linux
#     if [[ $cudaVersion == "" ]]; then
#         cudatoolkit="cpuonly"
#     else
#         cudatoolkit="cudatoolkit=${cudaVersion}"
#     fi
#     conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 $cudatoolkit -c pytorch
# fi

# Install detectron2
# FIXME: Remove dependency on detectron2
#pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# Install totalSegmentor
# FIXME: Add this to the setup.py file
pip3 install git+https://github.com/StanfordMIMI/TotalSegmentator.git

# cd $currDir/..
# echo $currDir
# exit 1
rm -rf /home/lblankem/.cache

pip install -e .

echo ""
echo ""
echo "Run 'conda activate ${ABCT_ENV_NAME}' to get started."