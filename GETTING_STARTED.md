## Getting Started with abCTSeg

This document provides a brief intro of the usage of builtin command-line tools in abCTSeg.
We provide a script in "abctseg/cli.py", which is the entry point for all command-line operations.

For more advanced tutorials, refer to our [documentation](https://ad12.github.io/abCTSeg/).

### Downloading pre-trained models
Fill out this 
[questionnaire](https://docs.google.com/forms/d/e/1FAIpQLSdohkf9aQHeRvITgbfpNyy5QteY0aP53EH_enZdReePXJ43mg/viewform?usp=sf_link)
and contact Arjun Desai (arjundd \<at> stanford \<dot> edu) to
get access to the Google Drive folder containing all models. 

Download the models you would like to use to your local machine. The directory where
the models are stored will be your models directory. Please specify the path to this folder
when setting up your preferences (see section below).

### Setting up preferences
To set up your preferences, run the script with the preference options you would like to 
save as defaults.

At minimum, set your data directory (where your data will be stored) and your models directory.
```bash
python abctseg/cli.py config save /
    OUTPUT_DIR /path/to/store/data /
    MODELS_DIR /path/to/models/folder
```


### Evaluating abdominal CT scans
All scans to be segmented must be stored in the DICOM format. 
Data will be saved in the `h5` format.

To segment a scan using the `stanford_v0.0.1` model with 1 gpu, run:

```bash
python -m abctseg.cli process --num-gpus 1 \
    --dicoms /path/to/scan/file \
    --models stanford_v0.0.1
```

To segment multiple scans using the `stanford_v0.0.1` model with 1 gpu, you
run in `batch` mode:

```bash
python -m abctseg.cli process --batch --num-gpus 1 \
    --dicoms /path/to/folder/with/dicoms \
    --models stanford_v0.0.1
```

### Summarizing results
To summarize parameters (HU and CSA) of a segmentation, run:

```bash
python -m abctseg.cli summarize --results-path /path/to/results/folder
```