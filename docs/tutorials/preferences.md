# Configure Preferences
abCTSeg uses preferences to track defaults when performing automated segmentation.
The preference config system uses yaml and [yacs](https://github.com/rbgirshick/yacs).

### Setting Default Preferences
Default preferences can either be set using command-line options or by specifying a
new preferences file. For more information on how to write configuration files in yacs,
see the [yacs project repo](https://github.com/rbgirshick/yacs).

To configure default preferences from the command line, use the script "abctseg/cli.py" with the
`init` sub-command followed by the key value pairs for what you would like to set:

```bash
python abctseg/cli.py config save KEY1 VALUE1 KEY2 VALUE2 ...
```

To configure default preferences from a properly formatted config file, use the `--config-file`
option:

```bash
python abctseg/cli.py config save --config-file /path/to/config/file
```

Note that the two initialization options can be combined, with the command-line specified key/values
taking precedence:

```bash
# Values for KEY1, KEY2, ... will be set to VALUE1, VALUE2, ...
# regardless of what is in the config file
python abctseg/cli.py config save --config-file /path/to/config/file KEY1 VALUE1 KEY2 VALUE2 ...
```

**DO NOT MODIFY `preferences.yaml` directly.**

### Changing Preferences at Runtime
If you want to temporarily change preferences for a specific run, 
you can use the same command-line
options detailed above with any sub-command in abctseg/cli.py:

```bash
python abctseg/cli.py process --dicoms /path/to/dicom/file KEY1 VALUE1 KEY2 VALUE2
```


If you are using abctseg as a library, you can set your preferences at runtime in your code:

```python
from abctseg.preferences import PREFERENCES

# Set preferences manually. 
PREFERENCES.BATCH_SIZE = 32
PREFERENCES.OUTPUT_DIR = "/path/to/data/dir"

# Load preferences from a file.
PREFERENCES.merge_from_file("/path/to/config/file")
...
```


### Config Params

Some common config params are listed below:
+ `OUTPUT_DIR`: Default directory where data will be stored.
+ `CACHE_DIR`: Default directory for abctseg-related caching
+ `MODELS_DIR`: Directory where downloaded models are stored.
+ `BATCH_SIZE`: Batch size for automated segmentation
+ `NUM_WORKERS`: Number of cpu workers for loading data
