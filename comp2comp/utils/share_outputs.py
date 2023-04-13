import os
import shutil
import sys

def isolate_output_type(outputs_dir, output_type, isolated_dir):
    for i, folder in enumerate(os.listdir(outputs_dir)):
        if os.path.isdir(os.path.join(outputs_dir, folder)):
            shutil.copytree(os.path.join(outputs_dir, folder, output_type), os.path.join(isolated_dir, f"{i}"))