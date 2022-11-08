import os

def python_submit(command, node = None):
    bash_file = open("./slurm.sh","w")
    bash_file.write(f'#!/bin/bash\n{command}')
    bash_file.close()
    if node == None:
        os.system('sbatch -c 8 --gres=gpu:1 --output ./slurm/slurm-%j.out --mem=256000 --time=30-00:00:00 slurm.sh ')
    else:
        os.system(f'sbatch -c 8 --gres=gpu:1 --output ./slurm/slurm-%j.out --nodelist={node} --mem=256000 --time=30-00:00:00 slurm.sh')
    os.remove("./slurm.sh")

#python abctseg/cli.py config save OUTPUT_DIR /bmrNAS/people/lblankem/abCTSeg MODELS_DIR /dataNAS/people/lblankem/abCTSeg_scratch

#python_submit("python -m abctseg.cli process --num-gpus 1 --dicoms /bmrNAS/scandata/abct_10_2021_examples/1.2.840.4267.32.24683016766545858583405016243455296198/1.2.840.4267.32.211081843080331613040924942706453851410/000076-1.2.840.4267.32.210318385502561945747692732255298835916.dcm --models stanford_v0.0.1")

#python -m abctseg.cli summarize --results-path /bmrNAS/people/lblankem/abCTSeg

python_submit("bin/C2C process_3d", "siena")