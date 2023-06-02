#!/bin/bash
#SBATCH --job-name=Name                         # Job name
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=uxxxxxxx@anu.edu.au         # Where to send mail
#SBATCH --ntasks=8                              # Run on a single CPU
#SBATCH --mem=16G                                # Job memory request
#SBATCH --time=5-00:00:00                         # Time limit hrs:min:sec
#SBATCH --partition=gpu                         # Partition to schedule to
#SBATCH --gres=gpu:a6000:1     				        # Request a GPU allocation
#SBATCH --output=JOBNAME_%j.log                 # Standard output and error log
#SBATCH --qos=normal

pwd; hostname; date
echo "--- starting ---"
# Job commands here
echo "Print NVIDIA INFO"
nvidia-smi
echo "Loading container"

singularity exec --nv /opt/apps/containers/pytorch22.01-libx11-1.sif \
python train.py sam --loss=bd --train_decoder --sub decoder_only

echo "--- done ---"
date