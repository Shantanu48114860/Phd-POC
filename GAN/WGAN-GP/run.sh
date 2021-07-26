#!/bin/sh
#SBATCH --output=WGAN_GP_log_%j.out
pwd; hostname; date
echo "WGAN-GP"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate POC
python3 -u train.py > WGAN-GP.out