#!/bin/sh
#SBATCH --output=Conditional_GAN_log_%j.out
pwd; hostname; date
echo "CGAN"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate POC
python3 -u train.py > Conditional_GAN.out