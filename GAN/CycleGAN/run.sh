#!/bin/sh
#SBATCH --output=Cycle-GAN_log_%j.out
pwd; hostname; date
echo "Cycle GAN"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate POC
python3 -u train.py > Cycle_GAN.out