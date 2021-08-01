#!/bin/sh
#SBATCH --output=SIM_CLR-LR_log_%j.out
pwd; hostname; date
echo "SIM_CLR"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
conda activate POC
#python3 -u train.py > SIM_CLR.out

python3 -u predict_LR.py > SIM_CLR_LR.out