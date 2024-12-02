#!/bin/bash -l 
#SBATCH --job-name=noise2void
#SBATCH --output=output_%j.log
#SBATCH --chdir /scratch/izar/mgatti/ML4Science-CS433/
#SBATCH --ntasks 1
#SBATCH --gres gpu:1

date
source env/bin/activate 
pip3 install -r requirements.txt
python3 main_n2v.py
