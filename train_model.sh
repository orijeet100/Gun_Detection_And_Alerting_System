#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --job-name=gpu_run
#SBATCH --mem=4GB
#SBATCH --ntasks=1
module load python/3.8.1
source "/home/mukherjee.o/PycharmProjects/Gun_Detection/venv/bin/activate"
python -m pip install --upgrade pip
python3 "/home/mukherjee.o/PycharmProjects/Gun_Detection/train.py"