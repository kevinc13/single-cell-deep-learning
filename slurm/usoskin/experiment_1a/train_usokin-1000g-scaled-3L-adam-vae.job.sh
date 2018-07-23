#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node 7
#SBATCH -t 48:00:00
#echo commands to stdout
set -x

pylon5="/pylon5/ms4s84p/kchen8"
project_name="single-cell-deep-learning"

# Move to working directory
cd $pylon5/$project_name

# Load TensorFlow & CUDA
module load tensorflow/1.1.0_nogpu
module load keras/2.0.4

# Activate TensorFlow virtualenv
source $KERAS_ENV/bin/activate

python main.py --e "train_usokin-1000g-scaled-3L-adam-vae"
