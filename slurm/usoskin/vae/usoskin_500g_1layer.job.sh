#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:k80:1
#SBATCH -t 24:00:00
#echo commands to stdout
set -x

pylon5="/pylon5/mc4s8ap/kchen8"
project_name="single-cell-deep-learning"

# Move to working directory
cd ${pylon5}/${project_name}

# Load Keras
module load tensorflow/1.0.1_anaconda
source activate $TENSORFLOW_ENV

python main.py "usoskin.vae.Train500g1LayerUsoskinVAE" -s 1013
