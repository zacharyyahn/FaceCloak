#!/bin/bash
#SBATCH -JFace
#SBATCH -N1
#SBATCH -t16:00:00
#SBATCH -ooutput/evaluate.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/pubfig_flat/"

python src/evaluate.py --dataset_file $DATASET_FILE --models "ArcFace"
