#!/bin/bash
#SBATCH -JFaceEval
#SBATCH -N1
#SBATCH -t16:00:00
#SBATCH -ooutput/evaluate.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/pubfig_flat/"
CLOAK_FILE="data/pubfig_cloaked/"


python src/evaluate.py --dataset_file $DATASET_FILE --cloaked_file $CLOAK_FILE --num_to_evaluate 200 --dataset_size 0.25 --models "ArcFace" --attack
