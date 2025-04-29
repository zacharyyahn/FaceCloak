#!/bin/bash
#SBATCH -JFaceEval
#SBATCH -N1
#SBATCH -t16:00:00
#SBATCH -ooutput/evaluate_deepface_arcface_on_facenet100.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

DATASET_FILE="data/pubfig_flat/"
CLOAK_FILE="data/pubfig_cloaked_facenet/"


python src/deepface_evaluate.py --dataset_file $DATASET_FILE --probe_file $CLOAK_FILE --num_probes 250 --dataset_size 1.0 --gallery_size 5 10 15 20 25 50 75 100 150 200 300 400 500 1000 2000 5000 10000 --models "Arcface"
