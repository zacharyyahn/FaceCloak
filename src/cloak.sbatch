#!/bin/bash
#SBATCH -JFaceCloak
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=4 --mem-per-gpu=32G 
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/evaluate_cloak.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX


# NOTE: Only support batch size of 1 right now because of errors with MTCNN breaking when not detecting a face.


#TODO:
# - Separate out the cloaking function so that it is modular and can incorporate different loss functions. Maybe even make it it's own class?
# - Add a seed functionality so that results are replicable
# - Need to see if MTCNN features can be improved. Does size of input image matter? What about postprocessing?


# ---- Parameters ----
# dataset_path: string path to dataset file
# batch_size: int batch_size for attacks. Currently only option is 1
# target_pool_size: int number of images in the target search pool
# cropped_im_size: int size of cropped image from MTCNN (square image)
# num_cloaked_images: int number of images to cloak
# num_dataset_images: float fraction of dataset to include in calculations

python src/cloak.py --dataset_path "data/pubfig_flat" --batch_size 1 --target_pool_size 100 --cropped_im_size 112 --num_cloaked_images 5 --num_dataset_images 1.0 --cloak_save_path "data/pubfig_cloaked_test" --verbosity "none"
