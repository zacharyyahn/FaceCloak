#!/bin/bash
#SBATCH -JFaceCloak
#SBATCH -N1 --gres=gpu:A100:1 --ntasks-per-node=8 --mem-per-gpu=80G 
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/cloak_pubfig_small_flat_Facenet_cosine_pgd_cloak_triplet_8.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX

nvidia-smi


# NOTE: Only support batch size of 1 right now because of errors with MTCNN breaking when not detecting a face.


#TODO:
# - Add a seed functionality so that results are replicable
# - Need to see if MTCNN features can be improved. Does size of input image matter? What about postprocessing?
# - See if all datasets (e.g. FaceScrub and LFW) also need MTCNN or if the pre-crop is enough
# - Get specific details on makeup methods to see what model/dataset they are customized to attack

# ---- Parameters ----
# dataset_path: string path to dataset file
# batch_size: int batch_size for attacks. Currently only option is 1
# target_pool_size: int number of images in the target search pool
# cropped_im_size: int size of cropped image from MTCNN (square image)
# num_cloaked_images: int number of images to cloak
# num_dataset_images: float fraction of dataset to include in calculations

python src/cloak.py \
	--dataset_path "data/pubfig_small_flat" \
	--batch_size 1 \
	--target_pool_size 100 \
	--cropped_im_size 112 \
	--num_cloaked_images 500 \
	--num_dataset_images 1.0 \
	--norm_function "tanh" \
	--cloak_save_path "data/cloaked/cloak_pubfig_small_flat_Facenet_cosine_pgd_cloak_triplet_8" \
	--extractor_type "Facenet" \
	--distance_function "cosine" \
	--cloak_function "pgd_cloak" \
	--multi_cloak_function "pgd_cloak_multi" \
	--cloak_loss "triplet" \
	--multi_cloak_loss "triplet" \
	--cloak_percep_loss "dssim" \
	--percep_loss_weight 0.3 \
	--cloak_function_iters 10 \
	--multi_cloak_function_iters 10 \
	--cloak_function_step 2 \
	--cloak_function_max_pert 16 \
	--cloak_function_lr 0.5 \
	--verbosity "error" \
	--mode "perturb" \
	--gen_save_path "data/gen/cloak_pubfig_small_flat_multi_ArcFaceR100"
	#--makeup_mode "AMT-GAN" \
