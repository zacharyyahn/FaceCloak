#!/bin/bash
#SBATCH -JFaceCloak
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=4 --mem-per-gpu=32G 
#SBATCH -n4 -t16:00:00
#SBATCH -ooutput/test_cloak_facescrub.out

module load anaconda3
conda init
conda deactivate
conda activate Face

echo $CONDA_PREFIX


# NOTE: Only support batch size of 1 right now because of errors with MTCNN breaking when not detecting a face.


#TODO:
# - Add a seed functionality so that results are replicable
# - Need to see if MTCNN features can be improved. Does size of input image matter? What about postprocessing?
# - See why we are sometimes getting None images if they aren't caught, in cloak_image
# - See if all datasets (e.g. FaceScrub and LFW) also need MTCNN or if the pre-crop is enough

# ---- Parameters ----
# dataset_path: string path to dataset file
# batch_size: int batch_size for attacks. Currently only option is 1
# target_pool_size: int number of images in the target search pool
# cropped_im_size: int size of cropped image from MTCNN (square image)
# num_cloaked_images: int number of images to cloak
# num_dataset_images: float fraction of dataset to include in calculations

python src/cloak.py --dataset_path "data/facescrub_small_flat" \
	--batch_size 1 \
	--target_pool_size 100 \
	--cropped_im_size 112 \
	--num_cloaked_images 5000 \
	--num_dataset_images 1.0 \
	--norm_function "tanh" \
	--cloak_save_path "data/cloaked/test_facescrub_small" \
	--extractor_type "Facenet" \
	--distance_function "l2" \
	--cloak_function "sgd_cloak" \
	--cloak_loss "fawkes" \
	--cloak_percep_loss "dssim" \
	--percep_loss_weight 0.3 \
	--cloak_function_iters 100 \
	--cloak_function_step 0.00784 \
	--cloak_function_max_pert 0.0627 \
	--cloak_function_lr 0.5 \
	--verbosity "none"
