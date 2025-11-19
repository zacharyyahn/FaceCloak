"""
Handler for running batch experiments or easily configuring experimental settings 
"""

import os
import sys
import subprocess
from itertools import product

dataset = "privacy_common"

distances = ["cosine"]                          # how to compare distances of two embeddings
cloak_funcs = ["pgd_cloak"]                     # which cloak function to use if mode is 'perturb' for single-image perturbations
multi_cloak_funcs = ["afog_cloak_multi"]        # which cloak function to use if mode is 'multi'
do_stickers = ["1"]                             # whether to use region-stickers
do_highpass = ["1"]                             # whether to use highpass-masks
use_real = ["0"]                                # whether to use real or synthetic images
multi_losses = ["triplet"]                      # what loss function to use if mode is 'multi'
losses = ["triplet"]                            # what loss function to use if mode is 'perturb'
iterations = ["0"]                              # how many optimization iterations if mode is 'perturb'
multi_iterations = ["10"]                       # how many optimization iterations if mode is 'multi'
gen_iterations = ["0"]                          # gen iterations
gen_lr = ["0.0"]                                # gen learning rate
models = ["ArcFace"]                            # which model to use as 
probe_dataset = [f"{dataset}/probe_small"]      # which probe set to use
gallery_dataset = [f"{dataset}/gallery_small"]  # which gallery set to use
pert_steps = ["2"]                              # step size of perturbation optimization if mode is 'perturb'
finetune_perts = ["0"]                          # step size of perturbation finetuning (if using)
multi_perts = ["8"]                             # step size of perturbation optimization if mode is 'multi'
modes = ["multi"]                               # mode to use. In our experiments we use 'multi' for all experiments
percep_funcs = ["dssim"]                        # type of perceptual loss, if using
percep_weights = ["0.0"]                        # weight of perceptual loss, if using
num_images_to_generate = ["8"]                  # number of synthetic images to generate per identity
n_to_eval = ["5"]                               # number of images to eval on. Leave as 5 for default experiments
notes = ["none"]                                # additional notes to add to name of output run

for dist, cf, mcf, stickers, highpass, mul_loss, loss, itr, mul_itr, gen_itr, gen_lr, model, probe_data, gal_data, ft_perts, mul_perts, pert_steps, mode, pc_f, pc_w, num_ims, n_eval, use_real, note in product(*[
    distances, 
    cloak_funcs, 
    multi_cloak_funcs,
    do_stickers,
    do_highpass,
    multi_losses, 
    losses,
    iterations,
    multi_iterations,
    gen_iterations,
    gen_lr,
    models,
    probe_dataset,
    gallery_dataset,
    finetune_perts,
    multi_perts,
    pert_steps,
    modes,
    percep_funcs,
    percep_weights,
    num_images_to_generate,
    n_to_eval,
    use_real,
    notes
    ]):
   
    template = open("src/cloak_template.txt",'r')
    template = template.read()

    # Save line for tracking each individual experiment
    save_line = f"cloak_{dataset}_{model}_{mcf}_sticker_{stickers}_highpass_{highpass}_dssim_{pc_w}_max_pert_{mul_perts}_pert_step_{pert_steps}_ims_{num_ims}_{note}"
    
    # Populate the template script
    f = open("autoscripts/" + save_line + ".sbatch", 'w')
    template = template.replace("PROBE_DATASET_PATH", "data/" + probe_data)
    template = template.replace("GALLERY_DATASET_PATH", "data/" + gal_data)
    template = template.replace("DISTANCE",dist)
    template = template.replace("CF",cf)
    template = template.replace("MULTI_CLOAK_FUNC",mcf)
    template = template.replace("DO_STICKERS",stickers)
    template = template.replace("DO_HIGHPASS",highpass)
    template = template.replace("LOSS_TYPE",loss)
    template = template.replace("LOSS_MULTI_TYPE", mul_loss)
    template = template.replace("NUM_ITERATIONS", itr)
    template = template.replace("MULTI_ITERATIONS", mul_itr)
    template = template.replace("EXTRACTOR_TYPE",model)
    template = template.replace("PERT_MAX", ft_perts)
    template = template.replace("MULTI_MAX_PERT", mul_perts)
    template = template.replace("ATTACK_MODE", mode)
    template = template.replace("SAVE_PATH","data/cloaked/" + save_line)
    template = template.replace("OUTPUT_PATH", "output/" + save_line + ".out")
    template = template.replace("SAVE_GEN_PATH", "data/gen/" + save_line)
    template = template.replace("PERT_STEP_SIZE", pert_steps)
    template = template.replace("GEN_IM_LR", gen_lr)
    template = template.replace("NUM_GEN_ITERS", gen_itr)
    template = template.replace("SIM_FUNC", pc_f)
    template = template.replace("SIM_VALUE", pc_w)
    template = template.replace("NITG", num_ims)
    template = template.replace("N_TO_EVAL", n_eval)
    template = template.replace("USE_REAL", use_real)
    f.write(template)
    f.close()

    # Run each script
    subprocess.run(["mkdir","data/cloaked/" + save_line])
    subprocess.run(["mkdir","data/gen/" + save_line ])
    subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
    print("Queued: autoscripts/" + save_line + ".sbatch")



