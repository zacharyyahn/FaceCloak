import os
import sys
import subprocess
from itertools import product

dataset = "privacy_celeb"

distances = ["cosine"]
cloak_funcs = ["pgd_cloak"]
multi_cloak_funcs = ["afog_cloak_multi"]
do_stickers = ["1"]
do_highpass = ["1"]
multi_losses = ["triplet"]
losses = ["triplet"]
iterations = ["0"]
multi_iterations = ["10"]
gen_iterations = ["0"]
gen_lr = ["0.0"]
models = ["ArcFace"]
probe_dataset = [f"{dataset}/figure"]
gallery_dataset = [f"{dataset}/probe"]
pert_steps = ["2"]
finetune_perts = ["0"]
multi_perts = ["8"]
modes = ["multi"]
percep_funcs = ["dssim"]
percep_weights = ["0.0"]
num_images_to_generate = ["8"]
n_to_eval = ["5"]
notes = ["none"]

for dist, cf, mcf, stickers, highpass, mul_loss, loss, itr, mul_itr, gen_itr, gen_lr, model, probe_data, gal_data, ft_perts, mul_perts, pert_steps, mode, pc_f, pc_w, num_ims, n_eval, note in product(*[
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
    notes
    ]):

    template = open("src/cloak_template.txt",'r')
    template = template.read()
    #save_line = f"cloak_{dataset}_{model}_{mcf}_sticker_{stickers}_highpass_{highpass}_dssim_{pc_w}_max_pert_{mul_perts}_pert_step_{pert_steps}_ims_{num_ims}_{note}"
    save_line = "cloak_figures_16"
    #save_line = f"cloak_{model}_{mcf}_ims_{num_ims}_dssim_{pc_w}"
    #save_line = "test_highpass_sticker"
    # if mode == "multi_finetune":
    #     save_line = f"cloak_{data}_{mode}_{model}_{dist}_{mcf}_{mul_loss}_{mul_itr}_{mul_perts}_{cf}_{loss}_{itr}_{ft_perts}_{pert_steps}_{pc_f}_{pc_w}_{note}"
    #     print("save line is:", save_line)
    # elif mode == "minmax":
    #     save_line = f"cloak_{data}_{mode}_{model}_{dist}_{mcf}_{mul_loss}_{mul_itr}_{mul_perts}_{pert_steps}_{gen_itr}_{gen_lr}_{itr}_{pc_f}_{pc_w}_{note}"
    #     print("save line is:", save_line)

    #note = f"_dssim_{dssim}_{mp}_p_{p}"
    # if note == None:
    #     save_line = f"{prefix}_{m}_{d}_{o}_{q}_{z}_{l}_{i}_{p}_{t}"
    # else:
    #     save_line = f"{prefix}_{m}_{d}_{o}_{q}_{z}_{l}_{i}_{p}_{t}_{note}"
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
    f.write(template)
    f.close()

    subprocess.run(["mkdir","data/cloaked/" + save_line])
    subprocess.run(["mkdir","data/gen/" + save_line ])
    subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
    print("Queued: autoscripts/" + save_line + ".sbatch")



