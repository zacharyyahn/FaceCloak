import os
import sys
import subprocess
from itertools import product

distances = ["cosine"]
cloak_funcs = ["afog_cloak"]
multi_cloak_funcs = ["afog_cloak_multi","pgd_cloak_multi"]
multi_losses = ["triplet"]
losses = ["triplet"]
iterations = ["0","2","10"]
multi_iterations = ["10"]
gen_iterations = ["0"]
gen_lr = ["0.0"]
models = ["Facenet"]
datasets = ["webface_tiny_flat"]
pert_steps = ["2"]
finetune_perts = ["16"]
multi_perts = ["16"]
modes = ["multi_finetune"]
percep_funcs = ["dssim"]
percep_weights = ["1.0"]
num_images_to_generate = ["4","64"]
notes = ["none"]

for dist, cf, mcf, mul_loss, loss, itr, mul_itr, gen_itr, gen_lr, model, data, ft_perts, mul_perts, pert_steps, mode, pc_f, pc_w, num_ims, note in product(*[
    distances, 
    cloak_funcs, 
    multi_cloak_funcs,
    multi_losses, 
    losses,
    iterations,
    multi_iterations,
    gen_iterations,
    gen_lr,
    models,
    datasets,
    finetune_perts,
    multi_perts,
    pert_steps,
    modes,
    percep_funcs,
    percep_weights,
    num_images_to_generate,
    notes
    ]):

    template = open("src/cloak_template.txt",'r')
    template = template.read()
    #save_line = f"cloak_{mcf}_dssim_{pc_w}_pert_{pert_steps}_{num_ims}ims"
    save_line = f"cloak_{mcf}_best_{data}_ft_itr_{itr}_{num_ims}ims"
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
    template = template.replace("DATASET_PATH", "data/" + data)
    template = template.replace("DISTANCE",dist)
    template = template.replace("CF",cf)
    template = template.replace("MULTI_CLOAK_FUNC",mcf)
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
    f.write(template)
    f.close()

    subprocess.run(["mkdir","data/cloaked/" + save_line])
    subprocess.run(["mkdir","data/gen/" + save_line ])
    subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
    print("Queued: autoscripts/" + save_line + ".sbatch")



