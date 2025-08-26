import os
import sys
import subprocess

distances = ["cosine"]
optimizers = ["pgd_cloak"]
multi_losses = ["triplet"]
losses = ["triplet"]
iterations = ["0"]
multi_iterations = ["10"]
models = ["Facenet"]
datasets = ["pubfig_small_flat"]
finetune_perts = ["32"]
multi_perts = ["32"]
modes = ["multi_finetune"]
dssims = ["0.0"]



for data in datasets:
    prefix = "cloak_" + data
    for d in distances:
        for o in optimizers:
            for l in losses:
                for i in iterations:
                    for m in models:
                        for p in finetune_perts:
                            for mp in multi_perts:
                                for t in modes:
                                    for q in multi_losses:
                                        for z in multi_iterations:
                                            for dssim in dssims:
                                                template = open("src/cloak_template.txt",'r')
                                                template = template.read()
                                                #note = f"_dssim_{dssim}_{mp}_p_{p}"
                                                # if note == None:
                                                #     save_line = f"{prefix}_{m}_{d}_{o}_{q}_{z}_{l}_{i}_{p}_{t}"
                                                # else:
                                                #     save_line = f"{prefix}_{m}_{d}_{o}_{q}_{z}_{l}_{i}_{p}_{t}_{note}"
                                                save_line = f"test_multi_only"
                                                f = open("autoscripts/" + save_line + ".sbatch", 'w')
                                                template = template.replace("DATASET_PATH", "data/" + data)
                                                template = template.replace("DISTANCE",d)
                                                template = template.replace("OPTIMIZER",o)
                                                template = template.replace("LOSS_TYPE",l)
                                                template = template.replace("LOSS_MULTI_TYPE", q)
                                                template = template.replace("NUM_ITERATIONS",i)
                                                template = template.replace("MULTI_ITERATIONS", z)
                                                template = template.replace("EXTRACTOR_TYPE",m)
                                                template = template.replace("PERT_MAX", p)
                                                template = template.replace("MULTI_MAX_PERT", mp)
                                                template = template.replace("ATTACK_MODE", t)
                                                template = template.replace("SAVE_PATH","data/cloaked/" + save_line)
                                                template = template.replace("OUTPUT_PATH", "output/" + save_line + ".out")
                                                template = template.replace("SAVE_GEN_PATH", "data/gen/" + save_line)
                                                template = template.replace("DSSIM_VALUE", dssim)
                                                f.write(template)
                                                f.close()

                                                subprocess.run(["mkdir","data/cloaked/" + save_line])
                                                subprocess.run(["mkdir","data/gen/" + save_line ])
                                                subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
                                                print("Queued: autoscripts/" + prefix + "_" + save_line + ".sbatch")

