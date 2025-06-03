import os
import sys
import subprocess

distances = ["l2"]
optimizers = ["sgd_cloak"]
losses = ["fawkes"]
iterations = ["100"]
models = ["Facenet", "ArcFaceR100", "CosFaceR100"]
datasets = ["facescrub_small_flat", "pubfig_small_flat", "vggface_small_flat", "webface_small_flat"]

for data in datasets:
    prefix = "cloak_" + data
    for d in distances:
        for o in optimizers:
            for l in losses:
                for i in iterations:
                    for m in models:
                        template = open("src/cloak_template.txt",'r')
                        template = template.read()
                        save_line = f"{prefix}_{m}_{d}_{o}_{l}_{i}"
                        f = open("autoscripts/" + save_line + ".sbatch", 'w')
                        template = template.replace("DATASET_PATH", "data/" + data)
                        template = template.replace("DISTANCE",d)
                        template = template.replace("OPTIMIZER",o)
                        template = template.replace("LOSS_TYPE",l)
                        template = template.replace("NUM_ITERATIONS",i)
                        template = template.replace("EXTRACTOR_TYPE",m)
                        template = template.replace("SAVE_PATH","data/cloaked/" + save_line)
                        template = template.replace("OUTPUT_PATH", "output/" + save_line + ".out")
                        f.write(template)
                        f.close()

                        subprocess.run(["mkdir","data/cloaked/" + save_line])
                        subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
                        print("Queued: autoscripts/" + prefix + "_" + save_line + ".sbatch")

