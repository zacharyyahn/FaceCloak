import os
import sys
import subprocess

distances = ["l2", "cosine"]
optimizers = ["sgd_cloak", "pgd_cloak"]
losses = ["fawkes","triplet"]
iterations = ["10", "100"]
models = ["Facenet"]

prefix = "cloak_pubfig"

for d in distances:
    for o in optimizers:
        for l in losses:
            for i in iterations:
                for m in models:
                    template = open("src/cloak_template.txt",'r')
                    template = template.read()
                    save_line = f"{prefix}_{m}_{d}_{o}_{l}_{i}"
                    f = open("autoscripts/" + save_line + ".sbatch", 'w')
                    template = template.replace("DISTANCE",d)
                    template = template.replace("OPTIMIZER",o)
                    template = template.replace("LOSS_TYPE",l)
                    template = template.replace("NUM_ITERATIONS",i)
                    template = template.replace("EXTRACTOR_TYPE",m)
                    template = template.replace("SAVE_PATH","data/" + save_line)
                    template = template.replace("OUTPUT_PATH", "output/" + save_line + ".out")
                    f.write(template)
                    f.close()

                    subprocess.run(["mkdir","data/" + save_line])
                    subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
                    print("Queued: autoscripts/" + prefix + "_" + save_line + ".sbatch")

