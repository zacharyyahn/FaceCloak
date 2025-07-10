import os
import sys
import subprocess

makeup_methods = ["amt-gan","diffam"]
#datasets = ["facescrub_small_flat", "pubfig_small_flat", "vggface_small_flat", "webface_small_flat"]
datasets = ["vggface_small_flat"]

for data in datasets:
    prefix = "makeup_" + data
    for m in makeup_methods:
        template = open("src/makeup_template.txt",'r')
        template = template.read()
        save_line = f"{prefix}_{m}"
        f = open("autoscripts/" + save_line + ".sbatch", 'w')
        template = template.replace("DATASET_PATH", "data/" + data)
        template = template.replace("MAKEUP_METHOD", m)
        template = template.replace("SAVE_PATH","data/cloaked/" + save_line)
        template = template.replace("OUTPUT_PATH", "output/" + save_line + ".out")
        f.write(template)
        f.close()

        subprocess.run(["mkdir","data/cloaked/" + save_line])
        subprocess.run(["sbatch","autoscripts/" + save_line + ".sbatch"])
        print("Queued: autoscripts/" + prefix + "_" + save_line + ".sbatch")

