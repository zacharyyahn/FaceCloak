import os
import subprocess

paths = [path for path in os.listdir("data/") if path.find("facescrub") != -1]

for path in paths:
   output_file = "output/evaluate_"+path+".out"
   template = open("src/eval_template.txt").read()
   template = template.replace("OUT_PATH", output_file)
   template = template.replace("CLOAKED_DATASET_FILE", "data/" + path)
   f = open("autoscripts/evaluate_" + path + ".sbatch", 'w')
   f.write(template)
   f.close()
   print("Queued: autoscripts/evaluate_" + path)
   subprocess.run(["sbatch", "autoscripts/evaluate_" + path + ".sbatch"])
