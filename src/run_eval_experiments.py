import os
import subprocess

#paths = [path for path in os.listdir("data/") if path.find("facescrub") != -1]
methods = ["l2_sgd_cloak_fawkes_100"]
paths = ["facescrub_small_flat", "webface_small_flat", "vggface_small_flat", "pubfig_small_flat"]
models = ["Facenet", "ArcFaceR100", "CosFaceR100"]

for path in paths:
   for model in models:
      for method in methods:
         prefix = "cloak_" + path + "_" + model + "_" + method
         output_file = "output/evaluate_"+prefix+"_"+model+".out"
         template = open("src/eval_template.txt").read()
         template = template.replace("OUT_PATH", output_file)
         template = template.replace("NONCLOAKED_DATASET_FILE", "data/" + path)
         template = template.replace("CLOAKED_DATASET_FILE", "data/cloaked/" + prefix)
         template = template.replace("MODEL_TO_EVAL", model)
         f = open("autoscripts/evaluate_" + prefix + "_" + model + ".sbatch", 'w')
         f.write(template)
         f.close()
         print("Queued: autoscripts/evaluate_" + prefix)
         subprocess.run(["sbatch", "autoscripts/evaluate_" + prefix + "_" + model + ".sbatch"])
