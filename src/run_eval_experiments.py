import os
import subprocess
from itertools import product

#paths = [path for path in os.listdir("data/") if path.find("facescrub") != -1]
#methods = [f"cloak_{method}_dssim_{dssim}_pert_{pert}_{ims}ims" for method, dssim, pert, ims in product(["afog_cloak_multi"],[0.0, 1.0], [1], [4, 16, 64, 128])]
#methods = [f"cloak_perturb_best_{dataset}_ft_itr_10" for dataset in ["pubfig_small_flat", "vggface_tiny_flat", "webface_tiny_flat", "facescrub_tiny_flat"] ]
methods = [f"cloak_{type}_cloak_multi_best_webface_tiny_flat_ft_itr_{itr}_{ims}ims" for type, itr, ims in product(["afog", "pgd"], ["0", "2", "10"], ["4","64"])]
paths = ["webface_tiny_flat"]
models = ["Facenet"]
eval_model = "Facenet"

notes = None

for path in paths:
   for model in models:
      for method in methods:
         #prefix = "cloak_" + path + "_" + method
         prefix = method
         if notes == None:
            output_file = "output/evaluate_"+prefix+".out"
         else:
            output_file = "output/evaluate_"+prefix+"_"+notes+".out"
         template = open("src/eval_template.txt").read()
         template = template.replace("OUT_PATH", output_file)
         template = template.replace("NONCLOAKED_DATASET_FILE", "data/" + path)
         template = template.replace("CLOAKED_DATASET_FILE", "data/cloaked/" + prefix)
         #template = template.replace("CLOAKED_DATASET_FILE", "data/cloaked/" + prefix)
         template = template.replace("MODEL_TO_EVAL", eval_model) #NEED TO CHANGE BACK TO model
         f = open("autoscripts/evaluate_" + prefix + ".sbatch", 'w')
         f.write(template)
         f.close()
         print("Queued: autoscripts/evaluate_" + prefix)
         subprocess.run(["sbatch", "autoscripts/evaluate_" + prefix + ".sbatch"])
