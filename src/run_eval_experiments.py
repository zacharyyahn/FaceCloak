import os
import subprocess
from itertools import product

#paths = [path for path in os.listdir("data/") if path.find("facescrub") != -1]
#methods = [f"cloak_{method}_dssim_{dssim}_pert_{pert}_{ims}ims" for method, dssim, pert, ims in product(["afog_cloak_multi"],[0.0, 1.0], [1], [4, 16, 64, 128])]
#methods = [f"cloak_perturb_best_{dataset}_ft_itr_10" for dataset in ["pubfig_small_flat", "vggface_tiny_flat", "webface_tiny_flat", "facescrub_tiny_flat"] ]
#methods = [f"cloak_afog_{mul_perts}_32_pc_w_{pc_w}_{num_ims}ims" for mul_perts, pc_w, num_ims in product(["4","8","16"], ["0.0","5.0", "10.0"],["1","4","16"])]
methods = ["test_no_resize"]
#methods = ["new_perturb_test"]
gallery_dataset = ["privacy_common/gallery"]
models = ["ArcFaceR100"]

notes = None

for gal_path in gallery_dataset:
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
         template = template.replace("NONCLOAKED_DATASET_FILE", "data/" + gal_path)
         #template = template.replace("CLOAKED_DATASET_FILE", "data/cloaked/" + prefix)
         #template = template.replace("CLOAKED_DATASET_FILE", "data/privacy_common/probe")
         template = template.replace("CLOAKED_DATASET_FILE", "data/cloaked/" + prefix)
         template = template.replace("MODEL_TO_EVAL", model)
         f = open("autoscripts/evaluate_" + prefix + ".sbatch", 'w')
         f.write(template)
         f.close()
         print("Queued: autoscripts/evaluate_" + prefix)
         subprocess.run(["sbatch", "autoscripts/evaluate_" + prefix + ".sbatch"])
