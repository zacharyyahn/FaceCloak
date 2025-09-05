import os
import subprocess

#paths = [path for path in os.listdir("data/") if path.find("facescrub") != -1]
methods = [f"test_multi_only_{num}_10.0_32" for num in [2]]
paths = ["vggface_tiny_flat"]
models = ["Facenet"]
eval_model = "Facenet"

notes = None

for path in paths:
   for model in models:
      for method in methods:
         #prefix = "cloak_" + path + "_" + model + "_" + method
         prefix = method

         #prefix = path + "_" + model + "_" + method
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
