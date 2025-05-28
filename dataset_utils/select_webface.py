import os
import random
import shutil

NUM_IDENTS = 500
NUM_IMGS = 10

d = "data/webface/"
ids = [id for id in os.listdir(d) if (id != "README.md" and len(os.listdir(d + id)) >= 10)]
keep_ids = random.sample(ids, NUM_IDENTS)
for id in keep_ids:
    ims = os.listdir(d + id)
    keep_imgs = random.sample(ims, NUM_IMGS)
    for im in keep_imgs:
        shutil.copy(d + id + "/" + im, "data/webface_small_flat/" + id + "_" + im)



