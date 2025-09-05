import os
import random
import shutil

NUM_IDENTS = 50
NUM_IMGS = 10

d = "data/vggface/"
ids = [id for id in os.listdir(d) if (id != "README.md" and len(os.listdir(d + id)) >= 10)]
keep_ids = random.sample(ids, NUM_IDENTS)
for id in keep_ids:
    ims = os.listdir(d + id)
    keep_imgs = random.sample(ims, NUM_IMGS)
    for im in keep_imgs:
        im_name = im.replace("_", "") #need to get rid of the excess underscore here
        shutil.copy(d + id + "/" + im, "data/vggface_tiny_flat/" + id + "_" + im_name)