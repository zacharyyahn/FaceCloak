import os
import random
import shutil

# NOTE: may need to re-download pubfig since some of the images were moved instead of copied, and then deleted

d = "data/pubfig83/"
celebs = os.listdir(d)
celebs_to_sample = 50
imgs_to_sample = 10
keep_celebs = random.sample(celebs, celebs_to_sample)

for celeb in keep_celebs:
    ims = os.listdir(d + celeb)
    ims = random.sample(ims, imgs_to_sample)
    for im in ims:
        celeb_name = celeb.replace(" ","")
        shutil.copy(d + celeb + "/" + im, "data/pubfig_small_flat/"+celeb_name+"_"+im)

