import os
from collections import defaultdict
import random
import shutil

NUM_CELEBS = 50
NUM_IMGS = 10

def find_underscore_backwards(word):
    for i in range(len(word)-1, 0, -1):
        if word[i] == "_":
            return i
    return -1

d = "data/facescrub_flat/"
name_to_num = defaultdict(int)
unique_names = []
for path in os.listdir(d):
    underscore_loc = find_underscore_backwards(path)
    name = path[:underscore_loc]
    name_to_num[name] += 1
    if name not in unique_names:
        unique_names.append(name)

keep_names = random.sample(unique_names, NUM_CELEBS)
for name in keep_names:
    images_for_this_name = [path for path in os.listdir(d) if path[:find_underscore_backwards(path)] == name]
    keep_images = random.sample(images_for_this_name, NUM_IMGS)
    for im in keep_images:
        name = im[:find_underscore_backwards(im)].replace(" ","")
        num = im[find_underscore_backwards(im)+1:]
        shutil.copy(d + im, "data/facescrub_tiny_flat/" + name + "_" + num)