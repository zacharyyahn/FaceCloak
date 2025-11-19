"""
File for defining a dataset of face images, which supports loading from any of the face datasets we use.
"""


from torch.utils.data import Dataset
import random
import os
import cv2

class FaceDataset(Dataset):
    def __init__(self, dataset_path, transforms = None, num_images=1.0):
        self.paths = []
        for path in os.listdir(dataset_path):
            self.paths.append(dataset_path + "/" + path)
        if num_images != 1.0:
            random.shuffle(self.paths)
            self.paths = self.paths[:int(num_images * len(self.paths))]
        print("Loading dataset with", len(self.paths), "images.")

    #NOTE: MAY NEED TO APPLY INPUT TRANSFORMS
    def __getitem__(self, idx):
        path = self.paths[idx]
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, path

    def __len__(self):
        return len(self.paths)

