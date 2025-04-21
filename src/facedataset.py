from torch.utils.data import Dataset
import os
import cv2

class FaceDataset(Dataset):
    def __init__(self, dataset_path, transforms = None):
        self.paths = []
        self.transforms = transforms
        self.tanh_constant = 2 - 1e-6
        for path in os.listdir(dataset_path):
            self.paths.append(dataset_path + "/" + path)

    #NOTE: MAY NEED TO APPLY INPUT TRANSFORMS
    def __getitem__(self, idx):
        path = self.paths[idx]
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, os.path.basename(path)[:os.path.basename(path).find("_")]

    def __len__(self):
        return len(self.paths)

