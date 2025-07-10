from torch.utils.data import Dataset
import numpy as np
import torch
import os
import cv2
import sys

class AdapterDataset(Dataset):
    def __init__(self, mode="train", dataset_path=None):
        self.image_path = "data/pubfig_small_flat"
        self.masks = []
        self.ims = []
        self.mode = mode
        self.mask_to_im = {} # use this during training to maximize masks

        # Read in all of the masks, combining from each directory
        if self.mode == "train":
            self.mask_paths = [dataset_path]

            for path in self.mask_paths:
                for mask_path in os.listdir(path):
                    im_path = self.image_path + "/" + mask_path[mask_path.find("_")+1:-4] + ".jpg"
                    self.mask_to_im[path + "/" + mask_path] = im_path
            self.masks = list(self.mask_to_im.keys())
        else:
            self.mask_paths = "data/masks/"
            for mask_path in os.listdir(self.mask_paths):
                im_path = self.image_path + "/" + mask_path[mask_path.find("_")+1:-4] + ".jpg"
                self.mask_to_im[self.mask_paths + "/" + mask_path] = im_path
            self.masks = list(self.mask_to_im.keys())

        self.im_to_mask = {im: mask for mask, im in self.mask_to_im.items()}


    def __getitem__(self, idx):
        if self.mode == "train":
            mask_path = self.masks[idx]
            im_path = self.mask_to_im[mask_path]
        else:
            im_path = list(self.im_to_mask.keys())[idx]
            mask_path = self.im_to_mask[im_path]

        mask = torch.Tensor(np.load(mask_path))
        mask = mask.permute(2, 0, 1)
        mask = mask / 255.0

        # Normalize the mask to [-1, 1]
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if im.shape[0] < 200 or im.shape[1] < 200: # if it's too small we need to resize
            im = cv2.resize(im, (im.shape[1] * 5, im.shape[0] * 5), cv2.INTER_CUBIC)
        im = torch.Tensor(im)
        im = im.permute(2, 0, 1)
        im = ((im / 255.0) - 0.5 ) * 2.0
        return im, mask, mask_path

    def __len__(self):
        if self.mode == "train":
            return len(self.masks)
        else:
            return len(self.im_to_mask)
