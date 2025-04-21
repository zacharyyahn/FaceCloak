import torch
from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from facedataset import FaceDataset
from torch.utils.data import DataLoader
import random

# NOTE: Fawkes used a different model/dataset, and they also did some tanh normalization on the images. See differentiator file in the repo or the paper. They also did not clip images.


class Cloaker():
    def __init__(self, dataset_path, extractor, cropper, batch_size):
        self.dataset = FaceDataset(dataset_path)
        self.face_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.extractor = extractor
        self.embeds = {}
        self.cloaked_ims = []
        self.cropper = cropper
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print("Cloaker is using device", self.device)

        for path in os.listdir(dataset_path):
            self.paths.append(dataset_path + "/" + path)

    def get_embeds(self):
        for ims, labels in self.face_loader:
            cropped_ims = self.cropper(ims)
            embeds = self.extractor(cropped_ims)
        
        # Sequentially save the embeds for use in selecting targets
        for i, label in enumerate(list(labels)):
            self.embeds[label] = embeds[i]

    def L2_dist(embed1, embed2):
        return torch.linalg.norm(embed1 - embed2, ord=2)

    # Get the target image according to Fawkes method by finding the maximally different image in a pool
    def get_target(self, path, pool_size=100):
        this_embed = self.embeds[path]
        path_name = os.path.basename(path_name)[:os.path.basename(path_name).find("_")]
        best_dist = 10000000
        best_target = None
        while pool_size > 0:
            idx = random.randint(0, len(self.embeds))
            tgt = list(self.embeds.keys())[idx]
            name = os.path.basename(tgt)[:os.path.basename(tgt).find("_")]
            if name == path_name: # if they're the same class ignore
                continue
            else:
                pool_size -= 1
                tgt_embed = self.embeds[tgt]
                sim = L2_dist(tgt_embed, this_embed)
                if sim < best_dist:
                    best_dist = sim
                    best_target = tgt
        print("Target for", path, "is", best_target)
        return best_target

    def weak_triplet_loss(self, embed_img, embed_tgt, embed_orig, factor):
        return torch.linalg.norm(embed_img - embed_tgt, ord=2) + factor * torch.linalg.norm(embed_img - embed_orig, ord=2)
    
    def cloak_image(self, img, tgt, iters=1000, lr=1e-2, pert_budget=8/255.):
        tgt_emb = self.embeds[tgt].clone()

        im = cv2.imread(img)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cropped = self.cropper(im)
        cropped = cropped.repeat((2, 1, 1, 1)) # batch the input because of a quirk with mozuma
        cropped.requires_grad = True

        orig_embed = self.extractor(cropped)
        optimizer = torch.optim.Adam([cropped], lr=1e-2)
        
        pbar = tqdm(range(iters))
        for i in pbar:
            optimizer.zero_grad()

            out_emb = self.extractor(cropped)
            loss = self.weak_triplet_loss(out_emb, tgt_emb, orig_embed, factor=0.1)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})

        return cropped[0]


    def cloak_all(self, save_dir=None):
        for path in self.paths:
            tgt = self.get_target(path)
            cloaked_cropped = self.cloak_image(path, tgt)
        if save_dir is not None:
            cv2.imsave(cloaked_cropped, save_dir + "/" + path)

    def apply_defense(self):
        None
