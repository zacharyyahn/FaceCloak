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
from torchvision import transforms as trans

# NOTE: Fawkes used a different model/dataset, and they also did some tanh normalization on the images. See differentiator file in the repo or the paper. They also did not clip images.


class Cloaker():
    def __init__(self, dataset_path, extractor, cropper, batch_size):
        self.dataset = FaceDataset(dataset_path)
        self.paths = []
        self.face_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.extractor = extractor
        self.embeds = {}
        self.cloaked_ims = []
        self.cropper = cropper
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
        # Define transforms for normalizing images. Can change them on a per-dataset basis
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        self.norm_transform = trans.Compose([
                    trans.Normalize(mean, std)
                ])
        self.inv_transform = trans.Compose([
                    trans.Normalize((-mean / std), 1.0 / std)
        ])

        # Map of each path to a face box
        self.boxes = {}
        
        # Map of each path to a cropped face box
        self.cropped_images = {}

        # Map of each path to the original image, necessary for reconstructing the face
        self.images = {}
        
        # Set out models in evaluation mode
        self.extractor.eval()
        self.cropper.eval()

        print("Cloaker is using device", self.device)
        
        # Read in all of the images
        self.paths = self.dataset.paths
        for path in self.paths:
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.images[path] = im

    # Get the embeddings of every image in the dataset by first cropping with MTCNN and then running through an extractor
    def get_embeds(self):
        for im, label in tqdm(self.face_loader):
            im = im.squeeze()
            label = str(label[0])
            #print("Looking at label", label, "for im of size", im.size())
            boxes, _ = self.cropper.detect(im)

            if len(boxes) != 1:
                print("ERROR: Boxes are", boxes)
                self.paths.remove(label)
                continue

            #print("Boxes are", boxes)
            self.boxes[label] = boxes[0]

            xmin = int(boxes[0][0])
            ymin = int(boxes[0][1])
            xmax = int(boxes[0][2])
            ymax = int(boxes[0][3])

            # Crop the area with the bounding box
            crop = im[ymin:ymax, xmin:xmax, :]
            #print("crop has shape", crop.size())

            # Resize to be 112x112
            crop = torch.Tensor(cv2.resize(crop.detach().cpu().numpy(), (112, 112), interpolation=cv2.INTER_LINEAR))
            crop = torch.permute(crop, (2, 0, 1))
            
            # Apply the normalization transform
            #print("Before transform, crop has range", torch.min(crop), torch.max(crop))
            crop /= 255.0
            crop = self.norm_transform(crop)
            #print("After transform, crop has range", torch.min(crop), torch.max(crop))

            # Add fake batching to work with mozuma
            crop = crop.repeat((2, 1, 1, 1))
            self.cropped_images[label] = crop
            
            # Get the embeddings and save them
            embeds = self.extractor(crop)
            #print("Finaly embed has shape", embeds.size())
            self.embeds[label] = embeds
        
        print("Finished embedding")

    # Calculate the L2 distance of two vectors
    def L2_dist(self, embed1, embed2):
        return torch.linalg.norm(embed1 - embed2, ord=2)

    # Get the target image according to Fawkes method by finding the maximally different image in a pool
    def get_target(self, path, pool_size=100):
        try:
            this_embed = self.embeds[path]
        except:
            return None
        
        # Get the name of the person we're looking at
        path_name = os.path.basename(path)[:os.path.basename(path).find("_")]
        best_dist = 0
        best_target = None
        
        # Repeat for the number of images in the pool
        while pool_size > 0:

            # Get a random index
            idx = random.randint(0, len(self.embeds)-1)
            tgt = list(self.embeds.keys())[idx]
            
            # Get the name for this target
            name = os.path.basename(tgt)[:os.path.basename(tgt).find("_")]
            
            # If they're the same class, ignore
            if name == path_name:
                continue
            else:
                tgt_embed = self.embeds[tgt]
                if torch.max(tgt_embed == 0): # make sure we aren't working towards a null embedding
                    continue
                pool_size -= 1
                sim = self.L2_dist(tgt_embed, this_embed)
                    
                # If this one is every farther away, then save it and continue
                if sim > best_dist:
                    best_dist = sim
                    best_target = tgt
        print("Target for", path, "is", best_target)
        return best_target

    def weak_triplet_loss(self, embed_img, embed_tgt, embed_orig, factor):
        return torch.linalg.norm(embed_img - embed_tgt, ord=2)# + factor * torch.linalg.norm(embed_img - embed_orig, ord=2)
    
    # Cloak a single image by making its feature space embedding more similar to another maximally different target image's embedding. Use 1000 iterations of Adam with a maximum perturbation budget (different from Fawkes)
    def cloak_image(self, img_path, tgt_path, iters=20, lr=1e-2, pert_budget=8./255., step=1./255.):
        
        # Retrieve the pre-computed target embedding
        tgt_emb = self.embeds[tgt_path].clone()
        
        # Retrieve the pre-computed image embedding
        cropped = self.cropped_images[img_path]

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.extractor(cropped).detach()
            tgt_emb = tgt_emb.detach()

        # Copy and save the original to measure modification
        orig_cropped = cropped.clone()

        # Record the original dimensions of cropped
        orig_min = torch.min(cropped)
        orig_max = torch.max(cropped)
        #print("orig_min and orig_max are", orig_min, orig_max)

        # Adjust the perturbation budget and step size
        pert_budget = pert_budget * (orig_max - orig_min)
        step = step * (orig_max - orig_min)

        # Make sure cropped will receive gradients as a fresh leaf tensor
        cropped.requires_grad = True
        cropped = cropped.clone().detach().requires_grad_()

        optimizer = torch.optim.Adam([cropped], lr=1e-2)
        
        # Iterate for iters iterations
        pbar = tqdm(range(iters))
        for i in pbar:
            cropped = cropped.clone().detach()
            cropped.requires_grad = True
            out_emb = self.extractor(cropped)
            loss = self.weak_triplet_loss(out_emb, tgt_emb, orig_embed, factor=0.3)
            loss.backward()
            pbar.set_postfix({'loss': loss.item()})
            #print("Loss is", loss.item())
            
            # Get the sign of the gradient
            signed_grad = torch.sign(cropped.grad)
            
            # Update cropped
            cropped = cropped - step * signed_grad
            
            #print("If we didn't clip, difference would be", (cropped - orig_cropped).abs().mean())
            # Clip the perturbation to be within the viable range
            pert = torch.clip(cropped - orig_cropped, min=-pert_budget, max=pert_budget)
            #print("Mean of pert is", pert.abs().mean())
            # Add the perturbation back, but make sure we're within the range of the original cropped
            cropped = torch.clip(orig_cropped + pert, min=orig_min, max=orig_max)
            #print("Since we did clip, difference is", (cropped - orig_cropped).abs().mean())
        
        diff = torch.clip(cropped - orig_cropped, min=-pert_budget, max=pert_budget)
        #print("Here max diff is", torch.max(diff))
        cropped = torch.clip(orig_cropped + diff, min=orig_min, max=orig_max)
        return cropped[0]

    # Cloak all images in the dataset
    def cloak_all(self, save_dir=None):

        # Iterate through each item in the dataset
        for path in self.paths:

            print("Cloaking image", path)
            
            # Find the maximally different image
            tgt = self.get_target(path)

            if tgt == None:
                print("No target for image", path)
                continue
            
            # Cloak the image and get the returned crop value
            cloaked_cropped = self.cloak_image(path, tgt)

            # Un-normalize to get back to image space
            cloaked_cropped = self.inv_transform(cloaked_cropped)

            # Return to numpy array
            cloaked_cropped = cloaked_cropped.squeeze().detach().cpu().numpy()
            print("Pre range fixing (post transform) its", np.min(cloaked_cropped), np.max(cloaked_cropped))
            cloaked_cropped = 255. * cloaked_cropped
            cloaked_cropped = np.transpose(cloaked_cropped, (1, 2, 0)) # set channel last
            
            # Re-arrange axes to fit with the original image
            boxes = self.boxes[path]
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])
            cloaked_cropped = cv2.resize(cloaked_cropped, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_CUBIC)

            # Retrieve the image and add the patch
            im = self.images[path]
            print("after retrieval im has range", np.min(im), np.max(im))
            im[ymin:ymax, xmin:xmax, :] = cloaked_cropped
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            print("Before saving im have shape", im.shape)
            print("Before saving my range is", np.min(im), np.max(im))
            if save_dir is not None:
                print("Saving to", save_dir + os.path.basename(path))
                cv2.imwrite(save_dir + os.path.basename(path)[:os.path.basename(path).find(".")] + ".png", im)

    def apply_defense(self):
        None
