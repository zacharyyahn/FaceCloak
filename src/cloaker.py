import torch
from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from facedataset import FaceDataset
from torch.utils.data import DataLoader
import random
import time
from torchvision import transforms as trans

# NOTE: Fawkes used a different model/dataset, and they also did some tanh normalization on the images. See differentiator file in the repo or the paper. They also did not clip images.


class Cloaker():
    def __init__(self, dataset_path, extractor, cropper, batch_size, cropped_im_size, target_pool_size, num_dataset_images, device, verbosity):
        self.dataset = FaceDataset(dataset_path, num_images=num_dataset_images)
        self.paths = []
        self.face_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.extractor = extractor
        self.embeds = {}
        self.cloaked_ims = []
        self.cropper = cropper
        self.batch_size = batch_size
        self.device = device
        self.cropped_im_size = cropped_im_size
        self.target_pool_size = target_pool_size
        self.tanh_constant = 2 - 1e-6
        self.verbosity = verbosity
    
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

        # Map of each cloaked image
        self.cloaked_images = {}
        
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

    def preprocess_tanh(self, im):
        im /= 255.0
        im -= 0.5
        im *= self.tanh_constant
        im = torch.tanh(im)
        return im

    def reverse_tanh(self, im):
        im = (np.arctanh(im) / self.tanh_constant + 0.5) * 255.0
        return im

    # Get the embeddings of just one image
    def get_one_embed(self, path):
        try:
            embed = self.embeds[path]
            #print("Found premade embed for", path)
            return embed
        except Exception as e:
            if self.verbosity == "error": print("Error:", e, "did not find embedding for", path)

            # Read in the image and get the boxes from MTCNN
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = torch.Tensor(im).to(self.device)
            with torch.no_grad():
                boxes, _ = self.cropper.detect(im)
            im = im.cpu().detach().numpy()
            
            # See if there are boxes. If not, or more than one, then we need to return
            try:
                if not boxes:
                    if self.verbosity == "error": print("ERROR 1 in get_one_embed: Boxes are None")
                    if self.verbosity == "error": print("REMOVING", path, "from dataset")
                    self.paths.remove(path)
                    return None
            except:
                None

            if len(boxes) != 1:
                if self.verbosity == "error": print("ERROR 2 in get_one_embed: Boxes are", boxes)
                if self.verbosity == "error": print("REMOVING", path, "from dataset")
                self.paths.remove(path)
                return None

            #print("Boxes are", boxes)
            self.boxes[path] = boxes[0]

            xmin = int(boxes[0][0])
            ymin = int(boxes[0][1])
            xmax = int(boxes[0][2])
            ymax = int(boxes[0][3])

            # Crop the area with the bounding box
            crop = im[ymin:ymax, xmin:xmax, :]

            # Resize to be 112x112
            try:
                crop = torch.Tensor(cv2.resize(crop, (self.cropped_im_size, self.cropped_im_size), interpolation=cv2.INTER_CUBIC))
                crop = torch.permute(crop, (2, 0, 1))
            except Exception as e:
                if self.verbosity == "error": print("ERROR 3 in get_one_embed:", e, "Empty crop on boxes", boxes)
                if self.verbosity == "error": print("REMOVING", path, "from dataset")
                self.paths.remove(path)
                return None

            # Apply the normalization transform
            #print("Before transform, crop has range", torch.min(crop), torch.max(crop))
            #crop /= 255.0
            #crop = self.norm_transform(crop)
            #crop -= 0.5
            #crop = torch.tanh(crop)
            crop = self.preprocess_tanh(crop)
            #print("After transform, crop has range", torch.min(crop), torch.max(crop))

            # Add fake batching to work with mozuma
            crop = crop.repeat((2, 1, 1, 1))
            self.cropped_images[path] = crop

            # Get the embeddings and save them
            crop = crop.to(self.device)
            with torch.no_grad():
                embed = self.extractor(crop)
            crop = crop.cpu().detach()

            #print("Finaly embed has shape", embeds.size())
            self.embeds[path] = embed

            #print("Finished embedding", path)

            return embed
        

    # Get the embeddings of every img in the dataset by first cropping with MTCNN and then running through an extractor
    def get_embeds(self):
        num = 0
        for im, label in tqdm(self.face_loader):
            num += 1
            if num > 150:
                break
            print("Size of self.paths:", sys.getsizeof(self.paths))
            print("Size of embeds:", sys.getsizeof(self.embeds))
            print("Size of crops:", sys.getsizeof(self.cropped_images))
            im = im.squeeze()
            label = str(label[0])
            #print("Looking at label", label, "for im of size", im.size())
            im = torch.Tensor(im).to(self.device)
            with torch.no_grad():
                boxes, _ = self.cropper.detect(im)
            im = im.cpu().detach().numpy()

            try:
                if not boxes:
                    print("ERROR: Boxes are None")
                    self.paths.remove(label)
                    continue
            except:
                None
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
            try:
                crop = torch.Tensor(cv2.resize(crop.detach().cpu().numpy(), (self.cropped_im_size, self.cropped_im_size), interpolation=cv2.INTER_CUBIC))
                crop = torch.permute(crop, (2, 0, 1))
            except:
                print("Empty crop on boxes", boxes)
                self.paths.remove(label)
                continue
            
            # Apply the normalization transform
            #print("Before transform, crop has range", torch.min(crop), torch.max(crop))
            crop /= 255.0
            crop = self.norm_transform(crop)
            #print("After transform, crop has range", torch.min(crop), torch.max(crop))

            # Add fake batching to work with mozuma
            crop = crop.repeat((2, 1, 1, 1))
            self.cropped_images[label] = crop
            
            # Get the embeddings and save them
            with torch.no_grad():
                embeds = self.extractor(crop)
            #print("Finaly embed has shape", embeds.size())
            self.embeds[label] = embeds
        
        print("Finished embedding")

    # Calculate the L2 distance of two vectors
    def L2_dist(self, embed1, embed2):
        return torch.linalg.norm(embed1 - embed2, ord=2)

    # Get the target image according to Fawkes method by finding the maximally different image in a pool
    def get_target(self, path, pool_size=100):
        start = time.perf_counter()
        try:
            this_embed = self.get_one_embed(path)

            # Check if the embedding is None
            try:
                if this_embed == None:
                    if self.verbosity == "error": print("Cannot find target for path", path)
                    return None
            except Exception as e:
                if self.verbosity == "error": print("Error 1 in get_target:", e)
        except Exception as e:
            if self.verbosity == "error": print("Error 2 in get_target:", e)
            return None

        # Get the name of the person we're looking at
        path_name = os.path.basename(path)[:os.path.basename(path).find("_")]
        best_dist = 0
        best_target = None
        
        # Repeat for the number of images in the pool
        while pool_size > 0:

            # Get a random index
            idx = random.randint(0, len(self.paths)-1)
            tgt = self.paths[idx]
            
            # Get the name for this target
            name = os.path.basename(tgt)[:os.path.basename(tgt).find("_")]
            
            # If they're the same class, ignore
            if name == path_name:
                continue
            else:
                tgt_embed = self.get_one_embed(tgt)
                
                try:
                    if tgt_embed == None:
                        if self.verbosity == "error": print("Cannot find target embedding")
                        continue
                except Exception as e:
                    if self.verbosity == "error": print("Error 3 in get_target:", e)
                
                if torch.max(tgt_embed) == 0: # make sure we aren't working towards a null embedding
                    continue
                pool_size -= 1
                sim = self.L2_dist(tgt_embed, this_embed)
                # If this one is even farther away, then save it and continue
                if sim > best_dist:
                    best_dist = sim
                    best_target = tgt
        if self.verbosity == "log": print("Target for", path, "is", best_target)
        end = time.perf_counter()
        if self.verbosity == "log": print(f"Finding target took %0.4f seconds" % (end - start))
        return best_target

    def weak_triplet_loss(self, embed_img, embed_tgt, embed_orig, factor):
        return torch.linalg.norm(embed_img - embed_tgt, ord=2)# + factor * torch.linalg.norm(embed_img - embed_orig, ord=2)
    
    # Cloak a single image by making its feature space embedding more similar to another maximally different target image's embedding. Use 1000 iterations of Adam with a maximum perturbation budget (different from Fawkes)
    def cloak_image(self, img_path, pool_size=100, iters=100, lr=1e-2, pert_budget=8./255., step=1./255.):
        
        # Find the maximally different image
        tgt_path = self.get_target(img_path, pool_size=pool_size)

        if tgt_path == None:
            if self.verbosity == "error": print("No target for image (may cause error on line below)", img_path)
            return np.zeros((self.cropped_im_size, self.cropped_im_size, 3))

        # Retrieve the pre-computed target embedding
        tgt_emb = self.embeds[tgt_path].clone()
        
        # Retrieve the pre-computed image embedding
        cropped = self.cropped_images[img_path]
        cropped = torch.Tensor(cropped).to(self.device)

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.extractor(cropped).detach()
            tgt_emb = tgt_emb.detach()

        # Copy and save the original to measure modification
        orig_cropped = cropped.detach().clone()

        # Record the original dimensions of cropped
        orig_min = torch.min(cropped)
        orig_max = torch.max(cropped)
        #print("orig_min and orig_max are", orig_min, orig_max)

        # Adjust the perturbation budget and step size
        #pert_budget = pert_budget * (orig_max - orig_min)
        #step = step * (orig_max - orig_min)

        # Make sure cropped will receive gradients as a fresh leaf tensor
        cropped = cropped.detach().clone()
        cropped.requires_grad = True

        #cropped = cropped.to(self.device)
        
        # Iterate for iters iterations
        pbar = tqdm(range(iters))
        for i in pbar:
            cropped = cropped.clone().detach().requires_grad_()
            #cropped.requires_grad = True
            out_emb = self.extractor(cropped)
            loss = self.weak_triplet_loss(out_emb, tgt_emb, orig_embed, factor=0.3)
            loss.backward(retain_graph=False)
            pbar.set_postfix({'loss': loss.item()})
            #print("Loss is", loss.item())     
            
            # Get the sign of the gradient
            signed_grad = torch.sign(cropped.grad)
            
            with torch.no_grad():
                # Update cropped
                cropped -= step * signed_grad

                # Clip the perturbation to be within the viable range
                pert = torch.clip(cropped - orig_cropped, min=-pert_budget, max=pert_budget)

                # Add the perturbation back, but make sure we're within [-1, 1]
                cropped = torch.clip(orig_cropped + pert, min=-1.0, max=1.0)

            #cropped = cropped.cpu().detach().clone()
            #print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
            #print("Cropped wants grad:", cropped.requires_grad, cropped.device)
            #print("Tgt:", tgt_emb.device, tgt_emb.requires_grad)
            #print("orig_embed:", orig_embed.device, orig_embed.requires_grad)
       
            del loss, out_emb
            cropped.grad = None
        #cropped = cropped.detach().clone()
            torch.cuda.empty_cache()
        
        cropped = cropped.detach()
        torch.cuda.empty_cache()

        diff = torch.clip(cropped - orig_cropped, min=-pert_budget, max=pert_budget)
        #print("Here max diff is", torch.max(diff))
        cropped = torch.clip(orig_cropped + diff, min=orig_min, max=orig_max)
        cropped = cropped[0]

        # Now that we have the cloaked cropped portion, it's time to fit that back into the image
        # Un-normalize to get back to image space
        #cropped = self.inv_transform(cropped)

        # Return to numpy array
        cropped = cropped.squeeze().detach().cpu().numpy()
        if self.verbosity == "log": print("Pre reverse_tanh range is", np.min(cropped), np.max(cropped))
        #cropped = 255. * cropped
        cropped = self.reverse_tanh(cropped)
        if self.verbosity == "log": print("Post reverse_tanh range is", np.min(cropped), np.max(cropped))
        cropped = np.transpose(cropped, (1, 2, 0)) # set channel last

        # Re-arrange axes to fit with the original image
        boxes = self.boxes[img_path]
        xmin = int(boxes[0])
        ymin = int(boxes[1])
        xmax = int(boxes[2])
        ymax = int(boxes[3])
        cropped = cv2.resize(cropped, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_CUBIC)

        # Retrieve the image and add the patch
        im = self.images[img_path]

        # Convert to int to match with image int
        cropped = cropped.astype(type(im))

        #print("after retrieval im has range", np.min(im), np.max(im))
        try:
            im[ymin:ymax, xmin:xmax, :] = cropped
        except Exception as e:
            if self.verbosity == "error": print("Error", e, "while cropping for boxes", boxes, "cropped shape", cropped.shape, "image shape", im.shape)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        # Save the image for reference later
        # self.cloaked_images[img_path] = im

        return im

    # Cloak all images in the dataset
    def cloak_all(self, num_images=100, save_dir=None):

        # Iterate through each item in the dataset
        num = 0
        for path in self.paths:
            num += 1
            if num > num_images:
                return
            if self.verbosity == "log": print("Cloaking image", path)

            # Cloak the image
            im = self.cloak_image(path, pool_size=self.target_pool_size)
            
            # If we couldn't find a target for the previous image, just skip
            if np.max(im) == 0:
                continue

            #print("Before saving im have shape", im.shape)
            #print("Before saving my range is", np.min(im), np.max(im))
            if save_dir is not None:
                if self.verbosity == "log": print("Saving to", save_dir + os.path.basename(path))
                cv2.imwrite(save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".png", im)

    def apply_defense(self):
        None
