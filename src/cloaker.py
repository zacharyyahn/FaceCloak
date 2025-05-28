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
from math import sqrt, log10
from torchvision import transforms as trans
from cloak_functions import pgd_cloak
from skimage.metrics import structural_similarity

# NOTE: Fawkes used a different model/dataset, and they also did some tanh normalization on the images. See differentiator file in the repo or the paper. They also did not clip images.


class Cloaker():
    def __init__(self, dataset_path, extractor, cropper, batch_size, cropped_im_size, target_pool_size, num_dataset_images, device, verbosity, distance_function, cloak_function, cloak_loss, cloak_function_iters, cloak_function_step, cloak_function_max_pert, cloak_function_lr, loss_func_select, norm_function, reverse_norm_function, percep_loss, percep_loss_weight):
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
        self.verbosity = verbosity
        self.distance_function = distance_function
        self.cloak_function = cloak_function
        self.cloak_loss = cloak_loss
        self.loss_func_select = loss_func_select
        self.norm_function = norm_function
        self.reverse_norm_function = reverse_norm_function

        self.cloak_func_args = {
                "iters":cloak_function_iters,
                "step":cloak_function_step,
                "max_pert":cloak_function_max_pert,
                "lr":cloak_function_lr,
                "dist_func":distance_function,
                "percep_loss":percep_loss,
                "loss_func_select":loss_func_select,
                "percep_loss_weight":percep_loss_weight
        }
    
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
        random.shuffle(self.paths)
        for path in self.paths:
            try:
                im = cv2.imread(path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                self.images[path] = im
            except Exception as e:
                if self.verbosity == "error": print("ERROR reading in images:", e)

    # Get the embeddings of just one image
    def get_one_embed(self, path):
        try:
            embed = self.embeds[path]
            #print("Found premade embed for", path)
            return embed
        except Exception as e:
            if self.verbosity == "log": print("Did not find embedding for", path)

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
            crop = im[ymin:ymax, xmin:xmax, :].copy()

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
            crop = self.norm_function(crop)
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
        
    # Get the maximally similar image in a pool
    def get_closest(self, path, pool_size=100):
        start = time.perf_counter()
        try:
            this_embed = self.get_one_embed(path)
            # Check if the embedding is None
            try:
                if this_embed == None:
                    if self.verbosity == "error": 
                        print("Cannot find target for path", path)
                        return None
            except Exception as e:
                if self.verbosity == "error": print("Error 1 in get_target:", e)
        except Exception as e:
            if self.verbosity == "error": print("Error 2 in get_target:", e)
            return None

        # Get the name of the person we're looking at
        path_name = os.path.basename(path)[:os.path.basename(path).find("_")]
        best_dist = 10000000
        best_target = None

        # Repeat for the number of images in the pool
        while pool_size > 0:

            # Get a random index
            idx = random.randint(0, len(self.paths)-1)
            tgt = self.paths[idx]

            # Get the name for this target
            name = os.path.basename(tgt)[:os.path.basename(tgt).find("_")]

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
            sim = self.distance_function(tgt_embed, this_embed)
            # If this one is even farther away, then save it and continue
            if sim < best_dist:
                best_dist = sim
                best_target = tgt
        if self.verbosity == "log": print("Target for", path, "is", best_target)
        end = time.perf_counter()
        if self.verbosity == "log": print(f"Finding target took %0.4f seconds" % (end - start))
        return best_target

    # Get maximally different image from the pool
    def get_farthest(self, path, pool_size=100):
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
                sim = self.distance_function(tgt_embed, this_embed)
                # If this one is even farther away, then save it and continue
                if sim > best_dist:
                    best_dist = sim
                    best_target = tgt
        if self.verbosity == "log": print("Target for", path, "is", best_target)
        end = time.perf_counter()
        if self.verbosity == "log": print(f"Finding target took %0.4f seconds" % (end - start))
        return best_target
    
    # Insert the cropped portion back into the image, resizing as necessary 
    def reconstruct_image(self, img_path, cropped):
        cropped = cropped[0]
        cropped = cropped.squeeze().detach().cpu().numpy()
        if self.verbosity == "log": print("Pre reverse_tanh range is", np.min(cropped), np.max(cropped))
        #cropped = 255. * cropped
        cropped = self.reverse_norm_function(cropped)
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
        im = im.copy()

        # Convert to int to match with image int
        cropped = np.clip(cropped, 0, 255)
        cropped_clean = np.ascontiguousarray(cropped.astype(np.uint8, copy=False))
        #im = im.astype(np.uint8, copy=False)

        try:
            im[ymin:ymax, xmin:xmax, :] = cropped
        except Exception as e:
            if self.verbosity == "error": print("Error", e, "while cropping for boxes", boxes, "cropped shape", cropped.shape, "image shape", im.shape)
            return None
        return im

    # Cloak a single image by making its feature space embedding more similar to another maximally different target image's embedding. Use 1000 iterations of Adam with a maximum perturbation budget (different from Fawkes)
    def cloak_image(self, img_path, pool_size=100):
        
        # Find the maximally different image
        tgt_path = self.get_farthest(img_path, pool_size=pool_size)

        if tgt_path == None:
            if self.verbosity == "error": print("No target for image (may cause error on line below)", img_path)
            return np.zeros((self.cropped_im_size, self.cropped_im_size, 3))
        
        # If we're doing triplet loss, also calculate closest
        if self.loss_func_select == "triplet":
            closest_path = self.get_closest(img_path, pool_size=pool_size)
            closest_emb = self.embeds[closest_path].clone()
            self.cloak_func_args["closest_emb"] = closest_emb

        # Make these things available to the cloak function
        self.cloak_func_args["reconstruct_func"] = self.reconstruct_image
        self.cloak_func_args["image_path"] = img_path
        self.cloak_func_args["image"] = self.images[img_path]
        
        # Retrieve the pre-computed target embedding
        tgt_emb = self.embeds[tgt_path].clone()
        
        # Retrieve the pre-computed image embedding
        cropped = self.cropped_images[img_path]
        cropped = torch.Tensor(cropped).to(self.device)

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.extractor(cropped).detach()
            tgt_emb = tgt_emb.detach()
        
        # Call our cloaking method to obsure this image
        cropped = self.cloak_function(cropped, tgt_emb, self.extractor, self.cloak_loss, self.device, self.cloak_func_args)

        # Now that we have the cloaked cropped portion, it's time to fit that back into the image

        # Return to numpy array
        im = self.reconstruct_image(img_path, cropped)
        
        try:
            if im == None:
                return None
        except:
            None

        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # Save the image for reference later
        # self.cloaked_images[img_path] = im

        return im

    # Cloak all images in the dataset
    def cloak_all(self, num_images=100, save_dir=None):
        
        total_ssim = 0.0
        total_psnr = 0.0
        total_mse = 0.0
        
        # Iterate through each item in the dataset
        num = 0
        for path in self.paths:
            num += 1
            if num > num_images:
                print(f"Average SSIM %0.4f" % (total_ssim / num))
                print(f"Average PSNR %0.4f" % (total_psnr / num))
                print(f"Average MSE %0.4f" % (total_mse / num))
                return
            if self.verbosity == "log": print("Cloaking image", path)

            # Cloak the image
            orig_im = self.images[path].copy()
            im = self.cloak_image(path, pool_size=self.target_pool_size)
            
            # Check for a none image
            try:
                if im == None:
                    continue
            except:
                None
            
            
            # NOTE: Sometimes these images have different dimensions...?
            # Double check that it's the right size
            im = np.clip(im, a_min=0, a_max=255)

            # Calculate PSNR, SSIM, and MSE
            try:
                total_ssim += structural_similarity(orig_im, im, channel_axis=2)
                mse = np.mean(np.square((orig_im - im)))
                total_mse += mse
                total_psnr += 20.0 * log10(255.0 / sqrt(mse))
            except:
                if self.verbosity == "error": print("ERROR in calculating metrics")
                None
            
            if self.verbosity == "log": print("Right before writing, range is", np.min(im), np.max(im))
            # If we couldn't find a target for the previous image, just skip
            if np.max(im) == 0:
                continue

            #print("Before saving im have shape", im.shape)
            #print("Before saving my range is", np.min(im), np.max(im))
            if save_dir is not None:
                if self.verbosity == "log": print("Saving to", save_dir + os.path.basename(path))
                cv2.imwrite(save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg", im)
        
        print(f"Average SSIM %0.4f" % (total_ssim / num))
        print(f"Average PSNR %0.4f" % (total_psnr / num))
        print(f"Average MSE %0.4f" % (total_mse / num))

    def apply_defense(self):
        None
