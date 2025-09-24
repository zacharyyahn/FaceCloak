import torch
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml
import math
import warnings
from facedataset import FaceDataset
from torch.utils.data import DataLoader
import random
import time
from math import sqrt, log10
from AMTGAN.setup import setup_config, setup_argparser
from AMTGAN.backbone import Inference, PostProcess, get_config
from torchvision import transforms as trans
from cloak_functions import pgd_cloak
from DiffAM.main import dict2namespace
from makeup_functions import diffam_makeup, amtgan_makeup
from DiffAM.models.ddpm.diffusion import DDPM
from skimage.metrics import structural_similarity
#from facechain.facechain.inference_fact import GenPortrait
import torch.nn.functional as F
from arc2face.Arc2Face.arc2face import CLIPTextModelWrapper, project_face_embs
import gc
from utils import pipeline_forward_with_grad
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis
import onnxruntime as ort
from PIL import Image
from insightface_code.recognition.arcface_torch.backbones import get_model

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# NOTE: Fawkes used a different model/dataset, and they also did some tanh normalization on the images. See differentiator file in the repo or the paper. They also did not clip images.

class Cloaker():
    def __init__(self, dataset_path, extractor, cropper, batch_size, cropped_im_size, target_pool_size, num_dataset_images, device, verbosity, distance_function, cloak_function, multi_cloak_function, cloak_loss, multi_cloak_loss, cloak_function_iters, multi_cloak_function_iters, cloak_function_step, cloak_function_max_pert, multi_cloak_function_max_pert, cloak_function_lr, loss_func_select, multi_loss_func_select, norm_function, reverse_norm_function, percep_loss, percep_loss_weight, num_gen_iterations, gen_learning_rate, mode, num_images_to_generate):
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
        self.multi_cloak_function = multi_cloak_function
        self.cloak_loss = cloak_loss
        self.multi_cloak_loss = multi_cloak_loss
        self.loss_func_select = loss_func_select
        self.multi_loss_func_select = multi_loss_func_select
        self.norm_function = norm_function
        self.reverse_norm_function = reverse_norm_function
        self.mode = mode
        self.gen_learning_rate = gen_learning_rate
        self.num_gen_iterations = num_gen_iterations
        self.num_images_to_generate = num_images_to_generate

        if self.mode == "multi_finetune" or self.mode == "minmax" or self.mode == "multi_minmax" or self.mode == "multi":
            #self.gen_portrait = GenPortrait()
            base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
            encoder = CLIPTextModelWrapper.from_pretrained(
                "src/arc2face/encoder", torch_dtype=torch.float16
            ).to(device=self.device, dtype=torch.float16)

            unet = UNet2DConditionModel.from_pretrained(
                "src/arc2face/Arc2Face", torch_dtype=torch.float16
            ).to(device=self.device, dtype=torch.float16)

            pipeline = StableDiffusionPipeline.from_pretrained(
                    base_model,
                    text_encoder=encoder,
                    unet=unet,
                    torch_dtype=torch.float16,
                    safety_checker=None
            )
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.enable_attention_slicing()
            #pipeline.enable_model_cpu_offload()
            self.pipeline = pipeline.to(device=self.device, dtype=torch.float16)

            # All of this is to try to stop annoying ONNX log attacks
            os.environ["OMP_NUM_THREADS"] = "1"        # or a small number like 4
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"  # optional, prevents spin-lock CPU usage
            os.environ["ONNX_DISABLE_CPU_AFFINITY"] = "1"
            os.environ["ORT_DISABLE_GPU"] = "1"
            ort.set_default_logger_severity(4)  # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal

            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # optional, forces single-thread
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            
            # This flag prevents ORT from messing with thread affinity
            so.add_session_config_entry("session.set_denormal_as_zero", "1")
            so.add_session_config_entry("session.disable_prepacking", "1")

            self.app = FaceAnalysis(name='antelopev2', root='src/arc2face', providers=['CUDAExecutionProvider'], sess_options=so)
            self.app.prepare(ctx_id=0, det_size=(640, 640))

            self.backup_arcface_model = get_model("r100", fp16=False)
            self.backup_arcface_model.load_state_dict(torch.load("model_checkpoints/arcface_r100_ms1mv3.pth", map_location=self.device))
            self.backup_arcface_model.eval().to(device=self.device, dtype=torch.float16)

        # This will be passed to the cloaking function to provide additional arguments
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

        # This will be passed to the multi-cloaking function to provide additional arguments
        self.multi_cloak_func_args = {
                "iters":multi_cloak_function_iters,
                "single_iters":cloak_function_iters,
                "step":cloak_function_step,
                "max_pert":multi_cloak_function_max_pert,
                "lr":cloak_function_lr,
                "dist_func":distance_function,
                "percep_loss":percep_loss,
                "loss_func_select":multi_loss_func_select,
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

        # Map of image name to multicloak
        self.multi_map = {}
        
        # Set out models in evaluation mode
        self.extractor = self.extractor.eval()
        self.cropper = self.cropper.eval()

        print("Cloaker is using device", self.device)
        
        # Read in all of the images
        self.paths = self.dataset.paths
        #random.shuffle(self.paths)
        for path in self.paths:
            try:
                im = np.array(Image.open(path).convert('RGB'))
                self.images[path] = im
            except Exception as e:
                if self.verbosity == "error": print("ERROR reading in images:", e)

        print("===============================================")
        print("\nSuccessfully initialized Cloaker.\n")
        print("--- General Parameters ---")
        print(f"+ Dataset: {self.dataset}")
        print(f"+ Batch Size: {self.batch_size}")
        print(f"+ Device: {self.device}")
        print(f"+ Cropped Image Size: {self.cropped_im_size}")
        print(f"+ Target Pool Size: {self.target_pool_size}")
        print(f"+ Verbosity: {self.verbosity}")
        print(f"+ Num Images Loaded: {len(self.images)}")
        print("\n--- General Optimization Parameters ---")
        print(f"+ Norm Function: {self.norm_function}")
        print(f"+ Distance Function: {self.distance_function}")
        print(f"+ Percep Loss: {percep_loss}")
        print(f"+ Percep Loss Weight: {percep_loss_weight}")
        print(f"+ Num Gen Images: {self.num_images_to_generate}")
        print(f"+ Mode: {self.mode}")
        if self.mode == "minmax":
            print("\n--- Min-Max Parameters ---")
            print(f"+ Cloak Function: {self.multi_cloak_function}")
            print(f"+ Cloak Loss: {self.multi_cloak_loss}")
            print(f"+ Loss Function: {self.multi_loss_func_select}")
            print(f"+ Min-Max Iters: {multi_cloak_function_iters}")
            print(f"+ Pert Iters: {cloak_function_iters}")
            print(f"+ Pert LR: {cloak_function_lr}")
            print(f"+ Pert Max: {multi_cloak_function_max_pert}")
            print(f"+ Pert Step: {cloak_function_step}")
            print(f"+ Gen Iters: {self.num_gen_iterations}")
            print(f"+ Gen LR: {self.gen_learning_rate}")
        if self.mode == "multi" or self.mode == "multi_finetune":
            print("\n--- Multi Parameters ---")
            print(f"+ Cloak Function: {self.multi_cloak_function}")
            print(f"+ Cloak Loss: {self.multi_cloak_loss}")
            print(f"+ Loss Function: {self.multi_loss_func_select}")
            print(f"+ Min-Max Iters: {multi_cloak_function_iters}")
            print(f"+ Pert Iters: {multi_cloak_function_iters}")
            print(f"+ Pert LR: {cloak_function_lr}")
            print(f"+ Pert Max: {multi_cloak_function_max_pert}")
            print(f"+ Pert Step: {cloak_function_step}")
        if self.mode == "single":
            print("\n--- Single Parameters ---")
            print(f"+ Cloak Function: {self.cloak_function}")
            print(f"+ Cloak Loss: {self.cloak_loss}")
            print(f"+ Loss Function: {self.loss_func_select}")
            print(f"+ Min-Max Iters: {cloak_function_iters}")
            print(f"+ Pert Iters: {cloak_function_iters}")
            print(f"+ Pert LR: {cloak_function_lr}")
            print(f"+ Pert Max: {cloak_function_max_pert}")
            print(f"+ Pert Step: {cloak_function_step}")
        if self.mode == "multi_finetune":
            print("\n--- Fine-Tune Parameters ---")
            print(f"+ Cloak Function: {self.cloak_function}")
            print(f"+ Cloak Loss: {self.cloak_loss}")
            print(f"+ Loss Function: {self.loss_func_select}")
            print(f"+ Min-Max Iters: {cloak_function_iters}")
            print(f"+ Pert Iters: {cloak_function_iters}")
            print(f"+ Pert LR: {cloak_function_lr}")
            print(f"+ Pert Max: {cloak_function_max_pert}")
            print(f"+ Pert Step: {cloak_function_step}")
        print("===============================================")

    # Get the embeddings of just one image
    def get_one_embed(self, path):
        try:
            embed = self.embeds[path]
            #print("Found premade embed for", path)
            return embed
        except Exception as e:
            if self.verbosity == "log": print("Did not find embedding for", path)

            # Read in the image and get the boxes from MTCNN. If we haven't encountered the image before (i.e. it was generated) then we load it in now
            try:
                im = self.images[path].copy()
            except:
                im = np.array(Image.open(path).convert('RGB'))
                self.images[path] = im
            im = torch.Tensor(im).to(self.device)

            if path not in self.boxes:
                boxes, _ = self.cropper.detect(im)

                # Check for null boxes. If a box is none, replace it with the entire image.
                if type(boxes) == type(None): 
                    boxes = [[0, 0, im.shape[1] - 1, im.shape[0] - 1]]
                
                boxes = boxes[0]

                # we will reshape to a 112x112, so only crop a square area
                side_length = max(boxes[3] - boxes[1], boxes[2] - boxes[0])
                diff = side_length - min(boxes[3] - boxes[1], boxes[2] - boxes[0])
                add_to_side = int(diff / 2)
                if boxes[3] - boxes[1] < boxes[2] - boxes[0]:
                    boxes[3] += add_to_side
                    boxes[1] -= add_to_side
                else:
                    boxes[2] += add_to_side
                    boxes[0] -= add_to_side

                # Make sure that the boxes do not exceed the image size
                for i in range(4):
                    boxes[i] = int(boxes[i]) if boxes[i] >= 0.0 else 0
                boxes[2] = boxes[2] if boxes[2] < im.shape[1] else im.shape[1] - 1
                boxes[3] = boxes[3] if boxes[3] < im.shape[0] else im.shape[0] - 1
                self.boxes[path] = boxes
            else:
                boxes = self.boxes[path]

            # Parse out the values
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])

            crop = im[ymin:ymax, xmin:xmax, :].clone()

            # Resize to be 112x112
            try:
                crop = torch.permute(crop, (2, 0, 1)).unsqueeze(0)
                crop = F.interpolate(crop, size=(self.cropped_im_size, self.cropped_im_size), mode="bilinear", align_corners=False)
            except Exception as e:
                if self.verbosity == "error": print("ERROR 3 in get_one_embed:", e, "Empty crop on boxes", boxes, "tried to crop to size", crop.shape)
                if self.verbosity == "error": print("REMOVING", path, "from dataset")
                self.paths.remove(path)
                return None

            # Apply the normalization transform and save the cropped image
            crop = self.norm_function(crop)

            self.cropped_images[path] = crop.detach().clone()

            # Get the embeddings and save them
            with torch.no_grad():
                embed = self.extractor(crop)
            embed = embed.detach()
            self.embeds[path] = embed

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
        cropped = cropped.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        if self.verbosity == "log": print("Pre reverse norm range is", np.min(cropped), np.max(cropped))
        #cropped = 255. * cropped
        cropped = self.reverse_norm_function(cropped)
        if self.verbosity == "log": print("Post reverse norm range is", np.min(cropped), np.max(cropped))



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
        #im = im.astype(np.uint8, copy=False)

        try:
            im[ymin:ymax, xmin:xmax, :] = cropped
        except Exception as e:
            if self.verbosity == "error": print("Error", e, "while cropping for boxes", boxes, "cropped shape", cropped.shape, "image shape", im.shape)
            return None
        return im

    # Cloak a single image by making its feature space embedding more similar to another maximally different target image's embedding. Use 1000 iterations of Adam with a maximum perturbation budget (different from Fawkes)
    def cloak_image(self, img_path, pool_size=100, force_target=None, force_closest=None, original_path=None):
        
        # Find the maximally different image
        if force_target == None:
            tgt_path = self.get_farthest(img_path, pool_size=pool_size)
        else:
            tgt_path = force_target

        if tgt_path == None:
            if self.verbosity == "error": print("No target for image (may cause error on line below)", img_path)
            return np.zeros((self.cropped_im_size, self.cropped_im_size, 3))
        
        # If we're doing triplet loss, also calculate closest
        if self.loss_func_select == "triplet":
            if force_closest == None:
                closest_path = self.get_closest(img_path, pool_size=pool_size)
            else:
                closest_path = force_closest
            closest_emb = self.get_one_embed(closest_path).clone()
            self.cloak_func_args["closest_emb"] = closest_emb

        # Make these things available to the cloak function
        self.cloak_func_args["reconstruct_func"] = self.reconstruct_image
        self.cloak_func_args["image_path"] = img_path
        self.cloak_func_args["image"] = self.images[img_path]
        
        #If we're doing fine-tuning, use the original image before deepfake universal.
        if original_path:
            self.cloak_func_args["original_image"] = torch.Tensor(self.cropped_images[original_path]).to(self.device)
        else:
            self.cloak_func_args["original_image"] = None
        
        # Retrieve the pre-computed target embedding
        tgt_emb = self.embeds[tgt_path].clone()
        
        # Retrieve the pre-computed image embedding. 
        cropped = self.cropped_images[img_path]
        cropped = torch.Tensor(cropped).to(self.device)

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.extractor(cropped).detach()
            tgt_emb = tgt_emb.detach()
        
        # Call our cloaking method to obsure this image
        cropped = self.cloak_function(cropped, tgt_emb, self.extractor, self.cloak_loss, self.device, self.cloak_func_args)

        # Now that we have the cloaked cropped portion, it's time to fit that back into the image
        im = self.reconstruct_image(img_path, cropped)
        #print("The after we reconstruct, min is", np.min(im), "max is", np.max(im), "mean is", np.mean(im), "std is", np.std(im))
        
        try:
            if im == None:
                return None
        except:
            None

        # Save the image for reference later
        # self.cloaked_images[img_path] = im

        return im

    # Cloak all images in the dataset
    def cloak_all(self, num_images=100, save_dir=None):
        # Make sure that the directory exists before we try to save anything there
        if not os.path.exists(save_dir) and save_dir != None:
            os.makedirs(save_dir)
        
        total_ssim = 0.0
        total_psnr = 0.0
        total_mse = 0.0
        
        # Iterate through each item in the dataset
        num = 0
        pbar = tqdm(total=num_images)
        for path in self.paths:
            num += 1
            pbar.update(1)
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
                    num -= 1
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
                Image.fromarray(im).save(save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg")
        
        print(f"Average SSIM %0.4f" % (total_ssim / num))
        print(f"Average PSNR %0.4f" % (total_psnr / num))
        print(f"Average MSE %0.4f" % (total_mse / num))

    def cloak_minmax(self, img_path, pool_size=100, force_target=None, force_closest=None, original_path=None):
        # Find the maximally different image
        if force_target == None:
            tgt_path = self.get_farthest(img_path, pool_size=pool_size)
        else:
            tgt_path = force_target

        if tgt_path == None:
            if self.verbosity == "error": print("No target for image (may cause error on line below)", img_path)
            return np.zeros((self.cropped_im_size, self.cropped_im_size, 3))
        
        # If we're doing triplet loss, also calculate closest
        if self.loss_func_select == "triplet":
            if force_closest == None:
                closest_path = self.get_closest(img_path, pool_size=pool_size)
            else:
                closest_path = force_closest
            closest_emb = self.get_one_embed(closest_path).clone()
            self.multi_cloak_func_args["closest_emb"] = closest_emb

        # Make these things available to the cloak function
        self.multi_cloak_func_args["reconstruct_func"] = self.reconstruct_image
        self.multi_cloak_func_args["image_path"] = img_path
        self.multi_cloak_func_args["image"] = self.images[img_path]
        self.multi_cloak_func_args["app"] = self.app
        self.multi_cloak_func_args["backup_arcface"] = self.backup_arcface_model
        self.multi_cloak_func_args["pipeline"] = self.pipeline
        self.multi_cloak_func_args["norm_function"] = self.norm_function
        self.multi_cloak_func_args["cropper"] = self.cropper
        self.multi_cloak_func_args["num_gen_iterations"] = self.num_gen_iterations
        self.multi_cloak_func_args["gen_learning_rate"] = self.gen_learning_rate
        self.multi_cloak_func_args["device"] = self.device

        #If we're doing fine-tuning, use the original image before deepfake universal.
        if original_path:
            self.multi_cloak_func_args["original_image"] = torch.Tensor(self.cropped_images[original_path]).to(device=self.device, dtype=torch.float16)
        else:
            self.multi_cloak_func_args["original_image"] = None
        
        # Retrieve the pre-computed target embedding
        tgt_emb = self.embeds[tgt_path].clone()
        
        # Retrieve the pre-computed image embedding. 
        cropped = self.cropped_images[img_path]
        cropped = torch.Tensor(cropped).to(device=self.device, dtype=torch.float32)

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.extractor(cropped).detach()
            tgt_emb = tgt_emb.detach()
        
        mask = self.multi_cloak_function(cropped, tgt_emb, self.extractor, self.cloak_loss, self.device, self.multi_cloak_func_args)

        try:
            if mask == None:
                print("ERROR: None mask")
                return None
        except:
            None

        return mask


    def cloak_all_minmax(self, num_images=100, save_dir=None, gen_save_path=None, do_paths=None):
        # Make sure that the directory exists before we try to save anything there
        if not os.path.exists(save_dir) and save_dir != None:
            os.makedirs(save_dir)
        
        if not os.path.exists(gen_save_path) and gen_save_path != None:
            os.makedirs(gen_save_path)

        warnings.filterwarnings("ignore")
        # Set up basic metrics
        total_ssim_before = 0.0
        total_psnr_before = 0.0
        total_mse_before = 0.0

        # Iterate through each item in the dataset. Accept do_paths in case we want to call this function externally on a limited dataset
        num = 0
        if do_paths == None:
            these_paths = self.paths
        else:
            these_paths = do_paths

        # Iterate for each path we want to cloak
        for path in these_paths:
            num += 1
            start_full = time.perf_counter()
            if num > num_images:
                break
            print("\n\n====== MinMax Cloaking (", str(num), "/", str(num_images), ":", path, ") =======\n\n")
            if self.verbosity == "log": print("Cloaking image", path)

            orig_farthest_path = self.get_farthest(path, self.target_pool_size)
            orig_closest_path = self.get_closest(path, self.target_pool_size)

            # Read in a given image from the real dataset. If we already have a mask for it, apply the mask, otherwise go to the logic that generates the mask.
            name = os.path.basename(path)[:os.path.basename(path).find("_")]
            file_name = os.path.basename(path)[:os.path.basename(path).find(".")]
            
            image = self.images[path]

            # Track the paths of the saved images
            gen_paths = []

            try:
                # Fetch the mask
                mask = self.multi_map[name]
                print("==== Successfully loaded premade mask ====")
            except:
                print("About to call minmax")
                mask = self.cloak_minmax(path, self.target_pool_size, force_target=orig_farthest_path, force_closest=orig_closest_path)
                self.multi_map[name] = mask

            #Apply pert to the cropped face region of the image and make sure it's a valid image range. Save original im for similarity metrics
            orig_im = image.copy()

            boxes = self.boxes[path]
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])

            # Make sure the typing is compatible. Images are usually np.uint8, and the masks begin as floats but are casted to int16 instead of uint8 to prevent integer overflow
            print("Image has range", np.min(image), np.max(image))
            print("Mask has range", np.min(mask), np.max(mask))
            image = image.astype(np.int16)
            mask = mask.astype(np.int16)
            
            # If the face area is (112, 112, 3) then simply add the mask
            if mask.shape == image[ymin:ymax, xmin:xmax, :].shape:
                image[ymin:ymax, xmin:xmax,:] += mask
            else:
                mask = cv2.resize(mask, ((xmax - xmin), (ymax - ymin)), interpolation=cv2.INTER_CUBIC)
            image[ymin:ymax, xmin:xmax, :] += mask

            # Make sure image is still in valid range before we convert back to unsigned ints
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)

            # Calculate PSNR, SSIM, and MSE before fine-tuning
            try:
                total_ssim_before += structural_similarity(orig_im, image, channel_axis=2)
                mse = np.mean(np.square((orig_im - image)))
                total_mse_before += mse
                total_psnr_before += 20.0 * log10(255.0 / sqrt(mse))
            except Exception as e:
                if self.verbosity == "error": print("ERROR in calculating metrics:", e)
                None
            
            if self.verbosity == "log": print("Right before writing, range is", np.min(image), np.max(image))
            
            # If we couldn't find a target for the previous image, just skip
            if np.max(image) == 0:
                if self.verbosity == "error": print("No target found for image, skipping")
                continue

            # # Make sure this is available for use later. We save the image now so that when we convert back to BGR it doesn't affect the second metric test
            total_path = save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg"
            self.boxes[total_path] = boxes.copy()
            self.images[total_path] = image.copy()

            if save_dir is not None:
                if self.verbosity == "log": print("Saving to", save_dir + os.path.basename(path))
                Image.fromarray(image).save(total_path)

        # Print similarity metrics
        print(f"Average SSIM Before Fine-Tuning %0.4f" % (total_ssim_before / (num-1)))
        print(f"Average PSNR Before Fine-Tuning  %0.4f" % (total_psnr_before / (num-1)))
        print(f"Average MSE Before Fine-Tuning %0.4f" % (total_mse_before / (num-1)))
       

    def cloak_multi(self, orig_im, img_paths, pool_size=100, force_target=None, force_closest=None):
        # Run each image in img_paths through get_one_embed so that there is a cropped version of them
        embeds = [self.get_one_embed(im) for im in img_paths]
        cropped_list = [self.cropped_images[im] for im in img_paths]

        # Find the target farthest and closest images to the original in case we need it
        if force_target == None:
            tgt_path = self.get_farthest(orig_im, pool_size=1)
        else:
            tgt_path = force_target
        
        if self.multi_loss_func_select == "triplet":
            if force_closest == None:
                closest_path = self.get_closest(orig_im, pool_size=pool_size)
            else:
                closest_path = force_closest
            closest_emb = self.get_one_embed(closest_path).clone()
            self.multi_cloak_func_args["closest_emb"] = closest_emb

        # Make these things available to the cloak function
        self.multi_cloak_func_args["reconstruct_func"] = self.reconstruct_image
        self.multi_cloak_func_args["image_path"] = orig_im
        self.multi_cloak_func_args["image"] = self.images[orig_im]
        self.multi_cloak_func_args["tgt_emb"] = self.get_one_embed(tgt_path).clone()
        
        # Retrieve the pre-computed target embedding
        if self.multi_loss_func_select == "untarget":
            tgt_emb = self.embeds[orig_im].clone() # Untargeted loss
        else:
            tgt_emb = self.multi_cloak_func_args["tgt_emb"] # Everything else
        
        # Retrieve the pre-computed image embedding
        cropped = self.cropped_images[orig_im]
        cropped = torch.Tensor(cropped).to(self.device)

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.extractor(cropped).detach()
            tgt_emb = tgt_emb.detach()
        
        # Call the cloak function, passing the list of cropped images as one of the inputs. Returns a mask
        mask = self.multi_cloak_function(cropped_list, tgt_emb, self.extractor, self.multi_cloak_loss, self.device, self.multi_cloak_func_args)
        mask *= 255.0
        return mask

    # NOTE: GENERATED IMAGES ARE IN CV2 STYLE SO MAY NEED TO REARRANGE MASK AXES
    # TODO: Need to account for different faces having different poses, etc. Maybe just apply to full 512x512 face instead of MTCNN, or maybe have some sort of anchor system
    # TODO: Need to figure out how to make images look more like their targets
    # TODO: Need to play around more with generation settings and read the paper to get more familiar with the system.
    # TODO: Check about if AMT-GAN does its own cropping, and check to make sure the channels were done correctly.
    # TODO: Make sure that the detection models are properly configured. The benign accuracies are still suspiciously low. Can check their papers to see if they have stats.

    def cloak_all_multi(self, num_images=100, save_dir=None, gen_save_path="/", do_paths=None):
        # Make sure that the directory exists before we try to save anything there
        if not os.path.exists(save_dir) and save_dir != None:
            os.makedirs(save_dir)
        
        if not os.path.exists(gen_save_path) and gen_save_path != None:
            os.makedirs(gen_save_path)

        warnings.filterwarnings("ignore")
        # Set up basic metrics
        total_ssim_before, total_ssim_after = 0.0, 0.0
        total_psnr_before, total_psnr_after = 0.0, 0.0
        total_mse_before, total_mse_after = 0.0, 0.0
        
        # Set up the generation pipeline
        use_face_swap = False
        multiplier_style = 0.25
        base_model_idx = 0

        # Iterate through each item in the dataset. Accept do_paths in case we want to call this function externally on a limited dataset
        num = 0
        if do_paths == None:
            these_paths = self.paths
        else:
            these_paths = do_paths

        generated_image_paths = []

        # Now call on every path we want to 
        for path in these_paths:
            num += 1
            start_full = time.perf_counter()
            if num > num_images:
                break
            print("\n\n====== Multi-Cloaking (", str(num), "/", str(num_images), ":", path, ") =======\n\n")
            if self.verbosity == "log": print("Cloaking image", path)

            orig_farthest_path = self.get_farthest(path, self.target_pool_size)
            orig_closest_path = self.get_closest(path, self.target_pool_size)

            # Read in a given image from the real dataset. If we already have a mask for it, apply the mask, otherwise go to the logic that generates the mask.
            name = os.path.basename(path)[:os.path.basename(path).find("_")]
            file_name = os.path.basename(path)[:os.path.basename(path).find(".")]
            
            image = self.images[path]

            # Track the paths of the saved images
            gen_paths = []

            try:
                # Fetch the mask
                mask = self.multi_map[name]
                print("==== Successfully loaded premade mask ====")
            except:
                # Read in image and embed it with arcface
                faces = self.app.get(image)
                if faces != []:
                    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
                    id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
                else:
                    if self.verbosity == "error":
                        print(f"Error on app face recog for image {path}, switching to second arcface.")
                    crop = self.norm_function(image)
                    crop = torch.tensor(crop.transpose((2, 0, 1)), dtype=torch.float16).unsqueeze(0).to(self.device)
                    crop = F.interpolate(crop, size=(self.cropped_im_size, self.cropped_im_size), mode="bilinear", align_corners=False)
                    id_emb = torch.tensor(self.backup_arcface_model(crop), dtype=torch.float16).detach().clone()
                id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True) 
                
                # Project the image embedding into the CLIP prompt embedding
                id_emb = project_face_embs(self.pipeline, id_emb).detach().clone()   # ensure no history
                id_emb.requires_grad_(True)
                # Generate num_generate_images fake images of the person and save them
                outputs = []

                start = time.perf_counter()
                for _ in range(self.num_images_to_generate):
                    images, pil_images = pipeline_forward_with_grad(
                        self.pipeline,
                        prompt_embeds=id_emb,
                        num_inference_steps=25,
                        guidance_scale=3.0,
                        height=512,
                        width=512,
                    )
                    # need to cast pil image to proper range
                    del images
                    outputs.append((255 * pil_images).astype(np.uint8))
                
                end = time.perf_counter()
                if self.verbosity == "error": print(f"Generating {self.num_images_to_generate} images took {end-start:4f} seconds")
                #outputs.append(output[0]) #comment out this line if not doing poses

                # Save a 512x512 version of the image for use in alternating with deepfakes
                # big_image = cv2.resize(image, (512,512), cv2.INTER_CUBIC)
                # big_image = cv2.cvtColor(big_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(gen_save_path + "/" + file_name + "_big.png", big_image)
                # gen_paths.append(gen_save_path + "/" + file_name + "_big.png") #interleave with original image but upscaled
                
                # Save the paths and the images. We need to save the paths in a list so that the code can use them elsewhere
                start = time.perf_counter()
                for i, im in enumerate(outputs):
                    gen_paths.append(gen_save_path + "/" + file_name + "_" + str(i) + ".png")
                    if gen_save_path != "/":
                        Image.fromarray(im).save(gen_save_path + "/" + file_name + "_" + str(i) + ".png")
                end = time.perf_counter()
                if self.verbosity == "error": print(f"Saving images took {end-start:4f} seconds")
                
                # Pass all of the fake image paths to the cloak_multi() function, as well as the real image for finding closest and farthest images, getting back the mask
                start = time.perf_counter()
                mask = self.cloak_multi(orig_im = path, img_paths=gen_paths, pool_size=self.target_pool_size, force_target=orig_farthest_path, force_closest=orig_closest_path)
                end = time.perf_counter()
                if self.verbosity == "error": print(f"Multi-cloaking images took {end-start:4f} seconds")
                self.multi_map[name] = mask.copy()

                # clean up by deleting generated images to save space
                
            #Apply pert to the cropped face region of the image and make sure it's a valid image range. Save original im for similarity metrics
            orig_im = image.copy()

            boxes = self.boxes[path]
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])

            # Make sure the typing is compatible. Images are usually np.uint8, and the masks begin as floats but are casted to int16 instead of uint8 to prevent integer overflow
            image = image.astype(np.int16)
            mask = mask.astype(np.int16)
            
            # If the face area is (112, 112, 3) then simply add the mask
            if mask.shape == image[ymin:ymax, xmin:xmax, :].shape:
                image[ymin:ymax, xmin:xmax,:] += mask
            else:
                mask = cv2.resize(mask, ((xmax - xmin), (ymax - ymin)), interpolation=cv2.INTER_CUBIC)
            image[ymin:ymax, xmin:xmax, :] += mask

            # Make sure image is still in valid range before we convert back to unsigned ints
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)

            # Calculate PSNR, SSIM, and MSE before fine-tuning
            image_copy = image.copy()
            orig_image_copy = orig_im.copy()
            try:
                total_ssim_before += structural_similarity(orig_im, image, channel_axis=2)
                mse = np.mean(np.square((orig_im - image)))
                total_mse_before += mse
                total_psnr_before += 20.0 * log10(255.0 / sqrt(mse))
            except Exception as e:
                if self.verbosity == "error": print("ERROR in calculating metrics:", e)
                None
            
            if self.verbosity == "log": print("Right before writing, range is", np.min(image), np.max(image))
            
            # If we couldn't find a target for the previous image, just skip
            if np.max(image) == 0:
                if self.verbosity == "error": print("No target found for image, skipping")
                continue

            # # Make sure this is available for use later. We save the image now so that when we convert back to BGR it doesn't affect the second metric test
            total_path = save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg"
            self.boxes[total_path] = boxes.copy()
            self.images[total_path] = image.copy()

            if save_dir is not None:
                if self.verbosity == "log": print("Saving to", save_dir + os.path.basename(path))
                Image.fromarray(image).save(total_path)

            if self.mode == "multi_finetune":
                # Try fine-tuning it a bit, but still targeting the original image
                _ = self.get_one_embed(total_path) # make sure we have access to the cropped image and the embedding
                #print("Just called get_one_embed on total_path", total_path)
                image = self.cloak_image(total_path, self.target_pool_size, force_target=orig_farthest_path, force_closest=orig_closest_path)#, original_path=path)

                # Save the fine-tuned image, overwriting the image from before
                if save_dir is not None:
                    if self.verbosity == "log": print("Saving fine-tuned to", save_dir + os.path.basename(path))
                    total_path = save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg"
                    Image.fromarray(image).save(total_path)

            # Calculate PSNR, SSIM, and MSE again now that we've fine-tuned
            # print("The second time we calculate, min is", np.min(image), "max is", np.max(image), "mean is", np.mean(image), "std is", np.std(image))
            # print("mean image difference is", np.mean(image - image_copy), np.max(image - image_copy), np.min(image - image_copy))
            # print("mean orig image difference is", np.mean(orig_im - orig_image_copy), np.max(orig_im - orig_image_copy), np.min(orig_im - orig_image_copy))
            try:
                total_ssim_after += structural_similarity(orig_im, image, channel_axis=2)
                mse = np.mean(np.square((orig_im - image)))
                total_mse_after += mse
                total_psnr_after += 20.0 * log10(255.0 / sqrt(mse))
            except Exception as e:
                if self.verbosity == "error": print("ERROR in calculating metrics:", e)
                None

            end_full = time.perf_counter()
            print("====== Full Iteration Took:", end_full-start_full, "seconds. =======")

        # Print similarity metrics
        print(f"Average SSIM Before Fine-Tuning %0.4f" % (total_ssim_before / (num-1)))
        print(f"Average PSNR Before Fine-Tuning  %0.4f" % (total_psnr_before / (num-1)))
        print(f"Average MSE Before Fine-Tuning %0.4f" % (total_mse_before / (num-1)))
        print(f"Average SSIM After Fine-Tuning %0.4f" % (total_ssim_after / (num-1)))
        print(f"Average PSNR After Fine-Tuning %0.4f" % (total_psnr_after / (num-1)))
        print(f"Average MSE After Fine-Tuning %0.4f" % (total_mse_after / (num-1)))

    # Apply makeup to all of the images in the dataset
    def makeup_all(self, num_images=100, save_dir=None, makeup_mode="DiffAM"):
        # Make sure that the directory exists before we try to save anything there
        if not os.path.exists(save_dir) and save_dir != None:
            os.makedirs(save_dir)
        
        total_ssim = 0.0
        total_psnr = 0.0
        total_mse = 0.0

        # If we want to use the DiffAM method, prepare to call that code
        if makeup_mode.lower() == "diffam":
            with open("src/DiffAM/configs/MT.yml", 'r') as f:
                config = yaml.safe_load(f)
            config = dict2namespace(config)
            model_path = "src/DiffAM/checkpoint/test_MT_CelebA_HQ_XMY-060_t60_ninv20_ngen6_dis1_dir0.3_lr8e-06_XMY-060-3.pth"
            model = DDPM(config)
            ckpt = torch.load(model_path)
            learn_sigma = False
            model.load_state_dict(ckpt)
            model.to(self.device, dtype=torch.float32)
            model = torch.nn.DataParallel(model)
            model.eval()
            print(f"{model_path} is loaded for makeup transfer.")

        if makeup_mode.lower() == "amt-gan":
            # parser = setup_argparser()
            # args = parser.parse_args()
            config = get_config()
            config.source_dir = "/"
            config.reference_dir = "/"
            config.save_path = save_dir
            config.device = 0
            config.model_path = "src/AMTGAN/checkpoints/G.pth"
            inference = Inference(config, self.device, "src/AMTGAN/checkpoints/G.pth")
            postprocess = PostProcess(config)

        # Iterate through each item in the dataset
        num = 0
        prog_bar = tqdm(range(num_images))
        for path in self.paths:
            num += 1
            prog_bar.update(1)
            prog_bar.refresh()
            if num > num_images:
                print(f"Average SSIM %0.4f" % (total_ssim / num))
                print(f"Average PSNR %0.4f" % (total_psnr / num))
                print(f"Average MSE %0.4f" % (total_mse / num))
                return
            if self.verbosity == "log": print("Applying makeup to image", path)

            # Cloak the image
            orig_im = self.images[path].copy()
            if makeup_mode.lower() == "diffam":
                im = diffam_makeup(path=path, model=model, config=config, device=self.device)
            if makeup_mode.lower() == "amt-gan":
                im = amtgan_makeup(path=path, inference=inference, postprocess=postprocess)
            # Check for a none image
            try:
                if im == None:
                    continue
            except:
                None
            
            # NOTE: Sometimes these images have different dimensions...?
            # Double check that it's the right size
            im = np.clip(im, a_min=0, a_max=255)
            orig_im = cv2.resize(orig_im, (im.shape[0], im.shape[1]), interpolation=cv2.INTER_CUBIC)

            # Calculate PSNR, SSIM, and MSE
            try:
                total_ssim += structural_similarity(orig_im, im, channel_axis=2)
                mse = np.mean(np.square((orig_im - im)))
                total_mse += mse
                total_psnr += 20.0 * log10(255.0 / sqrt(mse))
            except Exception as e:
                if self.verbosity == "error": print("ERROR in calculating metrics:", e)
                None
            
            if self.verbosity == "log": print("Right before writing, range is", np.min(im), np.max(im))
            # If we couldn't find a target for the previous image, just skip
            if np.max(im) == 0:
                continue

            #print("Before saving im have shape", im.shape)
            #print("Before saving my range is", np.min(im), np.max(im))
            if save_dir is not None:
                if self.verbosity == "log": print("Saving to", save_dir + os.path.basename(path))
                Image.fromarray(im).save(save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg")
        
        print(f"Average SSIM %0.4f" % (total_ssim / num))
        print(f"Average PSNR %0.4f" % (total_psnr / num))
        print(f"Average MSE %0.4f" % (total_mse / num))

    def apply_defense(self):
        None
