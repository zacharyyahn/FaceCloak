"""
This file defines the cloaker class, which contains the main logic for cloaking images.
"""


import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
import warnings
from facedataset import FaceDataset
from torch.utils.data import DataLoader
import time
from math import sqrt, log10
from AMTGAN.backbone import Inference, PostProcess, get_config
from torchvision import transforms as trans
from skimage.metrics import structural_similarity
from AMTGAN.assets.models.facenet import InceptionResnetV1
from AMTGAN.assets.models.ir152 import IR_152, IR_SE_50
from AMTGAN.assets.models.irse import MobileFaceNet
import torch.nn.functional as F
from arc2face.Arc2Face.arc2face import CLIPTextModelWrapper, project_face_embs
from utils import pipeline_forward_with_grad
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis
import onnxruntime as ort
from PIL import Image
from insightface_code.recognition.arcface_torch.backbones import get_model
from advcloak.model_irse import IR_50
from advcloak.ser50_webface_soft import SER50WebfaceSoft
from advcloak.m1_webface_soft import M1WebfaceSoft
from advcloak.i1_webface_soft import I1WebfaceSoft

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class Cloaker():
    def __init__(self, probe_dataset_path, gallery_dataset_path, extractor, cropper, batch_size, cropped_im_size, device, verbosity, distance_function, cloak_function, multi_cloak_function, do_stickers, do_highpass, cloak_loss, multi_cloak_loss, cloak_function_iters, multi_cloak_function_iters, cloak_function_step, cloak_function_max_pert, multi_cloak_function_max_pert, cloak_function_lr, loss_func_select, multi_loss_func_select, norm_function, reverse_norm_function, percep_loss, percep_loss_weight, num_gen_iterations, gen_learning_rate, n_to_eval, mode, num_images_to_generate, use_real):
        self.probe_dataset_path = probe_dataset_path
        self.probe_dataset = FaceDataset(probe_dataset_path, num_images=len(os.listdir(probe_dataset_path)))
        self.gallery_dataset = FaceDataset(gallery_dataset_path, num_images=len(os.listdir(gallery_dataset_path)))
        self.extractor = extractor
        self.embeds = {}
        self.cloaked_ims = []
        self.cropper = cropper
        self.batch_size = batch_size
        self.device = device
        self.cropped_im_size = cropped_im_size
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
        self.n_to_eval = n_to_eval
        self.use_real = use_real

        if self.mode == "multi_finetune" or self.mode == "multi":
            
            # Build the synthetic image generation model from arc2face
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
            self.pipeline = pipeline.to(device=self.device, dtype=torch.float16)

            # All of this is to try to stop annoying ONNX log attacks
            os.environ["OMP_NUM_THREADS"] = "1"        # or a small number like 4
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"  # optional, prevents spin-lock CPU usage
            os.environ["ONNX_DISABLE_CPU_AFFINITY"] = "1"
            os.environ["ORT_DISABLE_GPU"] = "1"
            ort.set_default_logger_severity(4)  # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal

            # Declare a face extractor that works with arc2face
            self.app = FaceAnalysis(name='antelopev2', root='src/arc2face', providers=['CUDAExecutionProvider']) #sess_options=so)
            self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5) # was 640, 640

            # Use a different model in case the FaceAnalysis one fails
            self.backup_arcface_model = get_model("r100", fp16=False)
            self.backup_arcface_model.load_state_dict(torch.load("model_checkpoints/arcface_r100_ms1mv3.pth", map_location=self.device))
            self.backup_arcface_model.eval().to(device=self.device, dtype=torch.float16)

        # Declare all the models we'll be using
        arcface = IR_50((112, 112))
        cosface = IR_50((112, 112))
        softmax = IR_50((112, 112))
        ser = SER50WebfaceSoft("model_checkpoints/ser50_webface_soft.npy")
        i50 = I1WebfaceSoft("model_checkpoints/webface_soft.npy")
        m50 = M1WebfaceSoft("model_checkpoints/m1_webface_soft.npy")
        facenet = InceptionResnetV1(pretrained="vggface2")
        ir152 = IR_152((112, 112))
        irse50 = IR_SE_50((112, 112))
        mf = MobileFaceNet(512)
        
        # Load in the state dicts
        arcface.load_state_dict(torch.load("model_checkpoints/arcface.pth", map_location=self.device))
        cosface.load_state_dict(torch.load("model_checkpoints/cosface.pth", map_location=self.device))
        softmax.load_state_dict(torch.load("model_checkpoints/softmax.pth", map_location=self.device))
        facenet.load_state_dict(torch.load("src/AMTGAN/assets/models/facenet.pth"))
        ir152.load_state_dict(torch.load("src/AMTGAN/assets/models/ir152.pth"))
        irse50.load_state_dict(torch.load("src/AMTGAN/assets/models/irse50.pth"))
        mf.load_state_dict(torch.load("src/AMTGAN/assets/models/mobile_face.pth"))

        arcface.eval().to(self.device)
        cosface.eval().to(self.device)
        softmax.eval().to(self.device)
        ser.eval().to(self.device)
        i50.eval().to(self.device)
        m50.eval().to(self.device)
        facenet.eval().to(self.device)
        ir152.eval().to(self.device)
        irse50.eval().to(self.device)
        mf.eval().to(self.device)

        # The full list of models. Any of these can also be used as the extractor
        self.eval_models = {
            "ArcFace":arcface,
            "CosFace":cosface,
            "Softmax":softmax,
            "MobileNet":m50,
            "SER50":ser,
            "IncRes50":i50, 
            "IncRes152":ir152,
            "IncResSE50":irse50,
            "Facenet": facenet,
            "MobileFace": mf,
        }
        
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
            "percep_loss_weight":percep_loss_weight,
            "do_highpass":do_highpass,
            "do_stickers":do_stickers,
        }
    
        # Map of each path to a face box
        self.boxes = {}
        
        # Map of each path to a cropped face box
        self.cropped_images = {}
        self.cropped_images_112 = {}

        # Map of each path to the original image, necessary for reconstructing the face
        self.images = {}

        # Map of each cloaked image
        self.cloaked_images = {}

        # Map of image name to multicloak
        self.multi_map = {}

        # Map of stickers for each class
        self.sticker_map = {}

        # Map of highpass filters for each class
        self.highpass_map = {}

        # Map of which images have been preprocessed
        self.preprocessed_ims = {}

        # Embeds for each model for each path
        self.embeds = {model_name: {} for model_name in list(self.eval_models.keys())}
        
        # Set out models in evaluation mode
        self.cropper = self.cropper.eval()

        print("Cloaker is using device", self.device)
        
        # Read in all of the images
        self.probe_paths = self.probe_dataset.paths
        self.gallery_paths = self.gallery_dataset.paths

        print("===============================================")
        print("\nSuccessfully initialized Cloaker.\n")
        print("--- General Parameters ---")
        print(f"+ Dataset: {self.probe_dataset}")
        print(f"+ Batch Size: {self.batch_size}")
        print(f"+ Device: {self.device}")
        print(f"+ Verbosity: {self.verbosity}")
        print(f"+ Cropped Image Size: {self.cropped_im_size}")
        print(f"+ Probe Set Size: {len(self.probe_paths)}")
        print(f"+ Gallery Set Size: {len(self.gallery_paths)}")
        print(f"+ Num Images Loaded: {len(self.images)}")
        print("\n--- General Optimization Parameters ---")
        print(f"+ Norm Function: {self.norm_function}")
        print(f"+ Distance Function: {self.distance_function}")
        print(f"+ Percep Loss: {percep_loss}")
        print(f"+ Percep Loss Weight: {percep_loss_weight}")
        print(f"+ Num Gen Images: {self.num_images_to_generate}")
        print(f"+ Mode: {self.mode}")
        print(f"+ Top-n Eval: {self.n_to_eval}")
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
            print(f"+ Do Stickers: {do_stickers}")
            print(f"+ Do Highpass: {do_highpass}")
            print(f"+ Cloak Loss: {self.multi_cloak_loss}")
            print(f"+ Loss Function: {self.multi_loss_func_select}")
            print(f"+ Min-Max Iters: {multi_cloak_function_iters}")
            print(f"+ Pert Iters: {multi_cloak_function_iters}")
            print(f"+ Pert LR: {cloak_function_lr}")
            print(f"+ Pert Max: {multi_cloak_function_max_pert}")
            print(f"+ Pert Step: {cloak_function_step}")
        if self.mode == "perturb":
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

        # First read in and convert the probe paths, embedding now so we have them all cached
        print("\nReading and embedding probe paths...")
        pbar = tqdm(total=len(self.probe_paths))
        for path in self.probe_paths:
            try:
                im = np.array(Image.open(path).convert('RGB'))
                self.images[path] = im
                for model in self.eval_models:
                    self.get_one_embed(path, model)
                pbar.update(1)
            except Exception as e:
                if self.verbosity == "error": print("ERROR reading in images:", e)
        
        # Now read in and convert the gallery paths
        print("\nReading and embedding gallery paths...")
        pbar = tqdm(total=len(self.gallery_paths))
        gallery_embs = {model: [] for model in self.eval_models}
        for path in self.gallery_paths:
            try:
                im = np.array(Image.open(path).convert('RGB'))
                self.images[path] = im
                for model in self.eval_models:
                    emb = self.get_one_embed(path, model).squeeze().cuda()
                    gallery_embs[model].append(emb)
                pbar.update(1)
            except Exception as e:
                if self.verbosity == "error": print("ERROR reading in images:", e)

        self.stacked_gallery_embeds = {model: torch.stack(gallery_embs[model]) for model in self.eval_models}
    
    # Get the embeddings of just one image
    @torch.no_grad()
    def get_one_embed(self, path, model):
        try:
            return self.embeds[model][path]
        except KeyError:
            if self.verbosity == "log":
                print("Did not find embedding for", path)

        # Preprocess image so that face crop = 112x112
        im_resized, boxes_resized = self.preprocess_image_for_user(path)
        
        # Parse box coordinates
        xmin = int(boxes_resized[0])
        ymin = int(boxes_resized[1])
        xmax = int(boxes_resized[2])
        ymax = int(boxes_resized[3])

        # Crop and normalize
        crop = torch.tensor(im_resized[ymin:ymax, xmin:xmax, :]).clone()
        crop = self.norm_function(crop)

        # Resize to 112×112 just in case rounding changed 1–2 px
        crop = F.interpolate(
            crop.permute(2, 0, 1).unsqueeze(0), 
            size=(self.cropped_im_size, self.cropped_im_size),
            mode="bilinear", 
            align_corners=False
        )
        
        self.cropped_images[path] = crop.detach().clone()
        crop = crop.to(self.device)

        # Get embedding
        with torch.no_grad():
            embed = self.eval_models[model](crop)
            embed = embed / torch.norm(embed, p=2, dim=1, keepdim=True)

        # Cache and return
        self.embeds[model][path] = embed.cpu()
        return embed
  
    # Get the maximally similar image in a pool
    @torch.no_grad()
    def get_closest(self, path, model, no_self=False, n=1):
        start = time.perf_counter()

        # Get the embedding of this image
        this_embed = self.get_one_embed(path, model)

        # Get the embeddings of all of the gallery images
        gallery_embeds = self.stacked_gallery_embeds[model].clone()
        this_embed = this_embed.clone()

        # Compare with each gallery embedding
        distances = self.distance_function(gallery_embeds, this_embed.cuda())
        sorted_dists, sorted_indices = torch.sort(distances)

        # Keep only the top n
        top_n_indices = sorted_indices[:n+1]
        top_n_dists = sorted_dists[:n+1]

        best_target = []

        i = -1
        num_found = 0
        while num_found < n:
            i += 1
            if no_self and os.path.basename(self.gallery_paths[top_n_indices[i]]) == os.path.basename(path):
                continue
            best_target.append((self.gallery_paths[top_n_indices[i]], top_n_dists[i].item()))
            num_found += 1

        print("Closest is", best_target, " and has distance:", sorted_dists[0])


        if self.verbosity == "log": print("Target for", path, "is", best_target)
        end = time.perf_counter()
        if self.verbosity == "log": print(f"Finding target took %0.4f seconds" % (end - start))
        
        return best_target

    # Get maximally different image from the pool
    @torch.no_grad()
    def get_farthest(self, path, model, n=1):
        start = time.perf_counter()
        try:
            this_embed = self.get_one_embed(path, model)

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

        # Get all of the embeddings from the gallery
        gallery_embeds = self.stacked_gallery_embeds[model]
        this_embed = this_embed.to(self.device)

        # Compute distances between this embedding and each gallery embedding
        distances = self.distance_function(gallery_embeds, this_embed)
        sorted_dists, sorted_indices = torch.sort(distances, descending=True)

        # Keep the farthest one
        top_index = sorted_indices[0]
        top_dist = sorted_dists[0]

        best_target = self.gallery_paths[top_index]

        print("Farthest is", best_target, " and has distance:", sorted_dists[0])

        if self.verbosity == "log": print("Target for", path, "is", best_target)
        end = time.perf_counter()
        if self.verbosity == "log": print(f"Finding target took %0.4f seconds" % (end - start))
        return best_target
    
    # Insert the cropped portion back into the image, resizing as necessary 
    @torch.no_grad()
    def reconstruct_image(self, img_path, cropped):
        cropped = cropped[0]
        cropped = cropped.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        if self.verbosity == "log": print("Pre reverse norm range is", np.min(cropped), np.max(cropped))
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

        try:
            im[ymin:ymax, xmin:xmax, :] = cropped
        except Exception as e:
            if self.verbosity == "error": print("Error", e, "while cropping for boxes", boxes, "cropped shape", cropped.shape, "image shape", im.shape)
            return None
        return im

    # Cloak a single image by making its feature space embedding more similar to another maximally different target image's embedding. Use 1000 iterations of Adam with a maximum perturbation budget (different from Fawkes)
    def cloak_image(self, img_path, force_target=None, force_closest=None, original_path=None):
        
        # Find the maximally different image
        if force_target == None:
            tgt_path = self.get_farthest(img_path, self.extractor)
        else:
            tgt_path = force_target

        if tgt_path == None:
            if self.verbosity == "error": print("No target for image (may cause error on line below)", img_path)
            return np.zeros((self.cropped_im_size, self.cropped_im_size, 3))
        
        # If we're doing triplet loss, also calculate closest
        if self.loss_func_select == "triplet":
            if force_closest == None:
                closest_path = self.get_closest(img_path, self.extractor)[0][0]
            else:
                closest_path = force_closest
            closest_emb = self.get_one_embed(closest_path, self.extractor).clone()
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
        tgt_emb = self.embeds[self.extractor][tgt_path].clone()
        
        # Retrieve the pre-computed image embedding. 
        cropped = self.cropped_images[img_path]
        cropped = torch.Tensor(cropped).to(self.device)

        # Make sure these aren't contributing to the computational graph
        with torch.no_grad():
            orig_embed = self.get_one_embed(img_path, self.extractor)
            tgt_emb = tgt_emb.detach()
        
        # Call our cloaking method to obsure this image
        cropped = self.cloak_function(cropped, tgt_emb, self.eval_models[self.extractor], self.cloak_loss, self.device, self.cloak_func_args)
        
        # Now that we have the cloaked cropped portion, it's time to fit that back into the image
        im = self.reconstruct_image(img_path, cropped)
        
        try:
            if im == None:
                return None
        except:
            None

        return im

    # Cloak all images in the dataset
    def cloak_all(self, save_dir=None, do_paths=None):
        
        # Make sure that the directory exists before we try to save anything there
        if not os.path.exists(save_dir) and save_dir != None:
            os.makedirs(save_dir)
        
        total_ssim = 0.0
        total_psnr = 0.0
        total_mse = 0.0
        total_correct_top1 = {model: 0 for model in list(self.eval_models.keys())}
        total_correct_top5 = {model: 0 for model in list(self.eval_models.keys())}

        if do_paths == None:
            paths = self.probe_paths
        else:
            paths = do_paths
        
        # Iterate through each item in the dataset
        num = 0
        pbar = tqdm(total=len(paths))
        for path in paths:
            num += 1
            pbar.update(1)
   
            if self.verbosity == "log": print("Cloaking image", path)

            # Cloak the image
            orig_im = self.images[path].copy()
            im = self.cloak_image(path)

            print("Got back im of shape", im.shape, "from cloaking")
            
            # Check for a none image
            try:
                if im == None:
                    num -= 1
                    continue
            except:
                None
            
            # Double check that it's the right size
            im = np.clip(im, a_min=0, a_max=255)

            # Calculate PSNR, SSIM, and MSE
            try:
                print("Orig im size:", orig_im.shape)
                print("New im size:", im.shape)
                total_ssim += structural_similarity(orig_im, im, channel_axis=2, data_range=255.0)
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

            total_path = save_dir + "/" + os.path.basename(path)[:os.path.basename(path).find(".")] + ".jpg"
            if save_dir is not None:
                if self.verbosity == "log": print("Saving to", save_dir + os.path.basename(path))
                Image.fromarray(im).save(total_path)

            print("About to call eval_on_all_models")
            self.images[total_path] = im
            top1_results = self.eval_on_all_models(total_path, n=1)
            for model in results:
                total_correct_top1[model] += results[model]
            top5_results = self.eval_on_all_models(total_path, n=5)
            for model in results:
                total_correct_top5[model] += results[model]
        
        print(f"Average SSIM %0.4f" % (total_ssim / num))
        print(f"Average PSNR %0.4f" % (total_psnr / num))
        print(f"Average MSE %0.4f" % (total_mse / num))
        for model in total_correct:
            print(f"{model} Accuracy: {total_correct[model] / num:.4f}")

    # Cloak a single image, generating an identity-specific cloak for that person
    def cloak_multi(self, orig_im, img_paths, force_target=None, force_closest=None):
        
        # Run each image in img_paths through get_one_embed so that there is a cropped version of them
        embeds = [self.get_one_embed(im, self.extractor) for im in img_paths]
        cropped_list = [self.cropped_images[im] for im in img_paths]

        # Find the target farthest and closest images to the original in case we need it
        if force_target == None:
            tgt_path = self.get_farthest(orig_im, self.extractor)
        else:
            tgt_path = force_target
        self.multi_cloak_func_args["tgt_emb"] = self.get_one_embed(tgt_path, self.extractor).clone()
        
        if self.multi_loss_func_select == "triplet":
            if force_closest == None:
                closest_path = self.get_closest(orig_im, self.extractor)[0][0]
            else:
                closest_path = force_closest
            self.multi_cloak_func_args["closest_emb"] = self.get_one_embed(closest_path, self.extractor).clone()

        # Make these things available to the cloak function
        self.multi_cloak_func_args["reconstruct_func"] = self.reconstruct_image
        self.multi_cloak_func_args["image_path"] = orig_im
        self.multi_cloak_func_args["image"] = self.images[orig_im]
        
        # Call the cloak function, passing the list of cropped images as one of the inputs. Returns a mask
        mask, sticker_handler, highpass_handler = self.multi_cloak_function(cropped_list, self.multi_cloak_func_args["tgt_emb"], self.eval_models[self.extractor], self.multi_cloak_loss, self.device, self.multi_cloak_func_args)
        print("Got back a mask with range", np.min(mask), np.max(mask))
        mask *= 255.0
        return mask, sticker_handler, highpass_handler

    # Cloak all images in a dataset, re-using the identity-specific mask if a repeat identity is found.
    def cloak_all_multi(self, save_dir=None, gen_save_path="/", do_paths=None):
        
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
        total_correct_top1 = {model: 0 for model in list(self.eval_models.keys())}
        total_correct_top5 = {model: 0 for model in list(self.eval_models.keys())}
        
        total_time = 0.0

        # Set up the generation pipeline
        use_face_swap = False
        multiplier_style = 0.25
        base_model_idx = 0

        # Iterate through each item in the dataset. Accept do_paths in case we want to call this function externally on a limited dataset
        num = 0
        if do_paths == None:
            these_paths = self.probe_paths
        else:
            these_paths = do_paths

        # Now call on every path we want to 
        for path in these_paths:
            num += 1
            start_full = time.perf_counter()
  
            print("\n\n====== Multi-Cloaking (", str(num), "/", len(self.probe_paths), ":", path, ") =======\n\n")
            if self.verbosity == "log": print("Cloaking image", path)
            print("Right at the start, image is size", self.images[path].shape)

            orig_farthest_path = self.get_farthest(path, self.extractor)
            orig_farthest_path = "data/celeba_verify/group2_gallery/200502_0.jpg"
            orig_closest_path = self.get_closest(path, self.extractor)[0][0]

            # Read in a given image from the real dataset. If we already have a mask for it, apply the mask, otherwise go to the logic that generates the mask.
            name = os.path.basename(path)[:os.path.basename(path).find("_")]
            
            image = self.images[path]

            try:
                # Fetch the identity-specific mask if it's available
                mask = self.multi_map[name]
                sticker_handler = self.sticker_map[name]
                print("==== Successfully loaded premade mask ====")
            except:
                
                # Read in image and embed it with arcface
                pil = Image.open(path).convert("RGB")
                w, h = pil.size
                image_to_gen = np.array(pil)[:, :, ::-1]  # RGB to BGR
                
                # Track the paths of the saved images
                gen_paths = []

                if self.use_real:
                    gen_paths = self.use_real_images(path)
                else:
                    gen_paths = self.generate_images(image_to_gen, path, gen_save_path)
                gen_paths.append(path)

                # Pass all of the fake image paths to the cloak_multi() function, as well as the real image for finding closest and farthest images, getting back the mask
                start = time.perf_counter()
                mask, sticker_handler, highpass_handler = self.cloak_multi(orig_im = path, img_paths=gen_paths, force_target=orig_farthest_path, force_closest=orig_closest_path)
                end = time.perf_counter()
                if self.verbosity == "error": print(f"Multi-cloaking images took {end-start:4f} seconds")
                self.multi_map[name] = mask.copy()
                self.sticker_map[name] = sticker_handler
                self.highpass_map[name] = highpass_handler                    

            image, _ = self.preprocess_image_for_user(path)
            image = image.detach().cpu().numpy().astype(np.int16)
            mask = mask.astype(np.float32)
            
            #Apply pert to the cropped face region of the image and make sure it's a valid image range. Save original im for similarity metrics
            orig_im = image.copy()

            boxes = self.boxes[path]
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])
            
            # Simply add the mask
            apply_start = time.perf_counter()
            mask = mask.astype(np.int16)
            image[ymin:ymin+mask.shape[0], xmin:xmin+mask.shape[1], :] += mask

            if sticker_handler:
                image[ymin:ymin+112, xmin:xmin+112, :] = sticker_handler.apply_stickers(image[ymin:ymin+112, xmin:xmin+112, :], mode="image")
            if highpass_handler:
                image[ymin:ymin+112, xmin:xmin+112, :] = highpass_handler.apply_highpass(image[ymin:ymin+112, xmin:xmin+112, :], mode="image")
            
            end_full = time.perf_counter()
            
            # Make sure image is still in valid range before we convert back to unsigned ints
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
            apply_end = time.perf_counter()
            print(f"Applying mask to image took: {apply_end-apply_start} seconds.")

            # Calculate PSNR, SSIM, and MSE
            image_copy = image.copy()
            orig_image_copy = orig_im.copy()
            try:
                total_ssim_before += structural_similarity(orig_im.astype(np.int8), image.astype(np.int8), channel_axis=2, data_range=255.0)
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

            # Evaluate which image in the gallery the cloaked probe would match with
            results_start = time.perf_counter()
            results_top1 = self.eval_on_all_models(total_path, n=1)
            for model in results_top1.keys():
                total_correct_top1[model] += results_top1[model]
            
            results_top5 = self.eval_on_all_models(total_path, n=5)
            for model in results_top5.keys():
                total_correct_top5[model] += results_top5[model]
            results_end = time.perf_counter()
            print(f"Calculating results took: {results_end-results_start:.4f} seconds")

            print("====== Full Iteration Took:", end_full-start_full, "seconds. =======")
            total_time += end_full-start_full
        # Print similarity metrics
        print("---- Stealthiness Scores ----")
        print(f"Average SSIM %0.4f" % (total_ssim_before / (num)))
        print(f"Average PSNR %0.4f" % (total_psnr_before / (num)))
        print(f"Average MSE %0.4f" % (total_mse_before / (num)))
        print("\n---- Timing ----")
        print(f"\nAverage Iteration Time: {total_time / num:.4f}")
        
        print("\n---- Top-1 Accuracy/PSR -----")
        for model in self.eval_models:
            #print(f"{model} Accuracy: {total_correct_top1[model] / num:.4f}")
            print(f"{model} PSR: {1 - total_correct_top1[model] / num:.4f}")
        print("\n---- Top-5 Accuracy/PSR -----")
        for model in self.eval_models:
            #print(f"{model} Accuracy: {total_correct_top5[model] / num:.4f}")
            print(f"{model} PSR: {1 - total_correct_top5[model] / num:.4f}")

    # Standard preprocessing to prepare images for face embedding extraction
    def preprocess_image_for_user(self, path):
        
        # Skip if we've already preprocessed this image
        if self.preprocessed_ims.get(path, False):
            return torch.tensor(self.images[path]), self.boxes[path]

        # Load image from cache or disk
        try:
            im = self.images[path].copy()
        except KeyError:
            im = np.array(Image.open(path).convert('RGB'))
            self.images[path] = im
        im = torch.tensor(im, dtype=torch.float32, device=self.device)
        H, W = im.shape[:2]

        # Detect face if not cached
        if path not in self.boxes:
            boxes, _ = self.cropper.detect(im)
            if boxes is None or len(boxes) == 0:
                # Fallback: use full image if no face detected
                boxes = [[0, 0, im.shape[1] - 1, im.shape[0] - 1]]
            boxes = boxes[0]
            self.boxes[path] = boxes
        else:
            boxes = self.boxes[path]

        # Parse box coordinates
        xmin, ymin, xmax, ymax = map(float, boxes)

        # Bounds checking before resize
        xmin = max(0, min(xmin, W - 1))
        ymin = max(0, min(ymin, H - 1))
        xmax = max(0, min(xmax, W - 1))
        ymax = max(0, min(ymax, H - 1))

        # Ensure non-empty box
        if xmax <= xmin + 1 or ymax <= ymin + 1:
            if self.verbosity == "warn":
                print(f"Warning: degenerate box for {path}, using full image.")
            xmin, ymin, xmax, ymax = 0, 0, W - 1, H - 1

        box_h = ymax - ymin
        box_w = xmax - xmin

        # Compute anisotropic scale factors
        scale_y = self.cropped_im_size / box_h
        scale_x = self.cropped_im_size / box_w

        # Resize the full image anisotropically
        im = im.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        im_resized = F.interpolate(
            im, scale_factor=(scale_y, scale_x),
            mode="bilinear", align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # Update the face box coordinates for the resized image
        boxes_resized = [
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y
        ]

        # Cache resized image and box
        self.preprocessed_ims[path] = True
        self.images[path] = im_resized.detach().cpu().numpy()
        self.boxes[path] = boxes_resized

        return im_resized, boxes_resized

    # Evaluate the given image on every eval model and return the accuracies
    def eval_on_all_models(self, path, n):
        # CelebA only has 1 for each identity - this is for face verification instead of identification
        if self.probe_dataset_path in ["data/celeba/probe", "data/celeba_small/probe", "data/celeba_verify/group1_probe", "data/celeba_verify/group2_probe", "data/celeba_verify/group3_probe", "data/celeba_verify/group4_probe", "data/celeba_verify/group5_probe"]:
            num_comparisons = 1
        else: # All others have 4 others for each identity
            num_comparisons = 4
        results = {model: 0 for model in self.eval_models.keys()}
        for model in self.eval_models.keys():
            misses = n - 1
            print("=== Eval Model: ", model, " ===")
            closest = self.get_closest(path, model, no_self=True, n=n+num_comparisons-1)
            print(closest)
            for i, closest_one in enumerate(closest):
                if os.path.basename(path)[:os.path.basename(path).find("_")] == os.path.basename(closest_one[0])[:os.path.basename(closest_one[0]).find("_")]:
                    results[model] += 1.0 / num_comparisons
                else:
                    if misses <= 0:
                        break
                    misses -= 1
                    
            print("Score:", results[model])
        return results

    # Generate synthetic images given a single input image
    def generate_images(self, image_to_gen, path, gen_save_path):
        gen_paths = []
        file_name = os.path.basename(path)[:os.path.basename(path).find(".")]

        faces = self.app.get(image_to_gen)
        if faces != []:
            faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
            id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
        else:
            if self.verbosity == "error":
                print(f"!!!!!!ERROR on app face recog for image {path}, switching to second arcface.")
            
            _ = self.get_one_embed(path, self.extractor)
            crop = self.cropped_images[path]
            crop = F.interpolate(
                crop, 
                size=(self.cropped_im_size, self.cropped_im_size),
                mode="bilinear", 
                align_corners=False
            )
            crop = (crop.squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1) * 127.5
            arcface_model = self.app.models['recognition']
            id_emb = torch.tensor(arcface_model.get_feat(crop), dtype=torch.float16).cuda()

        
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

        # Save the paths and the images. We need to save the paths in a list so that the code can use them elsewhere
        start = time.perf_counter()
        for i, im in enumerate(outputs):
            gen_paths.append(gen_save_path + "/" + file_name + "_" + str(i) + ".png")
            if gen_save_path != "/":
                Image.fromarray(im).save(gen_save_path + "/" + file_name + "_" + str(i) + ".png")
                self.images[gen_save_path + "/" + file_name + "_" + str(i) + ".png"] = im
        end = time.perf_counter()
        if self.verbosity == "error": print(f"Saving images took {end-start:4f} seconds")

        return gen_paths
    
    # Use real images instead, in case we want to experiment with this
    def use_real_images(self, path):
        count = 8
        paths = []
        dataset_dir = os.path.dirname(self.probe_dataset_path)
        for im in os.listdir(dataset_dir + "/train"):
            name = im[:im.find("_")]
            path_name = os.path.basename(path)[:os.path.basename(path).find("_")]
            if path_name == name:
                paths.append(dataset_dir + "/train/" + im)
                count -= 1
                if count <= 0:
                    return paths
        
        return paths
