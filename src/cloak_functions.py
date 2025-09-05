import torch
import numpy as np
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import cv2
from arc2face.Arc2Face.arc2face import CLIPTextModelWrapper, project_face_embs
from utils import pipeline_forward_with_grad
import gc
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from insightface.app import FaceAnalysis
from PIL import Image
import os

def afog_cloak(cropped, tgt_emb, extractor, loss_fn, device, args):
    orig_cropped = cropped.detach().clone()
    orig_cropped.requires_grad = True
    attn_lr = 0.01

    # Adjust the perturbation budget and step size
    #pert_budget = pert_budget * (orig_max - orig_min)
    #step = step * (orig_max - orig_min)

    # Make sure cropped will receive gradients as a fresh leaf tensor
    cropped = cropped.detach().clone()
    #cropped.requires_grad = True

    attn_map = torch.ones_like(cropped).to(device).float()
    #attn_map = torch.normal(1, 0.25, cropped.size()).to(device)
    attn_map.requires_grad = True

    pert = (2 * args["step"]) * torch.rand(cropped.size()).to(device) - args["step"]
    #pert = torch.zeros_like(cropped).to(device).float()
    pert.requires_grad = True

    loss_args = {
            "tgt_emb":tgt_emb,
            "dist_func":args["dist_func"]
            }

    if args["loss_func_select"] == "triplet":
        loss_args["closest_emb"] = args["closest_emb"]

    # Do the first step of the update
    cropped = orig_cropped + attn_map * pert

    # Iterate for iters iterations
    pbar = tqdm(range(args["iters"]))
    start_loss = 0
    for i in pbar:
        out_emb = extractor(cropped)

        # If we have a perceptual loss component, also add that to our loss calculation. Otherwise just do the normal loss
        if args["percep_loss"]:
            percep_factor = args["percep_loss_weight"]
            loss = loss_fn(out_emb, loss_args)
            percep_loss = args["percep_loss"](cropped, orig_cropped)
            loss = loss + float(args["percep_loss_weight"]) * percep_loss
        else:
            loss = loss_fn(out_emb, loss_args)
        
        loss.backward(retain_graph=False)

        if i == 0:
            start_loss = loss.item()
        pbar.set_postfix({'start_loss': start_loss, 'end_loss':loss.item()})

        # Get the sign of the gradient
        if torch.std(attn_map.grad) == 0:
            attn_grad = torch.zeros_like(attn_map)
        else:
            attn_grad = (attn_map.grad - torch.mean(attn_map.grad)) / torch.std(attn_map.grad)
        pert_grad = torch.sign(pert.grad)

        with torch.no_grad():

            attn_map = attn_map - attn_lr * attn_grad
            pert = pert - args["step"] * pert_grad
            
        # Update the actual changes to the attention map and perturbation
        attn_map = attn_map.detach().clone()
        attn_map.requires_grad = True
        pert = pert.detach().clone()
        pert.requires_grad = True

        # Reformulate the perturbed image and prepare to repeat. Detach from graph so we don't get double iteration
        new_pert = torch.clip(torch.multiply(attn_map, pert), -args["max_pert"], args["max_pert"])
        cropped = cropped.detach().clone()
        cropped = torch.clip(cropped + new_pert, min=-1.0, max=1.0)

        del loss, out_emb
        torch.cuda.empty_cache()

    cropped = cropped.clone()
    diff = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])
    cropped = torch.clip(orig_cropped + diff, min=-1.0, max=1.0)
    #cropped = cropped[0]
    return cropped

def sgd_cloak(cropped, tgt_emb, extractor, loss_fn, device, args):
    # Make sure cropped will receive gradients as a fresh leaf tensor
    orig_cropped = cropped.detach().clone()
    orig_cropped.requires_grad = True
    cropped = cropped.detach().clone()
    cropped.requires_grad = True

    # Iterate for iters iterations
    pbar = tqdm(range(args["iters"]))

    loss_args = {
            "tgt_emb":tgt_emb,
            "dist_func":args["dist_func"]
            }
    
    if args["loss_func_select"] == "triplet":
        loss_args["closest_emb"] = args["closest_emb"]

    start_loss = 0
    for i in pbar:
        #optimizer.zero_grad()
        cropped = cropped.clone().detach().requires_grad_()
        #cropped.requires_grad = True
        out_emb = extractor(cropped)
        
        # If we have a perceptual loss component, also add that to our loss calculation. Otherwise just do the normal loss
        if args["percep_loss"]:
            loss = loss_fn(out_emb, loss_args)
            percep_loss = args["percep_loss"](cropped, orig_cropped)
            loss = loss + float(args["percep_loss_weight"]) * percep_loss
        else:
            loss = loss_fn(out_emb, loss_args)
        loss.backward(retain_graph=False)

        #optimizer.step()
        if i == 0:
            start_loss = loss.item()
        pbar.set_postfix({'start_loss': start_loss, 'end_loss':loss.item()})
        #print("Loss is", loss.item())
        cropped = cropped - args["lr"] * cropped.grad
        cropped.grad = None
        #cropped = cropped.detach().clone()
        torch.cuda.empty_cache()

    #diff = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])
    cropped = torch.clip(cropped, min=-1.0, max=1.0)
    return cropped


def minmax_cloak(cropped, tgt_emb, extractor, loss_fn, device, args):
    # Copy and save the original to measure modification. If we have access to the original image  (before multi-deepfake) then use that. Otherwise use the image given
    if args["original_image"] is not None:
        print("** Using original image **")
        orig_cropped = args["original_image"].detach().clone()
    else:
        orig_cropped = cropped.detach().clone()

    # Make sure cropped will receive gradients as a fresh leaf tensor
    cropped = cropped.detach().clone().to(dtype=torch.float32)
    cropped.requires_grad = True

    loss_args = {
            "tgt_emb":tgt_emb,
            "dist_func":args["dist_func"]
            }

    if args["loss_func_select"] == "triplet":
        loss_args["closest_emb"] = args["closest_emb"]

    NUM_MINMAX_ITERATIONS = 2
    GEN_ITERATIONS = 2
    GEN_LEARNING_RATE = 100.0

    print("Original crop size is", torch.min(cropped), torch.max(cropped))

    for minmax_iter in range(NUM_MINMAX_ITERATIONS):
        print(f"---- Min-Max Iter {minmax_iter} -----")
        
        # ------------
        # FIRST: Generate the perturbation against the current image
        # ------------
        
        # Iterate for iters iterations
        pbar = tqdm(range(args["iters"]), total=args["iters"], desc="Generating Perturbation")
        start_loss = 0
        
        for i in pbar:
            cropped = cropped.clone().detach().to(dtype=torch.float32).requires_grad_(True)

            out_emb = extractor(cropped)
            
            # If we have a perceptual loss component, also add that to our loss calculation. Otherwise just do the normal loss
            if args["percep_loss"]:
                loss = loss_fn(out_emb, loss_args)
                percep_loss = args["percep_loss"](cropped.to(dtype=torch.float16), orig_cropped.to(dtype=torch.float16))
                print("Perturb loss is:", loss)
                print("Percep loss is:", percep_loss)
                loss = loss + float(args["percep_loss_weight"]) * percep_loss
            else:
                loss = loss_fn(out_emb, loss_args)
                
            loss.backward(retain_graph=False)
            
            if i == 0:
                start_loss = loss.item()
            pbar.set_postfix({'start_loss': start_loss, 'end_loss':loss.item()})
            pbar.update(1)

            # Get the sign of the gradient
            signed_grad = torch.sign(cropped.grad)

            with torch.no_grad():
                # Update cropped
                cropped -= args["step"] * signed_grad

                # Clip the perturbation to be within the viable range
                pert = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])

                # Add the perturbation back, but make sure we're within [-1, 1]
                cropped = torch.clip(orig_cropped + pert, min=0, max=1.0)

            # Clean up
            del loss, out_emb
            if cropped.grad is not None:
                cropped.grad = None
            torch.cuda.empty_cache()

        # Final clipping and conversion for the generation phase
        with torch.no_grad():
            diff = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])
            pert = pert.detach().clone()
            cropped = torch.clip(orig_cropped + diff, min=-1, max=1)
            cropped = cropped.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
            
        print("After PGD cropped is", cropped.shape)
        print("After PGD cropped has range", np.min(cropped), np.max(cropped))

        gc.collect()
        torch.cuda.empty_cache()

        # ---------
        # SECOND: Generate a new image that the perturbation fails on
        # ---------

        # Get arcface embeddings of the orig cropped image, using the fallback if necessary
        try:
            faces = args["app"].get(255.0 * ((cropped + 1) / 2))
            faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
            id_emb = torch.tensor(faces['embedding'], dtype=torch.float16, device=device)[None]
        except: # Note, the above try seems to always fail because of the first line
            #print(f"Error on app face recog for image, switching to second arcface INSIDE MINMAX.")
            crop = args["norm_function"](cropped)
            crop = torch.tensor(crop.transpose((2, 0, 1)), dtype=torch.float16, device=device).unsqueeze(0)
            id_emb = torch.tensor(args["backup_arcface"](crop), dtype=torch.float16, device=device).detach().clone()
        
        id_emb = torch.zeros((1, 512), dtype=torch.float16).to(args["device"])
        id_emb = project_face_embs(args["pipeline"], id_emb).detach().clone() # ensure no history
        
        #Prepare pert for applying to generated images - keep as tensor on GPU
        pert = pert.squeeze().permute(1, 2, 0).detach().clone()
        
        gc.collect()
        torch.cuda.empty_cache()

        for k in range(GEN_ITERATIONS+1):
            # Generate a new image from the embedding
            id_emb = id_emb.detach().to(device=device, dtype=torch.float16).contiguous().requires_grad_(True)

            images, _ = pipeline_forward_with_grad(
                args["pipeline"],
                prompt_embeds=id_emb,
                num_inference_steps=25,
                guidance_scale=7.5,
                height=256,
                width=256,
            )

            image = images[0].permute(1, 2, 0).type(dtype=torch.float16).contiguous()
            image_save = Image.fromarray((image.clone().detach().cpu().numpy() * 255.0).astype(np.uint8))
            this_image_path = os.path.basename(args["image_path"][:-4])
            image_save.save(f"data/cloaked/test_minmax/{this_image_path}_{minmax_iter}_{k}.png")

            with torch.no_grad():
                # Move to CPU for cropper detection to avoid device conflicts
                image_cpu = (255.0 * image.clone().detach().cpu()).numpy()
                boxes, _ = args["cropper"].detect(image_cpu)
            
            # Process the boxes
            if type(boxes) == type(None): 
                    boxes = [[0, 0, image.size()[1] - 1, image.size()[0] - 1]]

            # Make sure that the boxes do not exceed the image size
            for i in range(4):
                boxes[0][i] = int(boxes[0][i]) if boxes[0][i] >= 0.0 else 0
            boxes[0][2] = boxes[0][2] if boxes[0][2] < image.size()[1] else image.size()[1] - 1
            boxes[0][3] = boxes[0][3] if boxes[0][3] < image.size()[0] else image.size()[0] - 1
            boxes = boxes[0]
            
            xmin = int(boxes[0])
            ymin = int(boxes[1])
            xmax = int(boxes[2])
            ymax = int(boxes[3])

            # If we've reached the end, we just want to continue with the new image
            if k == GEN_ITERATIONS:
                with torch.no_grad():
                    cropped = image[ymin:ymax, xmin:xmax, :].detach().clone().permute(2, 0, 1).unsqueeze(0)
                    cropped = F.interpolate(cropped, size=(112, 112))
                    orig_cropped = cropped.clone() # we do this for perceptual comparison in the PGD loop
                break

            # Apply perturbation with proper device handling
            with torch.no_grad():
                face_region = image[ymin:ymax, xmin:xmax, :]
                
                if pert.shape == face_region.shape:
                    image[ymin:ymax, xmin:xmax,:] += pert.to(image.device)
                else:
                    pert_resized = F.interpolate(
                        pert.permute(2, 0, 1).unsqueeze(0), 
                        size=(ymax - ymin, xmax - xmin), 
                        mode="bilinear", 
                        align_corners=False
                    ).squeeze().permute(1, 2, 0)
                    image[ymin:ymax, xmin:xmax, :] += pert_resized.to(image.device)
            #print("Image is after mask add is:", image)
            # Measure loss on the perturbed gen image
            image = image.permute(2, 0, 1)
            
            # Ensure proper device placement and data type
            image_input = (255.0 * image).float().unsqueeze(0).to(device=device, dtype=torch.float32)
            out_emb = extractor(image_input)
            
            loss = -loss_fn(out_emb, loss_args) #negative because we want to do the opposite
            print("Gen loss is:", loss.item())
            
            loss.backward(retain_graph=False)

            # Update the gen image to do worse next time
            with torch.no_grad():
                if id_emb.grad is not None:
                    print("Found id_emb grad and propagated")
                    old_id_emb = id_emb.clone()
                    id_emb = id_emb - GEN_LEARNING_RATE * id_emb.grad
                    old_id_emb[:, 4, :] = id_emb[:, 4, :]
                    id_emb = old_id_emb.clone().detach()
            
            # Clean up
            if id_emb.grad is not None:
                id_emb.grad = None
            del loss, out_emb, image_input
            
                    
            torch.cuda.empty_cache()
            gc.collect()

    print("pert has range", torch.min(pert), torch.max(pert))

    return 255.0 * pert.to(dtype=torch.float32).detach().cpu().numpy()
    
def pgd_cloak(cropped, tgt_emb, extractor, loss_fn, device, args):
    # Copy and save the original to measure modification. If we have access to the original image  (before multi-deepfake) then use that. Otherwise use the image given
    if args["original_image"] is not None:
        print("** Using original image **")
        orig_cropped = args["original_image"].detach().clone()
    else:
        orig_cropped = cropped.detach().clone()

    # Adjust the perturbation budget and step size
    #pert_budget = pert_budget * (orig_max - orig_min)
    #step = step * (orig_max - orig_min)

    # Make sure cropped will receive gradients as a fresh leaf tensor
    cropped = cropped.detach().clone()
    cropped.requires_grad = True

    loss_args = {
            "tgt_emb":tgt_emb,
            "dist_func":args["dist_func"]
            }

    if args["loss_func_select"] == "triplet":
        loss_args["closest_emb"] = args["closest_emb"]

    # Iterate for iters iterations
    pbar = tqdm(range(args["iters"]))
    start_loss = 0
    for i in pbar:
        cropped = cropped.clone().detach().requires_grad_()

        #cropped.requires_grad = True
        out_emb = extractor(cropped)
        
        # If we have a perceptual loss component, also add that to our loss calculation. Otherwise just do the normal loss
        if args["percep_loss"]:
            loss = loss_fn(out_emb, loss_args)
            percep_loss = args["percep_loss"](cropped, orig_cropped)
            # print("Loss is:", loss)
            # print("Percep loss is:", percep_loss)
            # print("High pass loss is", high_pass_loss)
            loss = loss + float(args["percep_loss_weight"]) * percep_loss
            loss = loss_fn(out_emb, loss_args)
        loss.backward(retain_graph=False)
        
        if i == 0:
            start_loss = loss.item()
        pbar.set_postfix({'start_loss': start_loss, 'end_loss':loss.item()})
        #print("Loss is", loss.item())

        # Get the sign of the gradient
        signed_grad = torch.sign(cropped.grad)

        with torch.no_grad():
            # Update cropped
            cropped -= args["step"] * signed_grad

            # Clip the perturbation to be within the viable range
            pert = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])

            # Add the perturbation back, but make sure we're within [-1, 1]
            cropped = torch.clip(orig_cropped + pert, min=-1.0, max=1.0)

        del loss, out_emb
        cropped.grad = None
        #cropped = cropped.detach().clone()
        torch.cuda.empty_cache()

    diff = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])
    cropped = torch.clip(orig_cropped + diff, min=-1, max=1)
    return cropped

def pgd_cloak_multi(cropped_list, tgt_emb, extractor, loss_fn, device, args):
    # Set up the necessary elements here and the parameters for the loss function
    cropped_list = [cropped.detach().clone().to(device) for cropped in cropped_list]
    for cropped in cropped_list: cropped.requires_grad = True

    loss_args = {
            "tgt_emb":tgt_emb,
            "dist_func":args["dist_func"]
            }

    if args["loss_func_select"] == "triplet":
        loss_args["closest_emb"] = args["closest_emb"]

    # Initialize perturbation
    pert = args["max_pert"] * 2*(torch.rand(cropped_list[0].size()) - .5)
    pert.requires_grad = True
    pert = pert.to(device)

    # Two nested loops: One for iterations of the attack, the inner one for images in the cropped_list
    pbar = tqdm(total=args["iters"])
    start_loss = 0
    for i in range(args["iters"]):
        for cropped in cropped_list:

            # Add the perturbation and get the new embedding
            orig_cropped = cropped.clone()
            cropped = cropped + pert
            cropped = cropped.clone().detach()
            cropped.requires_grad = True
            out_emb = extractor(cropped)

            #blur = cv2.GaussianBlur(cropped, (13, 13), 0)

            # If we have a perceptual loss component, also add that to our loss calculation. Otherwise just do the normal loss
            if args["percep_loss"]:
                loss = loss_fn(out_emb, loss_args)
                percep_loss = args["percep_loss"](cropped, orig_cropped)
                loss = loss + float(args["percep_loss_weight"]) * percep_loss
            else:
                loss = loss_fn(out_emb, loss_args)
            loss.backward(retain_graph=False)

            # Calculate loss with current image
            if i == 0:
                start_loss = loss.item()
            pbar.update(1)
            pbar.set_postfix({'start_loss': start_loss, 'end_loss':loss.item()})

            # Get the sign of the gradient
            signed_grad = torch.sign(cropped.grad)

            # Update pert according to current image, discarding cropped since we don't care
            with torch.no_grad():
                # Update cropped
                cropped -= args["step"] * signed_grad

                # Clip the perturbation to be within the viable range
                pert = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])

                # Prepare pert for next iteration
                pert = pert.detach().clone()
                pert.requires_grad = True

            del loss, out_emb
            cropped.grad = None
            #cropped = cropped.detach().clone()
            torch.cuda.empty_cache()

    # Make sure perturbation is clipped and return perturbation
    return pert[0].detach().clone().cpu().squeeze().permute(1, 2, 0).numpy()
