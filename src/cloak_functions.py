import torch
import numpy as np
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import cv2

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
    # Copy and save the original to measure modification
    orig_cropped = cropped.detach().clone()

    # Make sure cropped will receive gradients as a fresh leaf tensor
    cropped = cropped.detach().clone()
    cropped.requires_grad = True

    loss_args = {
            "tgt_emb":tgt_emb,
            "dist_func":args["dist_func"]
            }

    if args["loss_func_select"] == "triplet":
        loss_args["closest_emb"] = args["closest_emb"]


    MAX_STEPS = 10
    MIN_STEPS = 1
    #print("MAX STEPS:",MAX_STEPS, "MIN_STEPS:", MIN_STEPS)

    # Iterate for iters iterations
    pbar = tqdm(range(args["iters"]))

    start_loss = 0
    for i in pbar:
        # FIRST LOOP: Optimize for maximum perturbation to disrupt facial detection
        for _ in range(MAX_STEPS):
            cropped = cropped.clone().detach().requires_grad_()
            out_emb = extractor(cropped)
            loss = loss_fn(out_emb, loss_args)
            #print("+ MAX Loss:", loss)
            loss.backward(retain_graph=False)

            signed_grad = torch.sign(cropped.grad)

            with torch.no_grad():
                # Update cropped
                cropped -= args["step"] * signed_grad

                # Clip the perturbation to be within the viable range
                pert = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])

                # Add the perturbation back, but make sure we're within [-1, 1]
                cropped = torch.clip(orig_cropped + pert, min=-1.0, max=1.0)

            del loss, out_emb
        
        # SECOND LOOP: Optimize for minimum perceptual perturbation
        for _ in range(MIN_STEPS):
            cropped = cropped.clone().detach().requires_grad_()
            out_emb = extractor(cropped)
            loss = args["percep_loss"](cropped, orig_cropped)
            #print("- MIN Loss:", loss)
            loss.backward(retain_graph=False)

            signed_grad = torch.sign(cropped.grad)

            with torch.no_grad():
                # Update cropped
                cropped -= args["step"] * signed_grad

                # Clip the perturbation to be within the viable range
                pert = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])

                # Add the perturbation back, but make sure we're within [-1, 1]
                cropped = torch.clip(orig_cropped + pert, min=-1.0, max=1.0)

            del loss, out_emb
        
        # if i == 0:
        #     start_loss = loss.item()
        # pbar.set_postfix({'start_loss': start_loss, 'end_loss':loss.item()})
        #print("Loss is", loss.item())

        # Get the sign of the gradient
        
        #cropped = cropped.detach().clone()
        torch.cuda.empty_cache()

    diff = torch.clip(cropped - orig_cropped, min=-args["max_pert"], max=args["max_pert"])
    cropped = torch.clip(orig_cropped + diff, min=-1, max=1)
    return cropped

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
