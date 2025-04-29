import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def sgd_cloak(cropped, tgt_emb, extractor, loss_fn, device, args):
    # Copy and save the original to measure modification
    # orig_cropped = cropped.detach().clone()

    # Adjust the perturbation budget and step size
    #pert_budget = pert_budget * (orig_max - orig_min)
    #step = step * (orig_max - orig_min)

    # Make sure cropped will receive gradients as a fresh leaf tensor
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

    #optimizer = torch.optim.Adam([cropped], lr=args["lr"])

    start_loss = 0
    for i in pbar:
        #optimizer.zero_grad()
        cropped = cropped.clone().detach().requires_grad_()
        #cropped.requires_grad = True
        out_emb = extractor(cropped)
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
    cropped = cropped[0]
    return cropped

def pgd_cloak(cropped, tgt_emb, extractor, loss_fn, device, args):
    # Copy and save the original to measure modification
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
    cropped = cropped[0]
    return cropped
