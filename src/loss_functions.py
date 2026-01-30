from pytorch_msssim import SSIM
import numpy as np
import scipy.stats as st
import torch
import lpips

loss_fn_alex = lpips.LPIPS(net='alex').to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def untarget_loss(out_emb, args):
    return -args["dist_func"](out_emb, args["tgt_emb"])

def fawkes_loss(out_emb, args):
    return args["dist_func"](out_emb, args["tgt_emb"])

# Decrease distance between self and target, and increase distance between self and closest
def triplet_loss(out_emb, args):
    far = args["dist_func"](out_emb, args["tgt_emb"])
    close = args["dist_func"](out_emb, args["closest_emb"])
    #print("Far dist:", far, "Close dist:", close)
    return far - close

def dssim_loss(out_img, tgt_img):
    #out_img = np.transpose(out_img, (2, 0, 1))
    #tgt_img = np.transpose(tgt_img, (2, 0, 1))
    #out_img = torch.Tensor(out_img).unsqueeze(0)
    #tgt_img = torch.Tensor(tgt_img).unsqueeze(0)
    out_img = torch.add(torch.div(out_img, 2.0), .5)
    tgt_img = torch.add(torch.div(tgt_img, 2.0), .5)
    ssim_loss = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
    ss = ssim_loss(out_img, tgt_img)
    return 1 - ss

def lpips_loss(out_img, tgt_img):
    #out_img = (out_img.copy() - 127.5) / 128.
    #tgt_img = (tgt_img.copy() - 127.5) / 128.
    #print("LPIPS: Out image, tgt image have ranges", np.min(out_img), np.max(out_img), np.min(tgt_img), np.max(tgt_img))
    lpips_loss = loss_fn_alex(out_img, tgt_img)[0][0][0][0]
    return lpips_loss

def mse_loss(out_img, tgt_img):
    return torch.mean(torch.multiply(out_img - tgt_img, out_img - tgt_img))
