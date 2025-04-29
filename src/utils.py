import torch
import numpy as np

def preprocess_tanh(im):
    tanh_constant = 2 - 1e-6
    im /= 255.0
    im -= 0.5
    im *= tanh_constant
    im = torch.tanh(im)
    return im

def reverse_tanh(im):
    tanh_constant = 2 - 1e-6
    im = np.clip(im, a_min=-1+1e-6, a_max=1-1e-6) # make sure we don't cause any infinite values
    im = (np.arctanh(im) / tanh_constant + 0.5) * 255.0
    #print("Null or inf values in reverse_tanh:", np.all(np.isinf(im)), np.all(np.isnan(im)))
    return im

def preprocess_divide(im):
    im = (im - 127.5) / 128.0
    return im

def reverse_divide(im):
    im = np.clip(im, -1, 1)
    im = (im * 128.0) + 127.5
    im = np.clip(im, 0, 255.0)
    return im

def do_nothing(im):
    return im
