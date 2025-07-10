import torch
import numpy as np

def preprocess_tanh(current_im):
    tanh_constant = 2 - 1e-6
    current_im /= 255.0
    current_im -= 0.5
    current_im *= tanh_constant
    current_im = torch.tanh(current_im)
    return current_im

def reverse_tanh(current_im):
    tanh_constant = 2 - 1e-6
    current_im = np.clip(current_im, a_min=-1+1e-6, a_max=1-1e-6) # make sure we don't cause any infinite values
    current_im = (np.arctanh(current_im) / tanh_constant + 0.5) * 255.0
    #print("Null or inf values in reverse_tanh:", np.all(np.isinf(im)), np.all(np.isnan(im)))
    return current_im

def preprocess_divide(current_im):
    current_im = (current_im - 127.5) / 128.0
    return current_im

def reverse_divide(h_current_im):
    h_current_im = np.clip(h_current_im, -1, 1)
    h_current_im = (h_current_im * 128.0) + 127.5
    h_current_im = np.clip(h_current_im, 0, 255.0)
    return h_current_im

def do_nothing(im):
    return im
