# Class for handling creation, application, and updating of highpass filter

import torch
import numpy as np
import torch.nn.functional as F
import math
import torchvision

class HighpassHandler():
    
    # Initialize random highpass of the size corresponding to input_image and global variables    
    def __init__(self, input_image, device):
        self.device = device

        self.HIGHPASS_MAX = 16 / 255.

        lowpass = torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=23, sigma=3.0)
        highpass = input_image - lowpass.squeeze()
        highpass = (highpass - torch.mean(highpass)) / torch.std(highpass)
        self.highpass_filter = 1.0 * (highpass > 0)
        self.highpass_pert = (2 / 255.) * 2 * ((self.highpass_filter) - 0.5)

        self.highpass_filter.to(device)
        self.highpass_pert.to(device)

    # Apply the stickers to the given input image and return the masked image
    # NOTE: In future iterations it may be beneficial to first calculate overlap between stickers and then apply them to not double up some pixels.
    def apply_highpass(self, input_image, mode="crop"):
        if mode == "crop":
            input_image = input_image.clone()
        else:
            input_image = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        
        input_image += self.highpass_pert

        if mode == "crop":
            return input_image
        else:
            return input_image.squeeze().permute(1, 2, 0).cpu().numpy()

    def extract_and_update_highpass(self, masked_input_image, orig_image):
        pert = masked_input_image - orig_image
        pert = pert.detach().clone()

        self.highpass_pert = pert * self.highpass_filter
        self.highpass_pert = torch.clamp(self.highpass_pert, -self.HIGHPASS_MAX, self.HIGHPASS_MAX)
        