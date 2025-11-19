"""
Class for handling creation, application, and updating of stickers
"""

import torch
import numpy as np
from facenet_pytorch import MTCNN
import torch.nn.functional as F
import math

class StickerHandler():
    
    # Initialize random stickers of the size corresponding to input_image and global variables    
    def __init__(self, input_image, device):
        self.sticker_model = MTCNN(image_size=112, device=device).to(device).eval()
        self.device = device
        image_size_x = input_image.size()[-1]
        image_size_y = input_image.size()[-2]

        self.STICKER_MAX = 32 / 255.

        # Get initial landmarks
        input_image = input_image.clone()
        resized_image = F.interpolate(input_image, (112, 112)).squeeze().permute(1, 2, 0)
        _, _, self.landmarks = self.sticker_model.detect(127.5 * (resized_image + 1.0), landmarks=True)
        try:
            if len(self.landmarks) == 0:
                print("Not able to compute stickers")
        except:
            self.landmarks = np.array([[
                [2* image_size_x / 3., image_size_y / 3.],
                [image_size_x / 3, image_size_y /3],
                [image_size_x / 2., image_size_y / 2.],
                [image_size_x / 3, 2 * image_size_y / 3],
                [2 * image_size_x / 3, 2 * image_size_y / 3]
                              ]])

        # Set the parameters for defining region sizes
        self.EYE_BOX_WIDTH = int(image_size_x / 5) + int(image_size_x / 5) % 2
        self.EYE_BOX_HEIGHT = int(image_size_y / 11) + int(image_size_y / 11) % 2
        self.NOSE_BOX_WIDTH = int(image_size_x / 5) + int(image_size_x / 5) % 2
        self.NOSE_BOX_HEIGHT = int(image_size_y / 4) + int(image_size_y / 4) % 2
        self.MOUTH_BOX_WIDTH = int(self.landmarks[0, -1, 0]) - int(self.landmarks[0, -2, 0]) + (int(self.landmarks[0, -1, 0]) - int(self.landmarks[0, -2, 0])) % 2
        self.MOUTH_BOX_HEIGHT = int(image_size_y / 4) + int(image_size_y / 4) % 2

        # Initialize the stickers
        self.right_eye_sticker = (2. / 255) * 2. * (torch.rand((3, self.EYE_BOX_HEIGHT, self.EYE_BOX_WIDTH)).to(device) - 0.5)
        self.left_eye_sticker = (2. / 255) * 2. * (torch.rand((3, self.EYE_BOX_HEIGHT, self.EYE_BOX_WIDTH)).to(device) - 0.5)
        self.nose_sticker = (2. / 255) * 2. * (torch.rand((3, self.NOSE_BOX_HEIGHT, self.NOSE_BOX_WIDTH)).to(device) - 0.5)
        self.mouth_sticker = (2. / 255) * 2. * (torch.rand((3, self.MOUTH_BOX_HEIGHT, self.MOUTH_BOX_WIDTH)).to(device) - 0.5)

    # Apply the stickers to the given input image and return the masked image
    # NOTE: In future iterations it may be beneficial to first calculate overlap between stickers and then apply them to not double up some pixels.
    def apply_stickers(self, input_image, mode="crop"):
        
        # If it's possible to find stickers then apply them
        if mode == "crop":
            input_image = input_image.clone()
            resized_image = F.interpolate(input_image, (112, 112)).squeeze().permute(1, 2, 0)
        else:
            input_image = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            resized_image = F.interpolate(input_image, (112, 112)).squeeze().permute(1, 2, 0)

        image_size_x = input_image.size()[-1]
        image_size_y = input_image.size()[-2]

        # Get the centroid coordinates of the landmarks
        right_eye_x = int(self.landmarks[0][0][0])
        right_eye_y = int(self.landmarks[0][0][1])
        left_eye_x = int(self.landmarks[0][1][0])
        left_eye_y = int(self.landmarks[0][1][1])
        nose_x = int(self.landmarks[0][2][0])
        nose_y = int(self.landmarks[0][2][1])
        mouth_x = int(self.landmarks[0][3][0]/2 + self.landmarks[0][4][0]/2)
        mouth_y = int(self.landmarks[0][3][1]/2 + self.landmarks[0][4][1]/2)

        orig_image = input_image.clone()

        # Add the right eye sticker
        try:
            input_image[..., 
                right_eye_y - int(self.EYE_BOX_HEIGHT / 2):right_eye_y + int(self.EYE_BOX_HEIGHT / 2),
                right_eye_x - int(self.EYE_BOX_WIDTH / 2):right_eye_x + int(self.EYE_BOX_WIDTH / 2)
                ] += self.right_eye_sticker * (1.0 if mode == "crop" else 255.0)
        except:
            pass
            
        # Add the left eye sticker
        try:
            input_image[..., 
                left_eye_y - int(self.EYE_BOX_HEIGHT / 2):left_eye_y + int(self.EYE_BOX_HEIGHT / 2),
                left_eye_x - int(self.EYE_BOX_WIDTH / 2):left_eye_x + int(self.EYE_BOX_WIDTH / 2)
                ] += self.left_eye_sticker * (1.0 if mode == "crop" else 255.0)
        except:
            pass
            
        # Add the nose sticker
        try:
            input_image[..., 
                nose_y - int(self.NOSE_BOX_HEIGHT / 2):nose_y + int(self.NOSE_BOX_HEIGHT / 2),
                nose_x - int(self.NOSE_BOX_WIDTH / 2):nose_x + int(self.NOSE_BOX_WIDTH / 2)
                ] += self.nose_sticker * (1.0 if mode == "crop" else 255.0)
        except:
            pass
            
        # Add the mouth sticker
        try:
            input_image[..., 
                mouth_y - int(self.MOUTH_BOX_HEIGHT / 2):mouth_y + int(self.MOUTH_BOX_HEIGHT / 2),
                mouth_x - int(self.MOUTH_BOX_WIDTH / 2):mouth_x + int(self.MOUTH_BOX_WIDTH / 2)
                ] += self.mouth_sticker * (1.0 if mode == "crop" else 255.0)
        except:            
            pass

        # Make sure pert is in the suitable range
        

        if mode == "crop":
            pert = torch.clamp(input_image - orig_image, min=-self.STICKER_MAX, max=self.STICKER_MAX)
            input_image = orig_image + pert
            return input_image
        else:
            pert = torch.clamp(input_image - orig_image, min=-255.0 * self.STICKER_MAX, max=255.0 * self.STICKER_MAX)
            input_image = orig_image + pert
            return input_image.squeeze().permute(1, 2, 0).cpu().numpy()

    def extract_and_update_stickers(self, masked_input_image, orig_image):
        
        pert = masked_input_image - orig_image
        pert = pert.detach().clone()

        # Get the centroid coordinates of the landmarks
        right_eye_x = int(self.landmarks[0][0][0])
        right_eye_y = int(self.landmarks[0][0][1])
        left_eye_x = int(self.landmarks[0][1][0])
        left_eye_y = int(self.landmarks[0][1][1])
        nose_x = int(self.landmarks[0][2][0])
        nose_y = int(self.landmarks[0][2][1])
        mouth_x = int(self.landmarks[0][3][0]/2 + self.landmarks[0][4][0]/2)
        mouth_y = int(self.landmarks[0][3][1]/2 + self.landmarks[0][4][1]/2)

        # Now extract the stickers
        # Extract the right eye sticker
        self.right_eye_sticker = pert[..., 
            right_eye_y - int(self.EYE_BOX_HEIGHT / 2):right_eye_y + int(self.EYE_BOX_HEIGHT / 2),
            right_eye_x - int(self.EYE_BOX_WIDTH / 2):right_eye_x + int(self.EYE_BOX_WIDTH / 2)
            ].squeeze(0)

        # Add the left eye sticker
        self.left_eye_sticker = pert[..., 
            left_eye_y - int(self.EYE_BOX_HEIGHT / 2):left_eye_y + int(self.EYE_BOX_HEIGHT / 2),
            left_eye_x - int(self.EYE_BOX_WIDTH / 2):left_eye_x + int(self.EYE_BOX_WIDTH / 2)
            ].squeeze(0)
        
        # Add the nose sticker
        self.nose_sticker = pert[..., 
            nose_y - int(self.NOSE_BOX_HEIGHT / 2):nose_y + int(self.NOSE_BOX_HEIGHT / 2),
            nose_x - int(self.NOSE_BOX_WIDTH / 2):nose_x + int(self.NOSE_BOX_WIDTH / 2)
            ].squeeze(0)

        # Add the nose sticker
        self.mouth_sticker = pert[..., 
            mouth_y - int(self.MOUTH_BOX_HEIGHT / 2):mouth_y + int(self.MOUTH_BOX_HEIGHT / 2),
            mouth_x - int(self.MOUTH_BOX_WIDTH / 2):mouth_x + int(self.MOUTH_BOX_WIDTH / 2)
            ].squeeze(0)

        self.right_eye_sticker = torch.clamp(self.right_eye_sticker, min=-self.STICKER_MAX, max=self.STICKER_MAX)
        self.left_eye_sticker = torch.clamp(self.left_eye_sticker, min=-self.STICKER_MAX, max=self.STICKER_MAX)
        self.nose_sticker = torch.clamp(self.nose_sticker, min=-self.STICKER_MAX, max=self.STICKER_MAX)
        self.mouth_sticker = torch.clamp(self.mouth_sticker, min=-self.STICKER_MAX, max=self.STICKER_MAX)

    def print_stickers(self):
        print("Stickers are:")
        print("Right eye sticker:", self.right_eye_sticker)
        
        
        