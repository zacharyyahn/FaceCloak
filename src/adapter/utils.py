from dataset import AdapterDataset
from torch.utils.data import DataLoader
from model import ConditionedUnet
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
import lpips
from torch.optim.adam import Adam
import argparse
from tqdm import tqdm
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
cropper = MTCNN(image_size=112, device=device).to(device)
cropper.eval()
extractor = InceptionResnetV1(pretrained='vggface2').to(device)
extractor.eval()

# Define transforms for normalizing images. Can change them on a per-dataset basis
mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])
norm_transform = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
inv_transform = transforms.Compose([
            transforms.Normalize((-mean / std), 1.0 / std)
])

def get_face_embed(im):
    boxes, _ = cropper.detect(im)

    # If we have null boxes, just set the box to the whole image
    if type(boxes) == type(None): 
        boxes = [[0, 0, im.shape[1] - 1, im.shape[0] - 1]]

    # Make sure that the boxes do not exceed the image size
    for i in range(4):
        boxes[0][i] = int(boxes[0][i]) if boxes[0][i] >= 0.0 else 0
    boxes[0][2] = boxes[0][2] if boxes[0][2] < im.shape[1] else im.shape[1] - 1
    boxes[0][3] = boxes[0][3] if boxes[0][3] < im.shape[0] else im.shape[0] - 1
    boxes = boxes[0]

    # Parse out the values
    xmin = int(boxes[0])
    ymin = int(boxes[1])
    xmax = int(boxes[2])
    ymax = int(boxes[3])

    # Crop the area with the bounding box
    crop = im[ymin:ymax, xmin:xmax, :]

    # Resize to 112x112
    #print("Image has mean and std:", torch.mean(crop), torch.std(crop))
    crop = crop / 255.
    #crop = torch.Tensor(cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC))
    crop = torch.permute(crop, (2, 0, 1)).unsqueeze(0)

    crop = norm_transform(crop)

    try:
        embed = extractor(crop)
    except Exception as e:
        print(e)
        return None
    return embed[0].squeeze()

ref_im = "src/adapter/refs/MariahCarey_30.jpg"
ref_im = cv2.imread(ref_im)
ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)
ref_im = cv2.resize(ref_im, (ref_im.shape[1] * 3, ref_im.shape[0] * 3), cv2.INTER_CUBIC)
ref_im = torch.Tensor(ref_im).to(device)
ref_emb = get_face_embed(ref_im).detach().clone()
ref_emb.requires_grad = True

def cosine_dist_loss(cloaked_im):
    #im_embed = get_face_embed(im)
    cloaked_embed = get_face_embed(cloaked_im)
    if cloaked_embed == None:
        return None
    return 1 - F.cosine_similarity(ref_emb, cloaked_embed, dim=-1) # we minimize similarity to maximize distance
    #cloaked_im_embed = get_face_embed(im)

def percep_loss(im, cloaked_im):   
    im = ((im.permute(2, 0, 1) / 255.0) - .5) * 2
    cloaked_im = ((cloaked_im.permute(2, 0, 1) / 255.0) - 0.5) * 2
    return loss_fn_alex(im, cloaked_im)[0][0][0]