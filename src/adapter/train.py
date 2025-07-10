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
import gc
import argparse
from tqdm import tqdm
from utils import get_face_embed, cosine_dist_loss, percep_loss
import psutil
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--model_save_path", type=str, default="model.pth")
parser.add_argument("--load_model_from", type=str, default="None")
args = parser.parse_args()

dataset = AdapterDataset(dataset_path=args.dataset_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConditionedUnet().to(device)
if args.load_model_from != "None":
    model = torch.load(args.load_model_from, map_location=device)
model.train()

optimizer = Adam(model.parameters(), lr=args.lr)

process = psutil.Process(os.getpid())

best_loss = 1000.0
for i in range(args.epochs):
    print("-------- Epoch:", i, "--------")
    total_loss = 0
    num = 0
    pbar = tqdm(dataloader)
    for im, mask, _ in pbar:
        num += 1
        im = im.to(device)
        mask = mask.to(device)
        cloak = model(im, mask).squeeze().permute(1, 2, 0)
        cloak = torch.clamp(cloak, min=-1., max=1) * 16.
        im = (((im.squeeze() + 1) / 2) * 255.0).permute(1, 2, 0)
        cloaked_im = torch.clamp(im + cloak, min=0, max=255)
        vis_loss = percep_loss(im, cloaked_im)
        dist_loss = cosine_dist_loss(cloaked_im)
        if dist_loss == None: # some images are too small so we need to skip them
            continue
        optimizer.zero_grad()
        loss = dist_loss + vis_loss
        #pbar.set_postfix({"mem (MB)":process.memory_info().rss / (1024 * 1014), "pid":os.getpid(), "num tensors":len([obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.is_cuda])})
        pbar.set_postfix({'dist_loss':dist_loss.item(),"vis_loss":vis_loss.item(), "avg_loss": total_loss / num})
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
    average_loss = total_loss / num
    print("Average Training Loss:", average_loss)
    if average_loss < best_loss:
        print("New best model found, saving...")
        best_loss = average_loss
        torch.save(model, args.model_save_path)
    