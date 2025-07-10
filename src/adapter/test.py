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
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity
from math import sqrt, log10
from utils import cosine_dist_loss

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="src/adapter/test")
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--model_load_path", type=str, default="src/adapter/model.pth")
args = parser.parse_args()

dataset = AdapterDataset(mode="test", dataset_path=args.dataset_path)
dataloader = DataLoader(dataset, batch_size=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConditionedUnet().to(device)
model = torch.load(args.model_load_path, map_location=device)
model.eval()

print("Dataset has length:", len(dataset))

num = 0
total_ssim = 0.0
total_mse = 0.0
total_psnr = 0.0
try:
    os.mkdir(args.save_path)
except:
    pass
for im, mask, mask_path in tqdm(dataloader):
    num += 1
    im = im.to(device)
    mask = mask.to(device)
    cloak = model(im, mask)

    # Post-Process
    cloak = (torch.clamp(cloak, min=-1., max=1.) * 16).squeeze().permute(1, 2, 0)
    im = (((im.squeeze() + 1) / 2) * 255.0).permute(1, 2, 0)
    cloaked_im = torch.clamp(im + cloak, min=0, max=255)
    cloaked_im = cloaked_im.detach().cpu().numpy().astype(np.uint8)
    cloaked_im = cv2.cvtColor(cloaked_im, cv2.COLOR_RGB2BGR)

    name = os.path.basename(mask_path[0])[os.path.basename(mask_path[0]).find("_")+1:-4] + ".jpg"
    cv2.imwrite(args.save_path + "/" + name, cloaked_im)

    im = im.detach().cpu().numpy()

    total_ssim += structural_similarity(im, cloaked_im, channel_axis=2, data_range=255.0)
    mse = np.mean(np.square((im - cloaked_im)))
    total_mse += mse
    total_psnr += 20.0 * log10(255.0 / sqrt(mse))

print(f"Average SSIM %0.4f" % (total_ssim / num))
print(f"Average PSNR %0.4f" % (total_psnr / num))
print(f"Average MSE %0.4f" % (total_mse / num))