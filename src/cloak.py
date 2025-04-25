from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from cloaker import Cloaker

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to clean images")
parser.add_argument("--extractor_path", type=str, default=None, help="Path to extractor")
parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
parser.add_argument("--cropped_im_size", type=int, default=112, help="Crop size to apply MTCNN")
parser.add_argument("--target_pool_size", type=int, default=100, help="Number of images to compare with to find target")
parser.add_argument("--num_cloaked_images", type=int, default=100, help="Number of images to cloak")
parser.add_argument("--num_dataset_images", type=float, default=1.0, help="Number of images to take from dataset")
parser.add_argument("--cloak_save_path", type=str, default="data/pubfig_cloaked", help="Path to save cloaked images")
parser.add_argument("--verbosity", type=str, default="none", help="Error verbosity level")
args = parser.parse_args()

#assert args.num_dataset_images > args.num_cloaked_images
#assert args.num_dataset_images > args.target_pool_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
arcface = torch_arcface_insightface(device).to(device)
mtcnn = MTCNN(image_size=args.cropped_im_size, device=device).to(device)
cloaker = Cloaker(args.dataset_path, extractor=arcface, cropper=mtcnn, batch_size=args.batch_size, cropped_im_size=args.cropped_im_size, target_pool_size=args.target_pool_size, num_dataset_images=args.num_dataset_images, device=device, verbosity=args.verbosity)
#cloaker.get_embeds()
cloaker.cloak_all(save_dir = args.cloak_save_path, num_images=args.num_cloaked_images)
