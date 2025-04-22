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
args = parser.parse_args()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
arcface = torch_arcface_insightface(device)
mtcnn = MTCNN(image_size=112)
cloaker = Cloaker(args.dataset_path, extractor=arcface, cropper=mtcnn, batch_size=args.batch_size)
cloaker.get_embeds()
cloaker.cloak_all(save_dir = "data/pubfig_cloaked/")
