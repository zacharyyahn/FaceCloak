from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from cloaker import Cloaker
from cloak_functions import pgd_cloak, sgd_cloak 
from loss_functions import fawkes_loss, triplet_loss
from dist_functions import cosine_dist, l2_dist
from utils import preprocess_tanh, reverse_tanh, preprocess_divide, reverse_divide

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
parser.add_argument("--extractor_type", type=str, default="none", help="The model to use for extracting embeddings")
parser.add_argument("--distance_function", type=str, default="L2", help="Distance function to use when comparing embedding vectors")
parser.add_argument("--norm_function", type=str, default="tanh", help="Normalization function for preparing images")
parser.add_argument("--cloak_function", type=str, default="pgd_cloak", help="The optimization to use for cloaking images")
parser.add_argument("--cloak_loss", type=str, default="fawkes", help="The loss function to use for cloaking images.")
parser.add_argument("--cloak_function_iters", type=int, default=10, help="Number of iterations to run the optimization")
parser.add_argument("--cloak_function_step", type=float, default=1./255, help="Step size for optimization methods with step sizes")
parser.add_argument("--cloak_function_max_pert", type=float, default=1./255, help="Maximum perturbation for optimization methods with max perturbations")
parser.add_argument("--cloak_function_lr", type=float, default=0.5, help="Learning rate for optimization methods with learning rates")
args = parser.parse_args()

#assert args.num_dataset_images > args.num_cloaked_images
#assert args.num_dataset_images > args.target_pool_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

extractors = {
        "ArcFace":torch_arcface_insightface(device).to(device),
        "Facenet":InceptionResnetV1(pretrained='vggface2').to(device)
    }

cloak_funcs = {
        "pgd_cloak":pgd_cloak,
        "sgd_cloak":sgd_cloak,
    }

cloak_losses = {
        "fawkes":fawkes_loss,
        "triplet":triplet_loss,
    }

distance_funcs = {
        "cosine":cosine_dist,
        "l2":l2_dist
    }

norm_funcs = {
        "tanh":preprocess_tanh,
        "divide":preprocess_divide
        }

reverse_norm_funcs = {
        "tanh":reverse_tanh,
        "divide":reverse_divide
        }

assert args.extractor_type in extractors.keys()

# Get the extractor, cloaking optimization function, loss function
extractor = extractors[args.extractor_type]
cloak_function = cloak_funcs[args.cloak_function]
loss_function = cloak_losses[args.cloak_loss]
distance_function = distance_funcs[args.distance_function]
norm_function = norm_funcs[args.norm_function]
reverse_norm_function = reverse_norm_funcs[args.norm_function]

# Get MTCNN for crop and align
mtcnn = MTCNN(image_size=args.cropped_im_size, device=device).to(device)

print("Extracting with model:", args.extractor_type)

# Initialize cloaker
cloaker = Cloaker(
        dataset_path=args.dataset_path, 
        extractor=extractor, 
        cropper=mtcnn, 
        batch_size=args.batch_size, 
        cropped_im_size=args.cropped_im_size, 
        target_pool_size=args.target_pool_size, 
        num_dataset_images=args.num_dataset_images, 
        device=device, 
        verbosity=args.verbosity,
        distance_function=distance_function,
        cloak_function=cloak_function,
        cloak_loss=loss_function,
        cloak_function_iters=args.cloak_function_iters,
        cloak_function_step=args.cloak_function_step,
        cloak_function_max_pert=args.cloak_function_max_pert,
        cloak_function_lr=args.cloak_function_lr,
        loss_func_select=args.cloak_loss,
        norm_function = norm_function,
        reverse_norm_function = reverse_norm_function
        )

# Cloak images according to args
cloaker.cloak_all(save_dir = args.cloak_save_path, num_images=args.num_cloaked_images)
