#from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from cloaker import Cloaker
from cloak_functions import pgd_cloak, sgd_cloak, afog_cloak, afog_cloak_multi, minmax_cloak, pgd_cloak_multi
from loss_functions import fawkes_loss, triplet_loss, dssim_loss, lpips_loss, mse_loss, untarget_loss
from dist_functions import cosine_dist, l2_dist
from advcloak.model_irse import IR_50
from utils import preprocess_tanh, reverse_tanh, preprocess_divide, reverse_divide
from insightface_code.recognition.arcface_torch.backbones import get_model

parser = argparse.ArgumentParser()
parser.add_argument("--probe_dataset_path", type=str, help="Path to probe images")
parser.add_argument("--gallery_dataset_path", type=str, help="Path to gallery images")
parser.add_argument("--extractor_path", type=str, default=None, help="Path to extractor")
parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
parser.add_argument("--cropped_im_size", type=int, default=112, help="Crop size to apply MTCNN")
parser.add_argument("--num_dataset_images", type=float, default=1.0, help="Number of images to take from dataset")
parser.add_argument("--cloak_save_path", type=str, default="data/pubfig_cloaked", help="Path to save cloaked images")
parser.add_argument("--verbosity", type=str, default="none", help="Error verbosity level")
parser.add_argument("--extractor_type", type=str, default="Facenet", help="The model to use for extracting embeddings")
parser.add_argument("--distance_function", type=str, default="l2", help="Distance function to use when comparing embedding vectors")
parser.add_argument("--norm_function", type=str, default="tanh", help="Normalization function for preparing images")
parser.add_argument("--cloak_function", type=str, default="pgd_cloak", help="The optimization to use for cloaking images")
parser.add_argument("--multi_cloak_function", type=str, default="pgd_cloak", help="The optimization for using multi-cloak on the images")
parser.add_argument("--do_stickers", type=int, default=False, help="Whether to add region stickers to the perturbation optimization")
parser.add_argument("--do_highpass", type=int, default=False, help="Whether to add highpass to the perturbation optimization")
parser.add_argument("--cloak_loss", type=str, default="fawkes", help="The loss function to use for cloaking images.")
parser.add_argument("--multi_cloak_loss", type=str, default="fawkes", help="The loss function to use for multi-cloaking images.")
parser.add_argument("--cloak_function_iters", type=int, default=10, help="Number of iterations to run the optimization")
parser.add_argument("--multi_cloak_function_iters", type=int, default=10, help="Number of iterations for multi-cloak optimization")
parser.add_argument("--cloak_function_step", type=float, default=1., help="Step size for optimization methods with step sizes")
parser.add_argument("--cloak_function_max_pert", type=float, default=1., help="Maximum perturbation for optimization methods with max perturbations")
parser.add_argument("--multi_cloak_function_max_pert", type=float, default=1., help="Maximum perturbation for multicloak pretraining")
parser.add_argument("--cloak_function_lr", type=float, default=0.5, help="Learning rate for optimization methods with learning rates")
parser.add_argument("--cloak_percep_loss", type=str, default="none", help="Perceptual loss to use in loss calculation. If none, only clipping will be used")
parser.add_argument("--percep_loss_weight", type=str, default=0.0, help="Weighted factor for adding perceptual loss to cloak loss")
parser.add_argument("--mode", type=str, default="perturb", help="Whether to use perturb mode, multi mode, or makeup mode")
parser.add_argument("--makeup_mode", type=str, default="diffam", help="Which makeup method to use, either diffam or amt-gan")
parser.add_argument("--gen_save_path", type=str, default="/", help="Path to save generated images, if using")
parser.add_argument("--num_gen_iterations", type=int, default=5, help="Gen iterations when using minmax")
parser.add_argument("--gen_learning_rate", type=float, default=0.1, help="Default learning rate for generating images")
parser.add_argument("--num_images_to_gen", type=int, default=4)
parser.add_argument("--n_to_eval", type=int, default=1, help="Top-n to images to eval over")
args = parser.parse_args()

#assert args.num_dataset_images > args.num_cloaked_images
#assert args.num_dataset_images > args.target_pool_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.mode == "multi":
      assert args.multi_cloak_function in ["pgd_cloak_multi", "afog_cloak_multi", "pgd_cloak_sticker", "afog_cloak_sticker"], "Multi mode is only compatible with multi cloak functions."

args.cloak_function_step = args.cloak_function_step / 255.
args.cloak_function_max_pert = args.cloak_function_max_pert / 255.
args.multi_cloak_function_max_pert = args.multi_cloak_function_max_pert / 255.

# model_shorthands = {
#         "ArcFaceR18":"r18",
#         "ArcFaceR34":"r34",
#         "ArcFaceR50":"r50",
#         "ArcFaceR100":"r100",
#         "CosFaceR18":"r18",
#         "CosFaceR34":"r34",
#         "CosFaceR50":"r50",
#         "CosFaceR100":"r100",
#         }

# model_paths = {
#         "ArcFaceR18":"model_checkpoints/arcface_r18_ms1mv3.pth",
#         "ArcFaceR34":"model_checkpoints/arcface_r34_ms1mv3.pth",
#         "ArcFaceR50":"model_checkpoints/arcface_r50_ms1mv3.pth",
#         "ArcFaceR100":"model_checkpoints/arcface_r100_ms1mv3.pth",
#         "CosFaceR18":"model_checkpoints/cosface_r18_glint360k.pth",
#         "CosFaceR34":"model_checkpoints/cosface_r34_glint360k.pth",
#         "CosFaceR50":"model_checkpoints/cosface_r50_glint360k.pth",
#         "CosFaceR100":"model_checkpoints/cosface_r100_glint360k.pth",
#         }

# Note: a function below adds CosFace and ArcFace models to this dictionary
extractors = {
        #"ArcFace":torch_arcface_insightface(device).to(device),
        "Facenet":InceptionResnetV1(pretrained='vggface2').to(device)
    }

cloak_funcs = {
        "pgd_cloak":pgd_cloak,
        "minmax_cloak":minmax_cloak,
        "pgd_cloak_multi":pgd_cloak_multi,
        "sgd_cloak":sgd_cloak,
        "afog_cloak":afog_cloak,
        "afog_cloak_multi":afog_cloak_multi,
        "none":None
    }

cloak_losses = {
        "fawkes":fawkes_loss,
        "triplet":triplet_loss,
        "untarget":untarget_loss,
        "none":None
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

percep_losses = {
        "dssim": dssim_loss,
        "lpips": lpips_loss,
        "mse": mse_loss,
        "none": None
        }

# Handle loading in the weights of an insightface model (arcface or cosface)
# def load_arcface_cosface_model(model):
#     if_model = get_model(model_shorthands[model], fp16=False)
#     if_model.load_state_dict(torch.load(model_paths[model], map_location=device))
#     if_model.eval().to(device)
#     extractors[model] = if_model

# Get the extractor, cloaking optimization function, loss function. If we're using ArcFace or CosFace, normalize accordingly
# if "ArcFace" in args.extractor_type or "CosFace" in args.extractor_type:
#     load_arcface_cosface_model(args.extractor_type)
#     norm_function = preprocess_divide
#     reverse_norm_function = reverse_divide

cloak_function = cloak_funcs[args.cloak_function]
multi_cloak_function = cloak_funcs[args.multi_cloak_function]
loss_function = cloak_losses[args.cloak_loss]
multi_loss_function = cloak_losses[args.multi_cloak_loss]
distance_function = distance_funcs[args.distance_function]
norm_function = norm_funcs[args.norm_function]
reverse_norm_function = reverse_norm_funcs[args.norm_function]
percep_loss_function = percep_losses[args.cloak_percep_loss]


# Get MTCNN for crop and align
mtcnn = MTCNN(image_size=args.cropped_im_size, device=device).to(device)

print("Extracting with model:", args.extractor_type)

# Initialize cloaker
cloaker = Cloaker(
        probe_dataset_path=args.probe_dataset_path, 
        gallery_dataset_path=args.gallery_dataset_path,
        extractor=args.extractor_type,
        cropper=mtcnn, 
        batch_size=args.batch_size, 
        cropped_im_size=args.cropped_im_size, 
        device=device, 
        verbosity=args.verbosity,
        distance_function=distance_function,
        cloak_function=cloak_function,
        multi_cloak_function = multi_cloak_function,
        do_stickers = int(args.do_stickers) == 1,
        do_highpass = int(args.do_highpass) == 1,
        cloak_loss=loss_function,
        multi_cloak_loss=multi_loss_function,
        cloak_function_iters=args.cloak_function_iters,
        multi_cloak_function_iters=args.multi_cloak_function_iters,
        cloak_function_step=args.cloak_function_step,
        cloak_function_max_pert=args.cloak_function_max_pert,
        multi_cloak_function_max_pert=args.multi_cloak_function_max_pert,
        cloak_function_lr=args.cloak_function_lr,
        loss_func_select=args.cloak_loss,
        multi_loss_func_select=args.multi_cloak_loss,
        norm_function = norm_function,
        reverse_norm_function = reverse_norm_function,
        percep_loss = percep_loss_function,
        percep_loss_weight = args.percep_loss_weight,
        num_gen_iterations = args.num_gen_iterations,
        gen_learning_rate = args.gen_learning_rate,
        num_images_to_generate=args.num_images_to_gen,
        n_to_eval=args.n_to_eval,
        mode=args.mode,
        )

paths = [
        "data/privacy_celeb/probe/467440_14832175.jpg",
        "data/privacy_celeb/probe/498299_15809804.jpg",
        "data/privacy_celeb/train/296813_9413123.jpg",
        "data/privacy_celeb/train/157494_4995450.jpg",
        "data/privacy_celeb/train/619592_19655786.jpg",
        "data/privacy_celeb/train/211164_6710801.jpg"
        ]


# Cloak images according to args
if args.mode == "perturb":
        cloaker.cloak_all(save_dir = args.cloak_save_path, do_paths=paths)
elif args.mode == "multi" or args.mode == "multi_finetune":
        cloaker.cloak_all_multi(save_dir = args.cloak_save_path, gen_save_path = args.gen_save_path)#, do_paths=paths)
elif args.mode == "minmax" or args.mode == "multi_minmax":
        cloaker.cloak_all_minmax(save_dir = args.cloak_save_path, gen_save_path = args.gen_save_path)
else:
     cloaker.makeup_all(save_dir = args.cloak_save_path, makeup_mode=args.makeup_mode)
