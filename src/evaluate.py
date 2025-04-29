from evaluator import Evaluator
from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN, InceptionResnetV1
from dist_functions import cosine_dist, l2_dist
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs="*", type=str, help="List of feature extractors to use.")
parser.add_argument("--dataset_file", type=str, help="Path to dataset")
parser.add_argument("--probe_file", type=str, help="Path to cloaked dataset")
parser.add_argument("--num_probes", type=int, help="How many probe images to evaluate")
parser.add_argument("--attack", action='store_true', help="Whether to evaluate on cloaked images")
parser.add_argument("--num_to_evaluate", type=int, help="How many images to evaluate")
parser.add_argument("--dataset_size", type=float, help="Proportion of evaluation dataset to use")
parser.add_argument("--gallery_size", nargs="*", type=int, help="Number of images to compare to")
parser.add_argument("--cropped_im_size", type=int, default=112, help="Size of cropped image")
parser.add_argument("--verbosity", type=str, default="none", help="debugging verbosity")
parser.add_argument("--distance_function", type=str, default="l2", help="Distance function, either l2 or cosine")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

extractors_list = {
        "ArcFace":torch_arcface_insightface(device).to(device).eval(),
        "Facenet":InceptionResnetV1(pretrained='vggface2').to(device).eval()
}

distance_funcs = {
        "l2": l2_dist,
        "cosine":cosine_dist
        }

# Copy over the selected models to use for our evaluator
extractors = {}
for model in args.models:
    extractors[model] = extractors_list[model]

# Get MTCNN for crop and align
mtcnn = MTCNN(image_size=args.cropped_im_size, device=device).to(device)
dist_func = distance_funcs[args.distance_function]

evaluator = Evaluator(dataset_path=args.dataset_file, probe_path=args.probe_file, num_probes=args.num_probes, models=extractors, cropper=mtcnn, dataset_size=args.dataset_size, gallery_size=args.gallery_size, verbosity=args.verbosity, device=device, cropped_im_size=args.cropped_im_size, dist_func=dist_func)

for gallery_size in args.gallery_size:
    print("----------- GALLERY SIZE:",gallery_size,"-----------")
    gallery_size = int(gallery_size)
    evaluator.gallery_size = gallery_size
    evaluator.evaluate_all(num_images=args.num_probes)
