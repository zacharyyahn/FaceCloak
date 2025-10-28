from evaluator import Evaluator
#from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface_code.recognition.arcface_torch.backbones import get_model
from dist_functions import cosine_dist, l2_dist
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs="*", type=str, help="List of feature extractors to use.")
parser.add_argument("--gallery_file", type=str, help="Path to dataset")
parser.add_argument("--probe_file", type=str, help="Path to cloaked dataset")
parser.add_argument("--num_probes", type=int, help="How many probe images to evaluate")
parser.add_argument("--attack", action='store_true', help="Whether to evaluate on cloaked images")
parser.add_argument("--num_to_evaluate", type=int, help="How many images to evaluate")
parser.add_argument("--dataset_size", type=float, help="Proportion of evaluation dataset to use")
parser.add_argument("--cropped_im_size", type=int, default=112, help="Size of cropped image")
parser.add_argument("--verbosity", type=str, default="none", help="debugging verbosity")
parser.add_argument("--distance_function", type=str, default="l2", help="Distance function, either l2 or cosine")
args = parser.parse_args()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_shorthands = {
        "ArcFaceR18":"r18",
        "ArcFaceR34":"r34",
        "ArcFaceR50":"r50",
        "ArcFaceR100":"r100",
        "CosFaceR18":"r18",
        "CosFaceR34":"r34",
        "CosFaceR50":"r50",
        "CosFaceR100":"r100",
        }

model_paths = {
        "ArcFaceR18":"model_checkpoints/arcface_r18_ms1mv3.pth",
        "ArcFaceR34":"model_checkpoints/arcface_r34_ms1mv3.pth",
        "ArcFaceR50":"model_checkpoints/arcface_r50_ms1mv3.pth",
        "ArcFaceR100":"model_checkpoints/arcface_r100_ms1mv3.pth",
        "CosFaceR18":"model_checkpoints/cosface_r18_glint360k.pth",
        "CosFaceR34":"model_checkpoints/cosface_r34_glint360k.pth",
        "CosFaceR50":"model_checkpoints/cosface_r50_glint360k.pth",
        "CosFaceR100":"model_checkpoints/cosface_r100_glint360k.pth",
        }

extractors_list = {
        "Facenet":InceptionResnetV1(pretrained='vggface2').to(device).eval()
}

# Handle loading in the weights of an insightface model (arcface or cosface)
def load_arcface_cosface_model(model):
    if_model = get_model(model_shorthands[model], fp16=False)
    if_model.load_state_dict(torch.load(model_paths[model]))
    if_model.eval().to(device)
    extractors_list[model] = if_model

distance_funcs = {
        "l2": l2_dist,
        "cosine":cosine_dist
        }

# Copy over the selected models to use for our evaluator
extractors = {}
for model in args.models:
    # Handle loading in models via insightface
    if "ArcFace" in model or "CosFace" in model:
        load_arcface_cosface_model(model)
    extractors[model] = extractors_list[model]

# Get MTCNN for crop and align
mtcnn = MTCNN(image_size=args.cropped_im_size, device=device).to(device).eval()
dist_func = distance_funcs[args.distance_function]

evaluator = Evaluator(gallery_path=args.gallery_file, probe_path=args.probe_file, models=extractors, cropper=mtcnn, dataset_size=args.dataset_size, verbosity=args.verbosity, device=device, cropped_im_size=args.cropped_im_size, dist_func=dist_func)

outs = {model: [] for model in args.models}
this_out = evaluator.evaluate_all()
for model, val in this_out.items():
    outs[model].append(val)
for model, l in outs.items():
    print(model, "accs:", l)
