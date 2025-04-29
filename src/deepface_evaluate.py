from deepface_evaluator import Evaluator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs="*", type=str, help="List of feature extractors to use.")
parser.add_argument("--dataset_file", type=str, help="Path to dataset")
parser.add_argument("--probe_file", type=str, help="Path to cloaked dataset")
parser.add_argument("--num_probes", type=int, help="How many probe images to evaluate")
parser.add_argument("--attack", action='store_true', help="Whether to evaluate on cloaked images")
parser.add_argument("--num_to_evaluate", type=int, help="How many images to evaluate")
parser.add_argument("--dataset_size", type=float, help="Proportion of evaluation dataset to use")
parser.add_argument("--gallery_size", nargs="*", type=int, help="Number of images to compare to")
args = parser.parse_args()

evaluator = Evaluator(dataset_path=args.dataset_file, probe_path=args.probe_file, num_probes=args.num_probes, models=args.models, dataset_size=args.dataset_size, gallery_size=args.gallery_size)
#evaluator.compute_embeddings()

for gallery_size in args.gallery_size:
    print("----------- GALLERY SIZE:",gallery_size,"-----------")
    gallery_size = int(gallery_size)
    evaluator.gallery_size = gallery_size
    evaluator.evaluate_all(num_images=args.num_probes)
