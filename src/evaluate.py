from evaluator import Evaluator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs="*", type=str, help="List of feature extractors to use.")
parser.add_argument("--dataset_file", type=str, help="Path to dataset")
parser.add_argument("--cloaked_file", type=str, help="Path to cloaked dataset")
parser.add_argument("--attack", action='store_true', help="Whether to evaluate on cloaked images")
args = parser.parse_args()

evaluator = Evaluator(dataset_path=args.dataset_file, cloaked_path=None, models=args.models)
evaluator.compute_embeddings()
if not args.attack:
    evaluator.evaluate(num_images=500)
else:
    evaluate.evaluate_cloaked(num_images=500)
