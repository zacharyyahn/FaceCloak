from evaluator import Evaluator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs="*", type=str, help="List of feature extractors to use.")
parser.add_argument("--dataset_file", type=str, help="Path to dataset")
parser.add_argument("--cloaked_file", type=str, help="Path to cloaked dataset")
parser.add_argument("--attack", action='store_true', help="Whether to evaluate on cloaked images")
parser.add_argument("--num_to_evaluate", type=int, help="How many images to evaluate")
parser.add_argument("--dataset_size", type=float, help="Proportion of evaluation dataset to use")
args = parser.parse_args()

evaluator = Evaluator(dataset_path=args.dataset_file, cloaked_path=args.cloaked_file, models=args.models, dataset_size=args.dataset_size)
evaluator.compute_embeddings()
if not args.attack:
    evaluator.evaluate(num_images=args.num_to_evaluate)
else:
    evaluator.evaluate_cloaked(num_images=args.num_to_evaluate)
