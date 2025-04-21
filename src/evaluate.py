from evaluator import Evaluator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--models", nargs="*", type=str, help="List of feature extractors to use.")
parser.add_argument("--dataset_file", type=str, help="Path to dataset")
args = parser.parse_args()

evaluator = Evaluator(args.dataset_file, args.models)
evaluator.compute_embeddings()
evaluator.evaluate(num_images=500)
