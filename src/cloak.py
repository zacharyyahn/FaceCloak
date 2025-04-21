from mozuma.models.arcface.pretrained import torch_arcface_insightface
from facenet_pytorch import MTCNN
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to clean images")
parser.add_argument("--extractor_path", type=str, help="Path to extractor")

