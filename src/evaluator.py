import os
import cv2
import torch
from utils import preprocess_tanh, reverse_tanh, preprocess_divide, reverse_divide, do_nothing
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random
import time

# Evaluator class takes a dataset and produces embeddings for every item in the dataset. It contains methods for evaluating the performance of a set of facial recognition models on that dataset by comparing embedding distances.
class Evaluator:
    def __init__(self, gallery_path, models, cropper, dataset_size = 0.01, probe_path=None, verbosity="none", device=None, cropped_im_size=112, dist_func=None):
        self.gallery_paths = []
        self.device = device
        self.probe_paths = []
        self.models = models
        self.cropped_im_size = cropped_im_size
        self.cropper = cropper
        self.verbosity = verbosity
        self.distance_func = dist_func
        self.embed_map = {model: {} for model in self.models} # save an embedding map for each model
        
        self.preprocessors = {
                "ArcFaceR18":preprocess_divide,
                "ArcFaceR34":preprocess_divide,
                "ArcFaceR50":preprocess_divide,
                "ArcFaceR100":preprocess_divide,
                "CosFaceR18":preprocess_divide,
                "CosFaceR34":preprocess_divide,
                "CosFaceR50":preprocess_divide,
                "CosFaceR100":preprocess_divide,
                "Facenet":preprocess_tanh
                }

        # Load in the dataset and save the paths
        for path in os.listdir(gallery_path):
            self.gallery_paths.append(gallery_path + path)
        
        random.shuffle(self.gallery_paths)
        self.gallery_paths = self.gallery_paths[:int(dataset_size * len(self.gallery_paths))]
        print("Finished reading in", len(self.gallery_paths), "gallery paths from", gallery_path)

        for path in os.listdir(probe_path):
            self.probe_paths.append(probe_path + path)
        print("Finished reading in", len(self.probe_paths), "probe paths from", probe_path)
    
    # Compute the L2 distance for two embeddings. Used to measure similarity of facial images
    # NOTE: If one model fails, all models fail in this implementation
    def compute_single_embedding(self, path):
        emb = {}
        for model in self.models.keys():
            try:
                emb[model] = self.embed_map[model][path]
                continue
            except Exception as e:
                if self.verbosity == "error": print("Did not find embedding for", path)

                # Read in the image and get the boxes from MTCNN
                im = cv2.imread(path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = torch.Tensor(im).to(self.device)

                with torch.no_grad():
                    boxes, _ = self.cropper.detect(im)

                # Check for null boxes. If a box is none, replace it with the entire image.
                if type(boxes) == type(None): 
                    boxes = [[0, 0, im.shape[1] - 1, im.shape[0] - 1]]

                # Make sure that the boxes do not exceed the image size
                for i in range(4):
                    boxes[0][i] = int(boxes[0][i]) if boxes[0][i] >= 0.0 else 0
                boxes[0][2] = boxes[0][2] if boxes[0][2] < im.shape[1] else im.shape[1] - 1
                boxes[0][3] = boxes[0][3] if boxes[0][3] < im.shape[0] else im.shape[0] - 1

                # Parse out the values
                xmin = int(boxes[0][0])
                ymin = int(boxes[0][1])
                xmax = int(boxes[0][2])
                ymax = int(boxes[0][3])

                # Crop the area with the bounding box
                crop = im[ymin:ymax, xmin:xmax, :].clone()

                # Resize to be 112x112
                try:
                    crop = torch.permute(crop, (2, 0, 1)).unsqueeze(0)
                    crop = F.interpolate(crop, size=(self.cropped_im_size, self.cropped_im_size), mode="bilinear", align_corners=False)
                except Exception as e:
                    if self.verbosity == "error": print("ERROR 3 in get_one_embed:", e, "Empty crop on boxes", boxes)
                    if self.verbosity == "error": print("REMOVING", path, "from dataset")
                    emb[model] = None
                    continue

                # Apply the normalization transform
                crop = self.preprocessors[model](crop)

                # Get the embeddings and save them
                with torch.no_grad():
                    embed = self.models[model](crop)
                    embed = embed / torch.norm(embed, p=2, dim=1, keepdim=True)
                    crop = crop.cpu().detach()

                #print("Finaly embed has shape", embeds.size())
                self.embed_map[model][path] = embed
                emb[model] = embed
                #print("Computed new embedding for", path)
        return emb


    # Find the top-k most similar images to a given image. #Randomly selects gallery images from the dataset. Currently only works for k=1
    def find_most_similar_by_embed(self, path, k=1):
        similars = {}
        for model in self.models.keys():
            min_dist = float('inf')
            most_similar = ""
            
            # If the embedding fails for whatever reason we skip to the next model
            path_embed = self.compute_single_embedding(path)
            if path_embed[model] == None:
                print("Couldn't find a path embedding for path", path)
                similars[model] = None
                continue

            for query_path in self.gallery_paths:                
                # Check to make sure we don't match with the exact same image
                if os.path.basename(path) == os.path.basename(query_path):
                    continue
                
                # Get the embedding for the current query
                query_emb = self.compute_single_embedding(query_path)
                
                # If query embedding is null, skip it
                if query_emb[model] == None:
                    continue

                # Compare this one with the gallery image and see if it is better
                this_similar = self.distance_func(path_embed[model], query_emb[model])
                #print("Model", model, "has similarity here", this_similar)
                if this_similar < min_dist:
                    min_dist = this_similar
                    most_similar = query_path
            similars[model] = most_similar
        return similars

    # For each selected image, find the most similar one. Mark if that is correct or incorrect. 
    def evaluate(self, num_images):
        totals = {model: 0 for model in self.models}
        
        for i in tqdm(range(num_images)):
            
            # Select a random image 
            idx = random.randint(0, len(self.gallery_paths)-1)
            
            # Get the path at this index
            this_path = self.gallery_paths[idx]
            
            # Skip if this is a null image
            if self.embed_map[self.models[0]][this_path] is None: # Skip if we have a null image with no face
                continue
            
            # Parse out the class name from the path
            path_name = os.path.basename(this_path)[:os.path.basename(this_path).find("_")]
            
            # Find the most similar image
            similars = self.find_most_similar(this_path, k=1)
            
            # If we couldn't find an embedding for the current face, we'll have to skip it
            if similars == None:
                continue
            
            # For each model, see if the classes match
            for model, match in similars.items():
                
                # Handle if we could not find a match or an embedding
                if match == None:
                    continue

                match_name = os.path.basename(match)[:os.path.basename(match).find("_")]
                #print("Query", path_name, "matched with", match_name) 
                if path_name == match_name:
                    totals[model] += 1
        
        for model in totals:
            print(f"{model} acc: %0.4f" % (totals[model] / float(num_images)))

    def evaluate_all(self):
        totals = {model: 0 for model in self.models}

        # Iterate through each item in the dataset and evaluate them
        print("Evaluating on", len(self.probe_paths), "cloaked images....")
        itr = 0
        num_skipped = 0 # make sure we don't count failed images in our denominator for metric calculation.
        for i in tqdm(range(len(self.probe_paths))):
            itr += 1
            this_path = self.probe_paths[i]
            # Track how many we have looked at at stop if we excede the desired number

            # Extract the base name from the path
            path_name = os.path.basename(this_path)[:os.path.basename(this_path).find("_")]

            # Find the similar items
            similars = self.find_most_similar_by_embed(this_path)

            # If we couldn't find a face in the given probe image, we need to skip
            if similars == None:
                #print("Similars were none, skipping...")
                num_skipped += 1
                continue

            # Look at the most similar item for each model, and see which ones are accurate
            for model, match in similars.items():
                if match == None:
                    #print("None match for model", model)
                    num_skipped += 1
                    continue
                match_name = os.path.basename(match)[:os.path.basename(match).find("_")]
                print("Path name:", path_name, "Match_name:", match_name)
                #print("Currently looking at model", model, "match", match)
                if path_name == match_name:
                    #print("Increasing score for model", model)
                    totals[model] += 1

        # Print the accuracy
        outs = {}
        for model in totals:
            print(f"{model} Cloaked acc: %0.4f" % (totals[model] / float(itr - num_skipped)))
            outs[model] = totals[model] / float(itr - num_skipped)
        return outs

