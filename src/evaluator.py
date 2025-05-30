import os
import cv2
import torch
from utils import preprocess_tanh, reverse_tanh, preprocess_divide, reverse_divide, do_nothing
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random

# Evaluator class takes a dataset and produces embeddings for every item in the dataset. It contains methods for evaluating the performance of a set of facial recognition models on that dataset by comparing embedding distances.
class Evaluator:
    def __init__(self, dataset_path, models, cropper, dataset_size = 0.01, probe_path=None, gallery_size=100, verbosity="none", device=None, cropped_im_size=112, dist_func=None):
        self.paths = []
        self.device = device
        self.probe_paths = []
        self.models = models
        self.cropped_im_size = cropped_im_size
        self.cropper = cropper
        self.gallery_size = gallery_size
        self.verbosity = verbosity
        self.distance_func = dist_func
        self.embed_map = {model: {} for model in self.models} # save an embedding map for each model
        
        self.preprocessors = {
                "ArcFaceR18":preprocess_tanh,
                "ArcFaceR34":preprocess_tanh,
                "ArcFaceR50":preprocess_tanh,
                "ArcFaceR100":preprocess_tanh,
                "CosFaceR18":preprocess_tanh,
                "CosFaceR34":preprocess_tanh,
                "CosFaceR50":preprocess_tanh,
                "CosFaceR100":preprocess_tanh,
                "Facenet":preprocess_tanh
                }

        # Load in the dataset and save the paths
        for path in os.listdir(dataset_path):
            self.paths.append(dataset_path + path)
        
        random.shuffle(self.paths)
        self.paths = self.paths[:int(dataset_size * len(self.paths))]
        print("Finished reading in", len(self.paths), "gallery paths from", dataset_path)

        for path in os.listdir(probe_path):
            self.probe_paths.append(probe_path + path)
        print("Finished reading in", len(self.probe_paths), "probe paths from", probe_path)

        #self.paths = self.paths[:50]
        #self.cloaked_paths = self.cloaked_paths[:10]
    
    # Compute the L2 distance for two embeddings. Used to measure similarity of facial images
    # NOTE: If one model fails, all models fail in this implementation
    def compute_single_embedding(self, path):
        emb = {}
        for model in self.models.keys():
            try:
                emb[model] = self.embed_map[model][path]
                #print("Found premade embed for", path)
                continue
            except Exception as e:
                if self.verbosity == "log": print("Did not find embedding for", path)

                # Read in the image and get the boxes from MTCNN
                im = cv2.imread(path)
                try:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    if self.verbosity == "error": print("ERROR in compute_single_embedding:", e)           
                    emb[model] = None
                    continue
                im = torch.Tensor(im).to(self.device)

                with torch.no_grad():
                    boxes, _ = self.cropper.detect(im)
                im = im.cpu().detach().numpy()

                # See if there are boxes. If not, or more than one, then we need to return
                try:
                    if not boxes:
                        if self.verbosity == "error": print("ERROR 1 in get_one_embed: Boxes are None")
                        if self.verbosity == "error": print("REMOVING", path, "from dataset")
                        #self.paths.remove(path)
                        emb[model] = None
                        continue
                except:
                    None

                if len(boxes) != 1:
                    if self.verbosity == "error": print("ERROR 2 in get_one_embed: Boxes are", boxes)
                    if self.verbosity == "error": print("REMOVING", path, "from dataset")
                    #self.paths.remove(path)
                    emb[model] = None
                    continue

                xmin = int(boxes[0][0])
                ymin = int(boxes[0][1])
                xmax = int(boxes[0][2])
                ymax = int(boxes[0][3])

                # Crop the area with the bounding box
                crop = im[ymin:ymax, xmin:xmax, :].copy()

                # Resize to be 112x112
            
                try:
                    crop = torch.Tensor(cv2.resize(crop, (self.cropped_im_size, self.cropped_im_size), interpolation=cv2.INTER_CUBIC))
                    crop = torch.permute(crop, (2, 0, 1))
                except Exception as e:
                    if self.verbosity == "error": print("ERROR 3 in get_one_embed:", e, "Empty crop on boxes", boxes)
                    if self.verbosity == "error": print("REMOVING", path, "from dataset")
                    #self.paths.remove(path)
                    emb[model] = None
                    continue

                # Apply the normalization transform
                crop = self.preprocessors[model](crop)
                #print("After transform, crop has range", torch.min(crop), torch.max(crop))

                # Add fake batching to work with mozuma
                crop = crop.repeat((2, 1, 1, 1))

                # Get the embeddings and save them
                crop = crop.to(self.device)
                with torch.no_grad():
                    embed = self.models[model](crop)
                    crop = crop.cpu().detach()

                #print("Finaly embed has shape", embeds.size())
                self.embed_map[model][path] = embed
                emb[model] = embed
                #print("Computed new embedding for", path)
        return emb


    # Find the top-k most similar images to a given image. #Randomly selects gallery images from the dataset. Currently only works for k=1
    def find_most_similar_by_embed(self, path, gallery_size=100, k=1):
        similars = {}
        for model in self.models.keys():
            gal_size = gallery_size
            min_dist = 1000000
            most_similar = ""
            num_seen = 0
            
            # Make sure we're randomizing how we select queries - different gallery each time
            queries = self.paths[:]
            random.shuffle(queries)

            # If the embedding fails for whatever reason we skip to the next model
            path_embed = self.compute_single_embedding(path)
            if path_embed[model] == None:
                #print("Couldn't find a path embedding for path", path)
                similars[model] = None
                continue


            while num_seen < gal_size:
                #idx = random.randint(0, len(queries)-1)
                query_path = queries[num_seen] # we no longer want to randomly select since we want to compare to the whole gallery
                
                # Check to make sure we don't match with the exact same image
                if path == query_path:
                    num_seen += 1
                    continue
                
                # Get the embedding for the current query
                query_emb = self.compute_single_embedding(query_path)
                
                # If query embedding is null, skip it
                if query_emb[model] == None:
                    num_seen += 1
                    continue

                #print("Model", model, "has non-none query_emb")

                # Compare this one with the gallery image and see if it is better
                this_similar = self.distance_func(path_embed[model], query_emb[model])
                #print("Model", model, "has similarity here", this_similar)
                if this_similar < min_dist:
                    min_dist = this_similar
                    most_similar = query_path
                num_seen += 1
            print("Saw", num_seen, "total images during evaluation.")
            similars[model] = most_similar
        return similars

    # For each selected image, find the most similar one. Mark if that is correct or incorrect. 
    def evaluate(self, num_images):
        totals = {model: 0 for model in self.models}
        
        print("Evaluating on", num_images, "images...")
        for i in tqdm(range(num_images)):
            
            # Select a random image 
            idx = random.randint(0, len(self.paths)-1)
            
            # Get the path at this index
            this_path = self.paths[idx]
            
            # Skip if this is a null image
            if self.embed_map[self.models[0]][this_path] is None: # Skip if we have a null image with no face
                continue
            
            # Parse out the class name from the path
            path_name = os.path.basename(this_path)[:os.path.basename(this_path).find("_")]
            
            # Find the most similar image
            similars = self.find_most_similar(this_path, k=1, gallery_size=self.gallery_size)
            
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

    def evaluate_all(self, num_images):
        totals = {model: 0 for model in self.models}

        # Iterate through each item in the dataset and evaluate them
        print("Evaluating on", num_images, "cloaked images....")
        itr = 0
        for i in tqdm(range(num_images)):
            this_path = self.probe_paths[i]
            # Track how many we have looked at at stop if we excede the desired number
            itr += 1
            if itr > num_images:
                break

            # Extract the base name from the path
            path_name = os.path.basename(this_path)[:os.path.basename(this_path).find("_")]

            # Find the similar items
            similars = self.find_most_similar_by_embed(this_path, gallery_size=self.gallery_size)
            # If we couldn't find a face in the given probe image, we need to skip
            if similars == None:
                #print("Similars were none, skipping...")
                continue

            # Look at the most similar item for each model, and see which ones are accurate
            for model, match in similars.items():
                if match == None:
                    #print("None match for model", model)
                    continue
                match_name = os.path.basename(match)[:os.path.basename(match).find("_")]
                #print("Currently looking at model", model, "match", match)
                if path_name == match_name:
                    #print("Increasing score for model", model)
                    totals[model] += 1

        # Print the accuracy
        outs = {}
        for model in totals:
            print(f"{model} Cloaked acc: %0.4f" % (totals[model] / float(itr)))
            outs[model] = totals[model] / float(itr)
        return outs

