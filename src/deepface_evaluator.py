from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import random

# Evaluator class takes a dataset and produces embeddings for every item in the dataset. It contains methods for evaluating the performance of a set of facial recognition models on that dataset by comparing embedding distances.
class Evaluator:
    def __init__(self, dataset_path, models, dataset_size = 0.01, probe_path=None, num_probes=100, gallery_size=100):
        self.paths = []
        self.probe_paths = []
        self.num_probes = num_probes
        self.models = models
        self.gallery_size = gallery_size
        self.embed_map = {model: {} for model in self.models} # save an embedding map for each model
        
        # Load in the dataset and save the paths
        for path in os.listdir(dataset_path):
            self.paths.append(dataset_path + path)
        
        random.shuffle(self.paths)
        self.paths = self.paths[:int(dataset_size * len(self.paths))]
        print("Finished reading in", len(self.paths), "dataset paths")

        for path in os.listdir(probe_path):
            self.probe_paths.append(probe_path + path)
        print("Finished reading in", len(self.probe_paths), "paths")
        

        #self.paths = self.paths[:50]
        #self.cloaked_paths = self.cloaked_paths[:10]
    # Compute the L2 distance for two embeddings. Used to measure similarity of facial images
    def L2_dist(self, embed1, embed2):
        np_embed1 = np.array(embed1)
        np_embed2 = np.array(embed2)
        return np.linalg.norm(np_embed1 - np_embed2, ord=2)

    # Compute the embeddings of every image in the dataset, for use in comparing similarities
    def compute_embeddings(self):
        for model in self.models:
            for i in tqdm(range(len(self.paths))):

                # Calculate the embedding
                try:
                    out = DeepFace.represent(img_path = self.paths[i], model_name = model)
                    emb = out[0]["embedding"]
                    self.embed_map[model][self.paths[i]] = emb
                except Exception as e:
                    print("Error in compute_embeddings", e)
                    #self.embed_map[model][self.paths[i]] = None
        print("Finished computing", len(self.paths), "embeddings")

    def compute_single_embedding(self, path):
        embs = {}
        for model in self.models:
            try:
                embs[model] = self.embed_map[model][path]
                #print("Found pre-computed embedding for path", path)
                continue
            except:
                None
            try:
                out = DeepFace.represent(img_path = path, model_name = model)
                embs[model] = out[0]["embedding"]

                # Add it to the global map
                self.embed_map[model][path] = out[0]["embedding"]
                #print("Computed new embedding for path", path)
            except Exception as e:
                #print("Error in compute_single_embedding", e, "this image did not work")
                None
        return embs


    # Find the top-k most similar images to a given image. Currently only works for k=1
    def find_most_similar(self, path, k=1):
        similars = {}
        for model in self.models:
            min_dist = 10000000
            most_simiar = ""
            for query in self.embed_map[model].keys(): # for every image except the given one
                if path != query and self.embed_map[model][query] != None: # check to make sure it exists
                    this_similar= self.L2_dist(self.embed_map[model][path], self.embed_map[model][query])
                    if this_similar < min_dist:
                        min_dist = this_similar
                        most_similar = query
            similars[model] = most_similar
        return similars

    def find_most_similar_by_embed(self, path, gallery_size=100, k=1):
        similars = {}
        for model in self.models:
            min_dist = 1000000
            most_similar = ""
            
            # Make sure we're randomizing how we select queries - different gallery each time
            queries = self.paths[:]
            random.shuffle(queries)

            # If the embedding fails for whatever reason we cannot continue
            path_embed = self.compute_single_embedding(path)
            if path_embed == {}:
                return None

            while gallery_size > 0:
                idx = random.randint(0, len(queries)-1)
                query_path = queries[idx]
                
                # Check to make sure we don't match with the exact same image
                if path == query_path:
                    continue
                
                # Get the embedding for the current query
                query_emb = self.compute_single_embedding(query_path)
                
                # If query embedding is null, skip it
                if query_emb == {}:
                    continue

                # Compare this one with the gallery image and see if it is better
                this_similar = self.L2_dist(path_embed[model], query_emb[model])
                if this_similar < min_dist:
                    min_dist = this_similar
                    most_similar = query_path
                gallery_size -= 1
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
                continue

            # Look at the most similar item for each model, and see which ones are accurate
            for model, match in similars.items():
                match_name = os.path.basename(match)[:os.path.basename(match).find("_")]
                if path_name == match_name:
                    totals[model] += 1

        # Print the accuracy
        for model in totals:
            print(f"{model} Cloaked acc: %0.4f" % (totals[model] / float(itr)))

