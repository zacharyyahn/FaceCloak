import boto3
import sys
from dotenv import load_dotenv
import os
load_dotenv()


probe_path = "data/celeba/gallery/000291_1.jpg"
gallery_path = "data/celeba/probe/000618_0.jpg"
threshold = 0.0

# --- Initialize Rekognition client using credentials from environment ---
rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
)

# --- Read the images as bytes ---
with open(probe_path, "rb") as probe_file:
    probe_bytes = probe_file.read()

with open(gallery_path, "rb") as gallery_file:
    gallery_bytes = gallery_file.read()

# --- Compare faces ---
response = rekognition.compare_faces(
    SourceImage={"Bytes": probe_bytes},
    TargetImage={"Bytes": gallery_bytes},
    SimilarityThreshold=threshold,
)

# --- Parse and display results ---
matches = response.get("FaceMatches", [])
if not matches:
    print(f"No matches found above {threshold}% similarity.")

best_match = max(matches, key=lambda x: x["Similarity"])
similarity = best_match["Similarity"]

print(f"✅ Match found with similarity: {similarity:.2f}%")

if similarity >= threshold:
    print("Faces verified ✅")
else:
    print("Faces not verified ❌")

print(similarity)