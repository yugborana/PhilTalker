from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Payload
import uuid

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)
collection_name = "philtalker"

# Try to create collection if it doesn't exist
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    print(f"Created new collection: {collection_name}")
except Exception as e:
    print(f"Collection may already exist: {e}")

# Initialize CLIP model and processor (more VRAM-efficient than ColIdefics)
model_name = "openai/clip-vit-base-patch32"  # Smaller footprint for 8GB VRAM
model = CLIPModel.from_pretrained(model_name).to("cuda")
processor = CLIPProcessor.from_pretrained(model_name)

# Set the model to evaluation mode
model = model.eval()

def generate_image_embeddings(images, model, processor, batch_size=2):
    """Generate embeddings for a list of images using the CLIP model"""
    embeddings = []
    
    # Create simple dataloader without custom collate_fn
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        
        # Process images with the CLIP processor
        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
            image_features = model.get_image_features(**inputs)
            
            # Normalize embeddings (important for cosine similarity)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Move to CPU and convert to numpy
            batch_embeddings = image_features.cpu().numpy()
            embeddings.extend(batch_embeddings)
            
    return embeddings

def load_images_from_folder(folder_path, max_height=800):
    """Load and resize images from a folder"""
    images = []
    image_paths = []
    
    image_files = sorted([f for f in os.listdir(folder_path) 
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Resize if needed
            width, height = image.size
            if height > max_height:
                ratio = max_height / height
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height))
                
            images.append(image)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return images, image_paths

# Load images
image_folder = "data/images"
images, image_paths = load_images_from_folder(image_folder)
print(f"Loaded {len(images)} images from {image_folder}")

# Generate embeddings
print("Generating embeddings...")
embeddings = generate_image_embeddings(images, model, processor)
print(f"Generated {len(embeddings)} embeddings")

# Create points to upload to Qdrant
points = []
for idx, (path, embedding) in enumerate(zip(image_paths, embeddings)):
    filename = os.path.basename(path)
    
    # Create a point
    point = PointStruct(
        id=str(uuid.uuid4()),  # Generate unique ID
        vector=embedding.tolist(),  # Full float embedding (more accurate for search)
        payload={
            "filename": filename,
            "path": path,
            "page_number": idx  # Could be replaced with actual page number if available
        }
    )
    points.append(point)

# Upload points to Qdrant in batches
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    client.upsert(
        collection_name=collection_name,
        points=batch
    )
    print(f"Inserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

print(f"Successfully inserted {len(points)} image embeddings into Qdrant collection '{collection_name}'")

# Sample code for querying (for reference)
def query_similar_images(text_query, top_k=5):
    """Search for images similar to a text query"""
    # Process text query
    inputs = processor(text=text_query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        query_vector = text_features.cpu().numpy()[0].tolist()
    
    # Search in Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    
    return search_result
