from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import torch
import open_clip
import os
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import uuid
import uvicorn
from config import settings
# Initialize FastAPI app
app = FastAPI(title="Multimodal RAG Search API")

# Global variables
device = None
model = None
preprocess = None
tokenizer = None
chroma_client = None
collection = None

# Paths
UPLOAD_DIR = settings.upload_dir
DB_PATH = settings.chroma_db_path
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Pydantic models
class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    id: str
    score: float
    url: str
    title: str
    metadata: dict

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    global device, model, preprocess, tokenizer, chroma_client, collection
    
    print(f"Service intialized")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model name and pretrained model config
    model_name = settings.vision_model_name,
    pretrained_model = settings.pretrained_model

    # Load CLIP model
    print(f"Loading CLIP model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained_model
    )
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded")
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    image_loader = ImageLoader()
    embedding_function = OpenCLIPEmbeddingFunction()
    
    collection = chroma_client.get_or_create_collection(
        "pinteresty_collection",
        embedding_function=embedding_function,
        data_loader=image_loader,
    )
    print(f"ChromaDB initialized. Collection size: {collection.count()}")


@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG API",
        "endpoints": {
            "text_search": "/search/text",
            "image_search": "/search/image",
            "collection_info": "/collection/info",
            "docs": "/docs"
        }
    }


@app.get("/collection/info")
async def collection_info():
    """Get collection information"""
    count = collection.count()
    return {
        "collection_name": "pinteresty_collection",
        "total_items": count,
        "status": "active" if count > 0 else "empty"
    }


def encode_text(text: str):
    """Encode text to embedding"""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    return embedding.cpu().numpy()[0].tolist()


def encode_image(image_path: str):
    """Encode image to embedding"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(img_tensor)
    return embedding.cpu().numpy()[0].tolist()


@app.post("/search/text")
async def search_text(request: TextSearchRequest):
    """Search using text query"""
    try:
        # Generate text embedding
        query_embedding = encode_text(request.query)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            include=["metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['ids'][0]:
            for i, (result_id, distance, metadata) in enumerate(
                zip(results['ids'][0], results['distances'][0], results['metadatas'][0])
            ):
                # Convert distance to similarity
                score = 1 - (distance / 2) 
                formatted_results.append({
                    "id": result_id,
                    "score": round(score, 3),
                    "url": metadata.get('url', ''),
                    "title": metadata.get('title', ''),
                    "metadata": metadata
                })
        
        return {
            "query": request.query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_image(
    image: UploadFile = File(...),
    top_k: int = Form(10)
):
    """Search using uploaded image"""
    temp_path = None
    try:
        # Save uploaded image temporarily
        file_ext = os.path.splitext(image.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(temp_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Generate image embedding
        query_embedding = encode_image(temp_path)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['ids'][0]:
            for i, (result_id, distance, metadata) in enumerate(
                zip(results['ids'][0], results['distances'][0], results['metadatas'][0])
            ):
                score = 1 - (distance / 2)
                formatted_results.append({
                    "id": result_id,
                    "score": round(score, 3),
                    "url": metadata.get('url', ''),
                    "title": metadata.get('title', ''),
                    "metadata": metadata
                })
        
        return {
            "query": f"Image: {image.filename}",
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/search/hybrid")
async def search_hybrid(
    image: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(10),
    text_weight: float = Form(0.5)
):
    """Hybrid search using both text and image"""
    temp_path = None
    try:
        # Save uploaded image
        file_ext = os.path.splitext(image.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(temp_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Generate embeddings
        text_embedding = encode_text(query)
        image_embedding = encode_image(temp_path)
        
        # Combine embeddings
        import numpy as np
        text_emb = np.array(text_embedding)
        image_emb = np.array(image_embedding)
        combined = text_weight * text_emb + (1 - text_weight) * image_emb
        combined = combined / np.linalg.norm(combined)
        
        # Search
        results = collection.query(
            query_embeddings=[combined.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['ids'][0]:
            for i, (result_id, distance, metadata) in enumerate(
                zip(results['ids'][0], results['distances'][0], results['metadatas'][0])
            ):
                score = 1 - (distance / 2)
                formatted_results.append({
                    "id": result_id,
                    "score": round(score, 3),
                    "url": metadata.get('url', ''),
                    "title": metadata.get('title', ''),
                    "metadata": metadata
                })
        
        return {
            "query": f"Text: '{query}' + Image: {image.filename}",
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)