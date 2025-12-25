# Objective

A Pinterest-style multimodal search engine that allows users to search for products using text descriptions, images, or both combined. Built with OpenCLIP, ChromaDB, and FastAPI. The multimodal item search engine enables users to find products either through textual descriptions and/or visual similarity. By combining image-based and text-based retrieval, it helps users quickly locate the exact items they need without extensive searching. This not only improves the shopping experience but also helps companies connect customers with the products that best match their preferences, increasing satisfaction and conversion rates.

## Features

- Text-based Search: Find items using natural language descriptions
- Image-based Search: Upload an image to find visually similar items
- Multimodal Search: Combine text and image queries for more precise results
- Vector Similarity: Uses CLIP embeddings for semantic understanding
- Fast Retrieval: ChromaDB for efficient vector search
- ChromaDB: open source vector database 
- CLIP model: multimodal vision and language model
- FastAPI: serves real-time queries 

## How It Works

1. Embedding Generation: OpenCLIP converts images and text into 512-dimensional vectors in a shared embedding space
2. Vector Storage: ChromaDB stores embeddings with metadata for fast retrieval
3. Similarity Search: Queries are embedded and compared using cosine similarity
4. Ranking: Top-k most similar items are returned based on distance metrics

## Tech Stack

*OpenCLIP*: Multimodal embeddings (ViT-B-32 model)
*ChromaDB*: Vector database for similarity search
*PyTorch*: Deep learning framework
*FastAPI*: Web framework for API (coming soon)
*Pandas*: Data manipulation
*Pillow*: Image processing

## Next Steps

Build Frontend: Connect a frontend application to these APIs
Deploy: Deploy to Docker.
