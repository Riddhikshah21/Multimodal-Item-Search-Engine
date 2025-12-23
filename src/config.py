from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    chroma_db_path: str = "./data/pinteresty.db"
    # File Upload
    upload_dir: str = "./uploads"
    max_upload_size_mb: int = 10
    
    # Model Configuration
    vision_model_name: str = "ViT-B-32"
    pretrained_model: str = "openai"
    
    # API Configuration
    api_title: str = "Multimodal RAG Search API"
    api_version: str = "1.0.0"
    api_description: str = "Search system using CLIP embeddings and ChromaDB"
  
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


# Create settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(os.path.dirname(settings.chroma_db_path), exist_ok=True)
