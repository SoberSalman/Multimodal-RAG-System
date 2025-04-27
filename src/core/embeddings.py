import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import logging
from typing import List, Union
import os


logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for text and image chunks"""
    
    def __init__(self, text_model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize text model
        self.text_model = SentenceTransformer(text_model_name)
        self.text_model.to(self.device)
        
        # Initialize CLIP model for images
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_available = True
        except Exception as e:
            logger.warning(f"Could not load CLIP model: {e}")
            self.clip_available = False
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.zeros(self.text_model.get_sentence_embedding_dimension())
    
    def generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for image"""
        if not self.clip_available:
            logger.warning("CLIP not available, using text embedding for image")
            # If CLIP is not available, use OCR text as fallback
            try:
                import pytesseract
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                return self.generate_text_embedding(text)
            except Exception as e:
                logger.error(f"Error in fallback image embedding: {e}")
                return np.zeros(self.text_model.get_sentence_embedding_dimension())
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features.cpu().numpy().squeeze()
            
            # Normalize to match text embedding dimensions
            if image_features.shape[0] != self.text_model.get_sentence_embedding_dimension():
                # Use a linear projection layer to match dimensions
                projection = torch.nn.Linear(
                    image_features.shape[0], 
                    self.text_model.get_sentence_embedding_dimension()
                )
                projection.to(self.device)
                image_features_tensor = torch.from_numpy(image_features).to(self.device)
                image_features = projection(image_features_tensor).detach().cpu().numpy()
            
            return image_features
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return np.zeros(self.text_model.get_sentence_embedding_dimension())
    
    def generate_embedding(self, content: Union[str, dict]) -> np.ndarray:
        """Generate embedding based on content type"""
        if isinstance(content, str):
            return self.generate_text_embedding(content)
        elif isinstance(content, dict) and 'image_path' in content:
            return self.generate_image_embedding(content['image_path'])
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    def batch_generate_embeddings(self, contents: List[Union[str, dict]]) -> List[np.ndarray]:
        """Generate embeddings for a batch of contents"""
        embeddings = []
        for content in contents:
            embedding = self.generate_embedding(content)
            embeddings.append(embedding)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings"""
        return self.text_model.get_sentence_embedding_dimension()
