# src/utils/config.py

import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the RAG system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set defaults if not specified
            config.setdefault('llm', {})
            config['llm'].setdefault('endpoint_url', "http://0.0.0.0:1234/v1/chat/completions")
            config['llm'].setdefault('model', "gemma-2-2b-it:2")
            config['llm'].setdefault('temperature', 0.7)
            config['llm'].setdefault('max_tokens', 2048)
            
            config.setdefault('embeddings', {})
            config['embeddings'].setdefault('model_name', "all-MiniLM-L6-v2")
            config['embeddings'].setdefault('dimension', 384)
            
            config.setdefault('vector_store', {})
            config['vector_store'].setdefault('type', "faiss")
            config['vector_store'].setdefault('storage_path', ".cache")
            
            config.setdefault('chunking', {})
            config['chunking'].setdefault('chunk_size', 500)
            config['chunking'].setdefault('chunk_overlap', 50)
            
            config.setdefault('retrieval', {})
            config['retrieval'].setdefault('top_k', 5)
            
            config.setdefault('data', {})
            config['data'].setdefault('pdf_directory', "data/")
            config['data'].setdefault('temp_directory', "temp/")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'llm': {
                'endpoint_url': "http://0.0.0.0:1234/v1/chat/completions",
                'model': "gemma-2-2b-it:2",
                'temperature': 0.7,
                'max_tokens': 2048
            },
            'embeddings': {
                'model_name': "all-MiniLM-L6-v2",
                'dimension': 384
            },
            'vector_store': {
                'type': "faiss",
                'storage_path': ".cache"
            },
            'chunking': {
                'chunk_size': 500,
                'chunk_overlap': 50
            },
            'retrieval': {
                'top_k': 5
            },
            'data': {
                'pdf_directory': "data/",
                'temp_directory': "temp/"
            }
        }
    
    def save_config(self):
        """Save configuration to YAML file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")