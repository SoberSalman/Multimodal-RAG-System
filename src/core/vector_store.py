# src/core/vector_store.py

import faiss
import numpy as np
import json
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple
from .pdf_extractor import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving embeddings"""
    
    def __init__(self, dimension: int = 384, storage_path: str = ".cache"):
        self.dimension = dimension
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.metadata = []
        
        # Paths for persistence
        self.index_path = os.path.join(storage_path, "faiss_index.bin")
        self.chunks_path = os.path.join(storage_path, "chunks.pkl")
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        
        # Load existing data if available
        self.load()
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """Add chunks to the vector store"""
        if not chunks:
            return
        
        # Extract embeddings
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping chunk {chunk.metadata.get('chunk_id', 'unknown')} with no embedding")
        
        if not embeddings:
            logger.warning("No valid embeddings to add to the vector store")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks.extend(valid_chunks)
        self.metadata.extend([chunk.metadata for chunk in valid_chunks])
        
        # Save to disk
        self.save()
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        # Reshape query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return results with distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(distance)))
        
        return results
    
    def save(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save chunks with embeddings preserved
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.info(f"Saved {len(self.chunks)} chunks to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self):
        """Load index and metadata from disk"""
        try:
            if os.path.exists(self.index_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                
                # Load chunks with embeddings
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # Verify chunks have embeddings
                missing_embeddings = 0
                for i, chunk in enumerate(self.chunks):
                    if chunk.embedding is None:
                        missing_embeddings += 1
                
                if missing_embeddings > 0:
                    logger.warning(f"Found {missing_embeddings} chunks without embeddings")
                
                logger.info(f"Loaded {len(self.chunks)} chunks from {self.storage_path}")
            else:
                logger.info("No existing vector store found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []
            self.metadata = []
    
    def clear(self):
        """Clear the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.metadata = []
        
        # Remove saved files
        for path in [self.index_path, self.chunks_path, self.metadata_path]:
            if os.path.exists(path):
                os.remove(path)
        
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        stats = {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "types": {},
            "chunks_with_embeddings": 0
        }
        
        # Count chunks by type and check embeddings
        for chunk in self.chunks:
            chunk_type = chunk.type
            stats["types"][chunk_type] = stats["types"].get(chunk_type, 0) + 1
            if chunk.embedding is not None:
                stats["chunks_with_embeddings"] += 1
        
        stats["chunks_without_embeddings"] = stats["total_chunks"] - stats["chunks_with_embeddings"]
        
        return stats