# rebuild_cache.py

"""Script to rebuild the vector store cache with proper embeddings"""

import os
import sys
import shutil
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.rag_pipeline import RAGPipeline
from src.utils.config import Config
from src.utils.helpers import setup_logging

def rebuild_cache():
    """Rebuild the vector store cache"""
    
    # Set up logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config().config
        
        # Clear cache directory
        cache_dir = config.get('vector_store', {}).get('storage_path', '.cache')
        if os.path.exists(cache_dir):
            logger.info(f"Clearing existing cache directory: {cache_dir}")
            shutil.rmtree(cache_dir)
        
        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline(config)
        
        # Process documents from scratch
        pdf_directory = config['data']['pdf_directory']
        pdf_files = [
            os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory)
            if f.endswith('.pdf')
        ]
        
        logger.info(f"Processing {len(pdf_files)} PDF files from scratch...")
        pipeline.process_documents(pdf_files, force_reload=True)
        
        # Get stats
        stats = pipeline.get_system_stats()
        logger.info(f"Vector store stats: {stats['vector_store']}")
        
        logger.info("Cache rebuild completed successfully!")
        
    except Exception as e:
        logger.error(f"Error rebuilding cache: {e}", exc_info=True)

if __name__ == "__main__":
    rebuild_cache()