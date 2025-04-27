#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH to ensure proper module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Disable streamlit's file watching for torch
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="none"

# Reduce asyncio debug mode which can cause issues with torch
export PYTHONMALLOC=malloc

# Start the application
streamlit run src/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# config.yaml
llm:
  endpoint_url: "http://0.0.0.0:1234/v1/chat/completions"
  model: "gemma-2-2b-it:2"
  temperature: 0.7
  max_tokens: 2048

embeddings:
  model_name: "all-MiniLM-L6-v2"
  dimension: 384

vector_store:
  type: "faiss"
  storage_path: ".cache"

chunking:
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5

data:
  pdf_directory: "data/"
  temp_directory: "temp/"

# test_pipeline.py
"""Test script to verify the RAG pipeline works correctly"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.rag_pipeline import RAGPipeline
from utils.config import Config
from utils.helpers import setup_logging

def test_pipeline():
    """Test the RAG pipeline with some sample queries"""
    
    # Set up logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config().config
        
        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = RAGPipeline(config)
        
        # Process documents
        pdf_directory = config['data']['pdf_directory']
        pdf_files = [
            os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory)
            if f.endswith('.pdf')
        ]
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        pipeline.process_documents(pdf_files)
        
        # Test queries
        test_queries = [
            {
                "question": "What was the revenue in 2023?",
                "prompt_type": "zero_shot"
            },
            {
                "question": "Explain the profit trend over the years.",
                "prompt_type": "cot"
            },
            {
                "question": "What are the key financial metrics?",
                "prompt_type": "few_shot"
            }
        ]
        
        # Run test queries
        for query in test_queries:
            logger.info(f"\nTesting query: {query['question']}")
            result = pipeline.query(query["question"], prompt_type=query["prompt_type"])
            
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Response time: {result['response_time']:.2f} seconds")
            logger.info(f"Sources: {len(result['sources'])} documents")
            logger.info("-" * 50)
        
        # Test system stats
        stats = pipeline.get_system_stats()
        logger.info(f"\nSystem Stats: {stats}")
        
        logger.info("\nPipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    test_pipeline()

# query_prompts.txt
# Example prompts used in the multimodal RAG system

## Zero-shot Prompting
Question: What was the revenue in 2023?
Context: [Retrieved content from documents]
Answer: Based on the context, the revenue in 2023 was [specific amount from documents].

## Chain-of-Thought (CoT) Prompting
Question: Explain the trend in profit margins over the years.
Let's approach this step by step:
1. First, I'll identify the key information in the context about profit margins
2. Then, I'll analyze how it relates to the question about trends
3. Finally, I'll formulate a clear and complete answer

Answer: Analyzing the profit margin data:
1. In 2021, the profit margin was X%
2. In 2022, it increased to Y%
3. In 2023, it reached Z%
The trend shows a consistent increase in profit margins over the three-year period, indicating improving operational efficiency.

## Few-shot Prompting
Example 1:
Context: The revenue for Q1 2023 was $1.2 million.
Question: What was the Q1 revenue?
Answer: The Q1 revenue was $1.2 million.

Example 2:
Context: The chart shows increasing profit margins from 15% to 25% over 3 years.
Question: How did profit margins change?
Answer: Profit margins increased from 15% to 25% over a 3-year period.

Now answer this question:
Context: [Retrieved content]
Question: What are the key financial metrics presented in the annual report?
Answer: [Model's response based on examples]

# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
env/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/*.pdf
temp/
.cache/
logs/
*.log
query_history.jsonl

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# README.md
# Multimodal RAG System

A complete Retrieval-Augmented Generation (RAG) system that processes PDF documents containing both text and visual information. The system uses local LLMs via LM Studio for generating responses.

## Features

- 📄 Extract text, images, and tables from PDF documents
- 🔢 Generate embeddings using Sentence-BERT and CLIP
- 🔍 Semantic search using FAISS
- 💬 Generate responses using local LLMs
- 📊 Visualization of embeddings and analytics
- 🖥️ User-friendly Streamlit interface

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/multimodal-rag-system.git
cd multimodal-rag-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

5. Start LM Studio and load your model (e.g., gemma-2-2b-it:2)

6. Place your PDF files in the `data/` directory

7. Run the application:
```bash
chmod +x run.sh  # Make executable (Linux/Mac)
./run.sh
```

## Configuration

Edit `config.yaml` to customize:
- LLM endpoint and model
- Embedding model
- Vector store settings
- Retrieval parameters

## Usage

1. Click "Initialize/Reset System" to process PDFs
2. Enter text queries or upload images
3. Select prompting strategy (Zero-shot, CoT, Few-shot)
4. View results, sources, and analytics

## Testing

Run the test script to verify the pipeline:
```bash
python test_pipeline.py
```

## Architecture

The system consists of:
- PDF Extractor: Extracts content from PDFs
- Embedding Generator: Creates embeddings for text/images
- Vector Store: Stores and retrieves embeddings
- RAG Pipeline: Orchestrates the system
- Streamlit UI: User interface

## License

MIT License