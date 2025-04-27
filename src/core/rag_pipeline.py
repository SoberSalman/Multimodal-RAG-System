# src/core/rag_pipeline.py

import os
import logging
from typing import List, Dict, Any, Union, Tuple
from tqdm import tqdm
import time
import numpy as np

from .llm_client import LocalLLM
from .pdf_extractor import PDFExtractor, DocumentChunk
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .evaluation import Evaluator
from .visualization import Visualizer

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Complete RAG pipeline for document processing and querying with evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.llm = LocalLLM(
            endpoint_url=config['llm']['endpoint_url'],
            model=config['llm']['model'],
            temperature=config['llm']['temperature'],
            max_tokens=config['llm']['max_tokens']
        )
        
        self.pdf_extractor = PDFExtractor(temp_dir=config.get('data', {}).get('temp_directory', 'temp'))
        self.embedding_generator = EmbeddingGenerator(
            text_model_name=config['embeddings']['model_name']
        )
        self.vector_store = VectorStore(
            dimension=config['embeddings']['dimension'],
            storage_path=config.get('vector_store', {}).get('storage_path', '.cache')
        )
        
        # Initialize evaluation and visualization
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        
        self.chunking_config = config.get('chunking', {})
        self.retrieval_config = config.get('retrieval', {})
        
        # Store query history for evaluation
        self.query_history = []
    
    def process_documents(self, pdf_paths: List[str], force_reload: bool = False):
        """Process PDF documents and build vector store"""
        if not force_reload and len(self.vector_store.chunks) > 0:
            logger.info(f"Vector store already contains {len(self.vector_store.chunks)} chunks. Use force_reload=True to reprocess.")
            
            # Ensure all chunks have embeddings
            missing_embeddings = False
            for i, chunk in enumerate(self.vector_store.chunks):
                if chunk.embedding is None:
                    logger.info(f"Generating missing embedding for chunk {i}")
                    missing_embeddings = True
                    if chunk.type == 'text' or chunk.type == 'table':
                        chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
                    elif chunk.type == 'image' and 'image_path' in chunk.metadata:
                        try:
                            chunk.embedding = self.embedding_generator.generate_image_embedding(chunk.metadata['image_path'])
                        except:
                            logger.warning(f"Could not generate image embedding for chunk {i}, using text content")
                            chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
            
            if missing_embeddings:
                # Update vector store
                self.vector_store.clear()
                self.vector_store.add_chunks(self.vector_store.chunks)
            
            return
        
        all_chunks = []
        
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            logger.info(f"Processing {pdf_path}")
            
            # Extract chunks from PDF
            chunks = self.pdf_extractor.extract_from_pdf(pdf_path)
            
            # Generate embeddings for each chunk
            for chunk in tqdm(chunks, desc=f"Generating embeddings for {os.path.basename(pdf_path)}"):
                if chunk.type == 'text':
                    chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
                elif chunk.type == 'image' and 'image_path' in chunk.metadata:
                    try:
                        chunk.embedding = self.embedding_generator.generate_image_embedding(chunk.metadata['image_path'])
                    except:
                        logger.warning(f"Could not generate image embedding, using text content")
                        chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
                elif chunk.type == 'table':
                    chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
                else:
                    logger.warning(f"Skipping chunk of unknown type: {chunk.type}")
                    continue
            
            all_chunks.extend(chunks)
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks)
        logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
    
    def query(self, question: str, prompt_type: str = "zero_shot", k: int = None) -> Dict[str, Any]:
        """Query the RAG system with evaluation metrics"""
        start_time = time.time()
        
        if k is None:
            k = self.retrieval_config.get('top_k', 5)
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_text_embedding(question)
        
        # Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, k=k)
        
        # Prepare context from retrieved chunks
        context_parts = []
        sources = []
        retrieved_embeddings = []
        
        for chunk, distance in results:
            if chunk.type == 'text' or chunk.type == 'table':
                context_parts.append(chunk.content)
            elif chunk.type == 'image':
                if chunk.metadata.get('has_text', False):
                    context_parts.append(f"[Image content]: {chunk.content}")
                else:
                    context_parts.append(f"[Image from page {chunk.metadata['page']}]")
            
            sources.append({
                'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                'type': chunk.type,
                'page': chunk.metadata['page'],
                'source': os.path.basename(chunk.metadata['source']),
                'distance': distance,
                'chunk_id': chunk.metadata.get('chunk_id', '')
            })
            
            # Only append embedding if it exists
            if chunk.embedding is not None:
                retrieved_embeddings.append(chunk.embedding)
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        prompt = self.llm.get_prompt_template(prompt_type, context, question)
        response = self.llm.generate(prompt)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Calculate evaluation metrics
        relevance_score = self.evaluator.calculate_relevance_score(
            question, response, sources, self.embedding_generator
        )
        
        # Calculate semantic similarity
        response_embedding = self.embedding_generator.generate_text_embedding(response)
        query_response_similarity = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
        )
        
        result = {
            "question": question,
            "answer": response,
            "sources": sources,
            "prompt_type": prompt_type,
            "response_time": response_time,
            "num_sources": len(sources),
            "relevance_score": relevance_score,
            "query_response_similarity": query_response_similarity,
            "avg_retrieval_distance": np.mean([s["distance"] for s in sources]) if sources else 0,
            "query_embedding": query_embedding.tolist(),
            "response_embedding": response_embedding.tolist(),
            "retrieved_embeddings": [emb.tolist() for emb in retrieved_embeddings if emb is not None]
        }
        
        # Store in query history
        self.query_history.append(result)
        
        return result
    
    def query_with_image(self, image_path: str, k: int = None) -> Dict[str, Any]:
        """Query using an image with evaluation metrics"""
        start_time = time.time()
        
        if k is None:
            k = self.retrieval_config.get('top_k', 5)
        
        # Generate image embedding
        query_embedding = self.embedding_generator.generate_image_embedding(image_path)
        
        # Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, k=k)
        
        # Prepare results
        sources = []
        for chunk, distance in results:
            sources.append({
                'content': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                'type': chunk.type,
                'page': chunk.metadata['page'],
                'source': os.path.basename(chunk.metadata['source']),
                'distance': distance,
                'chunk_id': chunk.metadata.get('chunk_id', '')
            })
        
        end_time = time.time()
        
        return {
            "query_type": "image",
            "sources": sources,
            "response_time": end_time - start_time,
            "num_sources": len(sources),
            "avg_retrieval_distance": np.mean([s["distance"] for s in sources]) if sources else 0
        }
    
    def evaluate_system(self) -> Dict[str, Any]:
        """Comprehensive system evaluation based on query history"""
        if not self.query_history:
            return {"error": "No queries in history to evaluate"}
        
        # Evaluate query performance
        performance_metrics = self.evaluator.evaluate_query_performance(self.query_history)
        
        # Calculate retrieval coverage
        retrieved_chunks = [result["sources"] for result in self.query_history]
        coverage_metrics = self.evaluator.calculate_retrieval_coverage(retrieved_chunks)
        
        # Calculate semantic similarity metrics
        queries = [result["question"] for result in self.query_history]
        responses = [result["answer"] for result in self.query_history]
        similarity_metrics = self.evaluator.calculate_semantic_similarity(
            queries, responses, self.embedding_generator
        )
        
        return {
            "performance_metrics": performance_metrics,
            "coverage_metrics": coverage_metrics,
            "similarity_metrics": similarity_metrics,
            "total_queries": len(self.query_history)
        }
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization"""
        # Prepare embeddings for visualization
        all_embeddings = []
        all_labels = []
        
        for chunk in self.vector_store.chunks:
            if chunk.embedding is not None:
                all_embeddings.append(chunk.embedding)
                all_labels.append(chunk.type)
        
        # Query history data
        query_embeddings = []
        response_embeddings = []
        
        for result in self.query_history:
            if 'query_embedding' in result and 'response_embedding' in result:
                query_embeddings.append(np.array(result['query_embedding']))
                response_embeddings.append(np.array(result['response_embedding']))
        
        return {
            "chunk_embeddings": np.array(all_embeddings) if all_embeddings else None,
            "chunk_labels": all_labels,
            "query_embeddings": query_embeddings,
            "response_embeddings": response_embeddings,
            "query_history": self.query_history
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics with evaluation metrics"""
        base_stats = {
            "vector_store": self.vector_store.get_stats(),
            "embedding_model": self.embedding_generator.text_model._get_name(),
            "llm_model": self.llm.model,
            "config": {
                "chunk_size": self.chunking_config.get('chunk_size', 'default'),
                "retrieval_k": self.retrieval_config.get('top_k', 5)
            }
        }
        
        # Add evaluation metrics if available
        if self.query_history:
            evaluation_stats = self.evaluate_system()
            base_stats["evaluation"] = evaluation_stats
        
        return base_stats
    
    def ensure_embeddings(self):
        """Ensure all chunks have embeddings"""
        for chunk in self.vector_store.chunks:
            if chunk.embedding is None:
                logger.info(f"Generating missing embedding for chunk {chunk.metadata.get('chunk_id', 'unknown')}")
                if chunk.type == 'text' or chunk.type == 'table':
                    chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
                elif chunk.type == 'image' and 'image_path' in chunk.metadata:
                    try:
                        chunk.embedding = self.embedding_generator.generate_image_embedding(chunk.metadata['image_path'])
                    except:
                        logger.warning(f"Could not generate image embedding, using text content")
                        chunk.embedding = self.embedding_generator.generate_text_embedding(chunk.content)
        
        # Update vector store
        self.vector_store.clear()
        self.vector_store.add_chunks(self.vector_store.chunks)
    
    def clear_query_history(self):
        """Clear the query history"""
        self.query_history = []
    
    def get_chunk_visualization_data(self, chunk_ids: List[str]) -> Dict[str, Any]:
        """Get visualization data for specific chunks"""
        chunk_data = []
        
        for chunk in self.vector_store.chunks:
            if chunk.metadata.get('chunk_id', '') in chunk_ids:
                chunk_data.append({
                    'id': chunk.metadata.get('chunk_id', ''),
                    'content': chunk.content,
                    'type': chunk.type,
                    'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None,
                    'metadata': chunk.metadata
                })
        
        return {"chunks": chunk_data}
    
    def batch_evaluate_queries(self, test_queries: List[Dict[str, Any]], 
                             ground_truth: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Batch evaluate multiple queries for benchmarking"""
        results = []
        
        for query_data in tqdm(test_queries, desc="Evaluating queries"):
            result = self.query(
                query_data['question'],
                prompt_type=query_data.get('prompt_type', 'zero_shot')
            )
            results.append(result)
        
        # If ground truth is provided, calculate additional metrics
        if ground_truth:
            generated_answers = [r['answer'] for r in results]
            reference_answers = [gt['answer'] for gt in ground_truth]
            
            generation_metrics = self.evaluator.calculate_generation_metrics(
                reference_answers, generated_answers
            )
            
            # Calculate retrieval metrics if ground truth includes relevant chunks
            if all('relevant_chunks' in gt for gt in ground_truth):
                retrieved_chunks = [[s['chunk_id'] for s in r['sources']] for r in results]
                ground_truth_chunks = [gt['relevant_chunks'] for gt in ground_truth]
                
                retrieval_metrics = self.evaluator.calculate_retrieval_metrics(
                    [q['question'] for q in test_queries],
                    ground_truth_chunks,
                    retrieved_chunks
                )
            else:
                retrieval_metrics = None
            
            return {
                "results": results,
                "generation_metrics": generation_metrics,
                "retrieval_metrics": retrieval_metrics
            }
        
        return {"results": results}