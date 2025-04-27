# src/core/evaluation.py

import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import time
import logging
from collections import Counter

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Could not download NLTK data. Some evaluation metrics may not work.")

class Evaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self):
        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()
    
    def calculate_retrieval_metrics(self, queries: List[str], ground_truth: List[List[str]], 
                                  retrieved: List[List[str]], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """Calculate retrieval metrics (Precision@K, Recall@K, MAP)"""
        metrics = {f"precision@{k}": [] for k in k_values}
        metrics.update({f"recall@{k}": [] for k in k_values})
        metrics['map'] = []
        
        for query_idx, (gt_docs, retrieved_docs) in enumerate(zip(ground_truth, retrieved)):
            # Calculate precision and recall at different k values
            for k in k_values:
                retrieved_at_k = retrieved_docs[:k]
                relevant_retrieved = len(set(retrieved_at_k) & set(gt_docs))
                
                precision = relevant_retrieved / k if k > 0 else 0
                recall = relevant_retrieved / len(gt_docs) if len(gt_docs) > 0 else 0
                
                metrics[f"precision@{k}"].append(precision)
                metrics[f"recall@{k}"].append(recall)
            
            # Calculate Average Precision
            ap = self._calculate_average_precision(gt_docs, retrieved_docs)
            metrics['map'].append(ap)
        
        # Calculate mean values
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def _calculate_average_precision(self, ground_truth: List[str], retrieved: List[str]) -> float:
        """Calculate Average Precision for a single query"""
        if not ground_truth:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, doc in enumerate(retrieved):
            if doc in ground_truth and doc not in retrieved[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / len(ground_truth) if ground_truth else 0.0
    
    def calculate_generation_metrics(self, reference_answers: List[str], 
                                   generated_answers: List[str]) -> Dict[str, Any]:
        """Calculate generation quality metrics (BLEU, ROUGE)"""
        metrics = {
            'bleu_1': [],
            'bleu_2': [],
            'bleu_3': [],
            'bleu_4': [],
            'rouge_1': [],
            'rouge_2': [],
            'rouge_l': []
        }
        
        for ref, gen in zip(reference_answers, generated_answers):
            # BLEU scores
            ref_tokens = nltk.word_tokenize(ref.lower())
            gen_tokens = nltk.word_tokenize(gen.lower())
            
            for n in range(1, 5):
                score = sentence_bleu([ref_tokens], gen_tokens, 
                                    weights=tuple([1/n] * n + [0] * (4-n)),
                                    smoothing_function=self.smoothing.method1)
                metrics[f'bleu_{n}'].append(score)
            
            # ROUGE scores
            try:
                rouge_scores = self.rouge.get_scores(gen, ref)[0]
                metrics['rouge_1'].append(rouge_scores['rouge-1']['f'])
                metrics['rouge_2'].append(rouge_scores['rouge-2']['f'])
                metrics['rouge_l'].append(rouge_scores['rouge-l']['f'])
            except:
                metrics['rouge_1'].append(0)
                metrics['rouge_2'].append(0)
                metrics['rouge_l'].append(0)
        
        # Calculate mean values
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def calculate_semantic_similarity(self, queries: List[str], responses: List[str], 
                                    embedding_generator) -> Dict[str, Any]:
        """Calculate semantic similarity between queries and responses"""
        similarities = []
        
        for query, response in zip(queries, responses):
            query_embedding = embedding_generator.generate_text_embedding(query)
            response_embedding = embedding_generator.generate_text_embedding(response)
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                response_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append(similarity)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def calculate_retrieval_coverage(self, retrieved_chunks: List[List[Dict]]) -> Dict[str, Any]:
        """Calculate coverage metrics for retrieved chunks"""
        all_types = []
        all_sources = []
        all_pages = []
        
        for chunk_list in retrieved_chunks:
            for chunk in chunk_list:
                all_types.append(chunk['type'])
                all_sources.append(chunk['source'])
                all_pages.append(chunk['page'])
        
        type_distribution = Counter(all_types)
        source_distribution = Counter(all_sources)
        page_distribution = Counter(all_pages)
        
        return {
            'type_distribution': dict(type_distribution),
            'source_distribution': dict(source_distribution),
            'unique_pages_accessed': len(set(all_pages)),
            'total_chunks_retrieved': len(all_types)
        }
    
    def evaluate_query_performance(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate overall query performance"""
        response_times = [result['response_time'] for result in query_results]
        num_sources = [result['num_sources'] for result in query_results]
        
        prompt_types = [result['prompt_type'] for result in query_results]
        prompt_distribution = Counter(prompt_types)
        
        # Calculate retrieval diversity
        all_distances = []
        for result in query_results:
            for source in result['sources']:
                all_distances.append(source['distance'])
        
        return {
            'avg_response_time': np.mean(response_times),
            'std_response_time': np.std(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'avg_sources_per_query': np.mean(num_sources),
            'prompt_type_distribution': dict(prompt_distribution),
            'avg_retrieval_distance': np.mean(all_distances) if all_distances else 0,
            'std_retrieval_distance': np.std(all_distances) if all_distances else 0
        }
    
    def calculate_relevance_score(self, query: str, response: str, sources: List[Dict], 
                                embedding_generator) -> float:
        """Calculate relevance score based on multiple factors"""
        # Semantic similarity between query and response
        query_emb = embedding_generator.generate_text_embedding(query)
        response_emb = embedding_generator.generate_text_embedding(response)
        query_response_sim = cosine_similarity(
            query_emb.reshape(1, -1),
            response_emb.reshape(1, -1)
        )[0][0]
        
        # Average similarity to retrieved sources
        source_similarities = []
        for source in sources:
            source_emb = embedding_generator.generate_text_embedding(source['content'])
            sim = cosine_similarity(
                query_emb.reshape(1, -1),
                source_emb.reshape(1, -1)
            )[0][0]
            source_similarities.append(sim)
        
        avg_source_sim = np.mean(source_similarities) if source_similarities else 0
        
        # Combine scores (weighted average)
        relevance_score = (0.6 * query_response_sim + 0.4 * avg_source_sim)
        
        return relevance_score