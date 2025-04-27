# test_evaluation.py

"""Test script to demonstrate RAG system evaluation metrics"""

import os
import sys
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.rag_pipeline import RAGPipeline
from core.visualization import Visualizer
from utils.config import Config
from utils.helpers import setup_logging

def test_evaluation(force_reload=False):
    """Test the RAG pipeline with evaluation metrics"""
    
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
        pipeline.process_documents(pdf_files, force_reload=force_reload)
        
        # Ensure all chunks have embeddings
        pipeline.ensure_embeddings()
        
        # Test queries for evaluation
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
            },
            {
                "question": "Describe the organizational structure.",
                "prompt_type": "zero_shot"
            },
            {
                "question": "What are the main business segments?",
                "prompt_type": "cot"
            }
        ]
        
        # Run test queries
        logger.info("\n" + "="*50)
        logger.info("Running test queries...")
        logger.info("="*50 + "\n")
        
        for i, query in enumerate(test_queries):
            logger.info(f"Query {i+1}: {query['question']}")
            result = pipeline.query(query["question"], prompt_type=query["prompt_type"])
            
            logger.info(f"Answer: {result['answer'][:200]}...")
            logger.info(f"Response time: {result['response_time']:.2f} seconds")
            logger.info(f"Relevance score: {result['relevance_score']:.2%}")
            logger.info(f"Semantic similarity: {result['query_response_similarity']:.2%}")
            logger.info(f"Sources: {len(result['sources'])} documents")
            logger.info("-" * 50)
        
        # System evaluation
        logger.info("\n" + "="*50)
        logger.info("System Evaluation Metrics")
        logger.info("="*50 + "\n")
        
        evaluation_metrics = pipeline.evaluate_system()
        
        # Display performance metrics
        perf_metrics = evaluation_metrics['performance_metrics']
        logger.info("Performance Metrics:")
        logger.info(f"  Average response time: {perf_metrics['avg_response_time']:.2f}s")
        logger.info(f"  95th percentile response: {perf_metrics['p95_response_time']:.2f}s")
        logger.info(f"  Average sources per query: {perf_metrics['avg_sources_per_query']:.1f}")
        logger.info(f"  Average retrieval distance: {perf_metrics['avg_retrieval_distance']:.4f}")
        
        # Display coverage metrics
        coverage_metrics = evaluation_metrics['coverage_metrics']
        logger.info("\nCoverage Metrics:")
        logger.info(f"  Total chunks retrieved: {coverage_metrics['total_chunks_retrieved']}")
        logger.info(f"  Unique pages accessed: {coverage_metrics['unique_pages_accessed']}")
        logger.info(f"  Type distribution: {coverage_metrics['type_distribution']}")
        
        # Display similarity metrics
        sim_metrics = evaluation_metrics['similarity_metrics']
        logger.info("\nSimilarity Metrics:")
        logger.info(f"  Mean similarity: {sim_metrics['mean_similarity']:.2%}")
        logger.info(f"  Std similarity: {sim_metrics['std_similarity']:.2%}")
        logger.info(f"  Min similarity: {sim_metrics['min_similarity']:.2%}")
        logger.info(f"  Max similarity: {sim_metrics['max_similarity']:.2%}")
        
        # Create visualizations
        logger.info("\nGenerating visualizations...")
        visualizer = Visualizer()
        
        # Get visualization data
        vis_data = pipeline.get_visualization_data()
        
        # Create embedding visualization
        if vis_data["chunk_embeddings"] is not None:
            fig_embeddings = visualizer.create_embedding_visualization(
                vis_data["chunk_embeddings"],
                vis_data["chunk_labels"],
                method='t-SNE'
            )
            fig_embeddings.write_html("embedding_visualization.html")
            logger.info("Saved embedding visualization to embedding_visualization.html")
        
        # Create performance dashboard
        metrics_data = {
            'response_times': [q['response_time'] for q in pipeline.query_history],
            'retrieval_metrics': perf_metrics,
            'generation_metrics': {
                'avg_relevance': sum(q['relevance_score'] for q in pipeline.query_history) / len(pipeline.query_history),
                'avg_similarity': sim_metrics['mean_similarity']
            },
            'prompt_distribution': perf_metrics['prompt_type_distribution']
        }
        
        fig_dashboard = visualizer.create_performance_dashboard(metrics_data)
        fig_dashboard.write_html("performance_dashboard.html")
        logger.info("Saved performance dashboard to performance_dashboard.html")
        
        # Create coverage visualization
        fig_coverage = visualizer.create_coverage_visualization(coverage_metrics)
        fig_coverage.write_html("coverage_analysis.html")
        logger.info("Saved coverage analysis to coverage_analysis.html")
        
        # Create semantic drift plot
        if vis_data["query_embeddings"] and vis_data["response_embeddings"]:
            fig_drift = visualizer.create_semantic_drift_plot(
                vis_data["query_embeddings"],
                vis_data["response_embeddings"]
            )
            fig_drift.write_html("semantic_drift.html")
            logger.info("Saved semantic drift analysis to semantic_drift.html")
        
        # Export evaluation results
        evaluation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(pipeline.query_history),
            "performance_metrics": perf_metrics,
            "coverage_metrics": coverage_metrics,
            "similarity_metrics": sim_metrics,
            "query_results": []
        }
        
        for result in pipeline.query_history:
            # Simplify the result for export
            simplified_result = {
                "question": result["question"],
                "answer": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
                "prompt_type": result["prompt_type"],
                "response_time": result["response_time"],
                "relevance_score": result["relevance_score"],
                "query_response_similarity": result["query_response_similarity"],
                "num_sources": result["num_sources"],
                "avg_retrieval_distance": result["avg_retrieval_distance"]
            }
            evaluation_results["query_results"].append(simplified_result)
        
        # Save evaluation results to JSON
        with open("evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info("Saved evaluation results to evaluation_results.json")
        
        # Create a simple matplotlib/seaborn visualization for response times
        plt.figure(figsize=(10, 6))
        response_times = [q['response_time'] for q in pipeline.query_history]
        plt.hist(response_times, bins=20, edgecolor='black')
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig('response_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved response time distribution to response_time_distribution.png")
        
        # Create a bar chart for prompt type performance
        plt.figure(figsize=(10, 6))
        prompt_types = [q['prompt_type'] for q in pipeline.query_history]
        relevance_scores = [q['relevance_score'] for q in pipeline.query_history]
        
        import pandas as pd
        df = pd.DataFrame({'prompt_type': prompt_types, 'relevance_score': relevance_scores})
        avg_relevance = df.groupby('prompt_type')['relevance_score'].mean()
        
        bars = plt.bar(avg_relevance.index, avg_relevance.values)
        plt.title('Average Relevance Score by Prompt Type')
        plt.xlabel('Prompt Type')
        plt.ylabel('Average Relevance Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')
        
        plt.savefig('prompt_type_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved prompt type performance to prompt_type_performance.png")
        
        logger.info("\nEvaluation test completed successfully!")
        logger.info("Check the generated files for detailed visualizations and metrics.")
        
    except Exception as e:
        logger.error(f"Error in evaluation testing: {e}", exc_info=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RAG system evaluation')
    parser.add_argument('--force-reload', action='store_true', help='Force reload all documents')
    args = parser.parse_args()
    
    test_evaluation(force_reload=args.force_reload)