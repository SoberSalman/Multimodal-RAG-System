# src/ui/streamlit_app.py

import streamlit as st
import os
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import logging

# Fix the PyTorch/Streamlit compatibility issue
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import our modules
from core.rag_pipeline import RAGPipeline
from core.visualization import Visualizer
from utils.config import Config
from utils.helpers import setup_logging, save_query_history, load_query_history, format_time

# Set up logging
setup_logging(log_file="logs/rag_system.log")
logger = logging.getLogger(__name__)

# Disable Streamlit's file watcher for torch modules
if "streamlit" in sys.modules:
    from streamlit.watcher import local_sources_watcher
    def get_module_paths_patched(module):
        if module.__name__.startswith('torch'):
            return []
        return local_sources_watcher.get_module_paths_original(module)
    
    local_sources_watcher.get_module_paths_original = local_sources_watcher.get_module_paths
    local_sources_watcher.get_module_paths = get_module_paths_patched

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System with Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .highlight {
        background-color: #fffacd;
        padding: 2px 5px;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
    st.session_state.processed_pdfs = False
    st.session_state.embeddings_vis = None
    st.session_state.query_history = []
    st.session_state.config = None
    st.session_state.visualizer = Visualizer()

def initialize_pipeline():
    """Initialize the RAG pipeline"""
    with st.spinner("Initializing RAG pipeline..."):
        try:
            # Load configuration
            config = Config().config
            st.session_state.config = config
            
            # Initialize pipeline
            st.session_state.rag_pipeline = RAGPipeline(config)
            
            # Get PDF files
            pdf_directory = config['data']['pdf_directory']
            pdf_files = [
                os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory)
                if f.endswith('.pdf')
            ]
            
            if not pdf_files:
                st.error(f"No PDF files found in {pdf_directory}")
                return False
            
            # Process documents
            st.session_state.rag_pipeline.process_documents(pdf_files)
            st.session_state.processed_pdfs = True
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing pipeline: {str(e)}")
            logger.error(f"Error initializing pipeline: {e}", exc_info=True)
            return False

def display_source(source: dict, index: int):
    """Display a single source in a formatted box"""
    with st.expander(f"Source {index + 1} - {source['source']} (Page {source['page']})"):
        cols = st.columns(3)
        cols[0].markdown(f"**Type:** {source['type'].capitalize()}")
        cols[1].markdown(f"**Relevance:** {1 / (1 + source['distance']):.2%}")
        cols[2].markdown(f"**Distance:** {source['distance']:.4f}")
        
        st.markdown("**Content:**")
        st.markdown(f"```\n{source['content']}\n```")

def main():
    st.title("📊 Multimodal RAG System with Evaluation")
    st.markdown("### Analyze and evaluate information retrieval from PDF documents")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ System Controls")
        
        if st.button("Initialize/Reset System", type="primary"):
            if initialize_pipeline():
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system")
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        
        prompt_type = st.selectbox(
            "Prompting Strategy",
            ["zero_shot", "cot", "few_shot"],
            format_func=lambda x: {
                "zero_shot": "Zero-shot",
                "cot": "Chain-of-Thought",
                "few_shot": "Few-shot"
            }[x]
        )
        
        num_results = st.slider("Number of retrieved chunks", 1, 10, 5)
        
        st.markdown("---")
        
        st.header("📊 System Status")
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_system_stats()
            st.markdown(f"**Total Chunks:** {stats['vector_store']['total_chunks']}")
            st.markdown(f"**Embedding Model:** {stats['embedding_model']}")
            st.markdown(f"**LLM Model:** {stats['llm_model']}")
            
            if 'evaluation' in stats:
                st.markdown("### 📈 Performance")
                perf = stats['evaluation']['performance_metrics']
                st.markdown(f"**Avg Response Time:** {perf['avg_response_time']:.2f}s")
                st.markdown(f"**Queries Processed:** {stats['evaluation']['total_queries']}")
    
    # Main content with more tabs
    tabs = st.tabs([
        "🔍 Query Interface", 
        "📊 Embeddings Visualization", 
        "📈 Performance Dashboard",
        "🔬 Evaluation Metrics",
        "🗺️ Retrieval Analysis",
        "⚡ System Analytics"
    ])
    
    # Query Interface Tab
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Text Query")
            
            query_text = st.text_area(
                "Enter your question:",
                placeholder="e.g., What was the revenue in 2023?",
                height=100
            )
            
            if st.button("🔍 Search", type="primary", disabled=not st.session_state.processed_pdfs):
                if query_text:
                    with st.spinner("Processing query..."):
                        try:
                            result = st.session_state.rag_pipeline.query(
                                query_text,
                                prompt_type=prompt_type,
                                k=num_results
                            )
                            
                            # Display answer
                            st.markdown("### 💡 Answer")
                            st.markdown(f'<div class="source-box">{result["answer"]}</div>', unsafe_allow_html=True)
                            
                            # Display evaluation metrics
                            metrics_cols = st.columns(4)
                            metrics_cols[0].metric("Response Time", format_time(result["response_time"]))
                            metrics_cols[1].metric("Sources Used", result["num_sources"])
                            metrics_cols[2].metric("Relevance Score", f"{result['relevance_score']:.2%}")
                            metrics_cols[3].metric("Semantic Similarity", f"{result['query_response_similarity']:.2%}")
                            
                            # Display sources
                            st.markdown("### 📚 Sources")
                            for i, source in enumerate(result["sources"]):
                                display_source(source, i)
                            
                            # Visualization of retrieval
                            st.markdown("### 🔍 Retrieval Visualization")
                            vis_data = st.session_state.rag_pipeline.get_visualization_data()
                            if vis_data["chunk_embeddings"] is not None:
                                # Create retrieval visualization
                                query_embedding = np.array(result["query_embedding"])
                                chunk_embeddings = vis_data["chunk_embeddings"]
                                retrieved_indices = [i for i, chunk in enumerate(st.session_state.rag_pipeline.vector_store.chunks) 
                                                   if chunk.metadata.get('chunk_id', '') in [s['chunk_id'] for s in result['sources']]]
                                
                                fig = st.session_state.visualizer.create_retrieval_visualization(
                                    query_embedding, chunk_embeddings, retrieved_indices
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Save to history
                            st.session_state.query_history.append(result)
                            save_query_history(result)
                            
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                            logger.error(f"Error processing query: {e}", exc_info=True)
                else:
                    st.warning("Please enter a question")
        
        with col2:
            st.header("Image Query")
            uploaded_image = st.file_uploader("Upload an image to find similar content", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_image and st.session_state.processed_pdfs:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🖼️ Find Similar Content", type="primary"):
                    with st.spinner("Processing image..."):
                        try:
                            # Save temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                tmp_file.write(uploaded_image.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Query with image
                            result = st.session_state.rag_pipeline.query_with_image(tmp_path, k=num_results)
                            
                            # Display metrics
                            metrics_cols = st.columns(3)
                            metrics_cols[0].metric("Response Time", format_time(result["response_time"]))
                            metrics_cols[1].metric("Similar Items", result["num_sources"])
                            metrics_cols[2].metric("Avg Distance", f"{result['avg_retrieval_distance']:.4f}")
                            
                            # Display results
                            st.markdown("### 🔍 Similar Content")
                            for i, source in enumerate(result["sources"]):
                                display_source(source, i)
                            
                            # Clean up
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            logger.error(f"Error processing image: {e}", exc_info=True)
    
    # Embeddings Visualization Tab
    with tabs[1]:
        st.header("Embeddings Visualization")
        
        if st.session_state.rag_pipeline:
            vis_data = st.session_state.rag_pipeline.get_visualization_data()
            
            if vis_data["chunk_embeddings"] is not None:
                # Choose visualization method
                viz_method = st.selectbox("Visualization Method", ["t-SNE", "PCA"])
                
                # Create embedding visualization
                fig_embeddings = st.session_state.visualizer.create_embedding_visualization(
                    vis_data["chunk_embeddings"],
                    vis_data["chunk_labels"],
                    method=viz_method
                )
                st.plotly_chart(fig_embeddings, use_container_width=True)
                
                # Similarity heatmap
                if st.checkbox("Show Similarity Heatmap"):
                    # Sample chunks for better visualization
                    sample_size = min(50, len(vis_data["chunk_embeddings"]))
                    indices = np.random.choice(len(vis_data["chunk_embeddings"]), sample_size, replace=False)
                    
                    fig_heatmap = st.session_state.visualizer.create_similarity_heatmap(
                        vis_data["chunk_embeddings"][indices],
                        [f"{vis_data['chunk_labels'][i]}_{i}" for i in indices]
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Initialize the system to see embeddings visualization.")
    
    # Performance Dashboard Tab
    with tabs[2]:
        st.header("Performance Dashboard")
        
        if st.session_state.rag_pipeline and st.session_state.rag_pipeline.query_history:
            evaluation_data = st.session_state.rag_pipeline.evaluate_system()
            
            # Prepare metrics for dashboard
            metrics_data = {
                'response_times': [q['response_time'] for q in st.session_state.rag_pipeline.query_history],
                'retrieval_metrics': evaluation_data['performance_metrics'],
                'generation_metrics': {
                    'avg_relevance': np.mean([q['relevance_score'] for q in st.session_state.rag_pipeline.query_history]),
                    'avg_similarity': evaluation_data['similarity_metrics']['mean_similarity']
                },
                'prompt_distribution': evaluation_data['performance_metrics']['prompt_type_distribution']
            }
            
            # Create dashboard
            fig_dashboard = st.session_state.visualizer.create_performance_dashboard(metrics_data)
            st.plotly_chart(fig_dashboard, use_container_width=True)
        else:
            st.info("Process some queries to see the performance dashboard.")
    
    # Evaluation Metrics Tab
    with tabs[3]:
        st.header("Evaluation Metrics")
        
        if st.session_state.rag_pipeline and st.session_state.rag_pipeline.query_history:
            evaluation_data = st.session_state.rag_pipeline.evaluate_system()
            
            # Performance Metrics
            st.subheader("📊 Performance Metrics")
            perf_metrics = evaluation_data['performance_metrics']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Response Time", f"{perf_metrics['avg_response_time']:.2f}s")
            col2.metric("95th Percentile Response", f"{perf_metrics['p95_response_time']:.2f}s")
            col3.metric("Avg Sources/Query", f"{perf_metrics['avg_sources_per_query']:.1f}")
            
            # Similarity Metrics
            st.subheader("🔍 Similarity Metrics")
            sim_metrics = evaluation_data['similarity_metrics']
            
            cols = st.columns(4)
            cols[0].metric("Mean Similarity", f"{sim_metrics['mean_similarity']:.2%}")
            cols[1].metric("Std Similarity", f"{sim_metrics['std_similarity']:.2%}")
            cols[2].metric("Min Similarity", f"{sim_metrics['min_similarity']:.2%}")
            cols[3].metric("Max Similarity", f"{sim_metrics['max_similarity']:.2%}")
            
            # Coverage Metrics
            st.subheader("📈 Coverage Metrics")
            coverage_metrics = evaluation_data['coverage_metrics']
            
            fig_coverage = st.session_state.visualizer.create_coverage_visualization(coverage_metrics)
            st.plotly_chart(fig_coverage, use_container_width=True)
            
            # Semantic Drift Plot
            st.subheader("📉 Semantic Drift Analysis")
            vis_data = st.session_state.rag_pipeline.get_visualization_data()
            
            if vis_data["query_embeddings"] and vis_data["response_embeddings"]:
                fig_drift = st.session_state.visualizer.create_semantic_drift_plot(
                    vis_data["query_embeddings"],
                    vis_data["response_embeddings"]
                )
                st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info("Process some queries to see evaluation metrics.")
    
    # Retrieval Analysis Tab
    with tabs[4]:
        st.header("Retrieval Analysis")
        
        if st.session_state.rag_pipeline and st.session_state.rag_pipeline.vector_store.chunks:
            # Chunk distribution analysis
            chunk_data = [{
                'source': chunk.metadata['source'],
                'type': chunk.type,
                'page': chunk.metadata['page']
            } for chunk in st.session_state.rag_pipeline.vector_store.chunks]
            
            fig_chunks = st.session_state.visualizer.create_chunk_distribution_plot(chunk_data)
            st.plotly_chart(fig_chunks, use_container_width=True)
            
            # Retrieval patterns
            if st.session_state.rag_pipeline.query_history:
                st.subheader("Retrieval Patterns")
                
                # Analyze which chunks are retrieved most frequently
                chunk_retrieval_count = {}
                for result in st.session_state.rag_pipeline.query_history:
                    for source in result['sources']:
                        chunk_id = source.get('chunk_id', '')
                        chunk_retrieval_count[chunk_id] = chunk_retrieval_count.get(chunk_id, 0) + 1
                
                # Display most frequently retrieved chunks
                sorted_chunks = sorted(chunk_retrieval_count.items(), key=lambda x: x[1], reverse=True)
                top_chunks = sorted_chunks[:10]
                
                if top_chunks:
                    st.markdown("### Most Frequently Retrieved Chunks")
                    for chunk_id, count in top_chunks:
                        st.markdown(f"- Chunk ID: {chunk_id}, Retrieved: {count} times")
        else:
            st.info("Initialize the system to see retrieval analysis.")
    
    # System Analytics Tab
    with tabs[5]:
        st.header("System Analytics")
        
        if st.session_state.rag_pipeline:
            # System resource usage
            st.subheader("🖥️ System Resources")
            stats = st.session_state.rag_pipeline.get_system_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Vector Store")
                st.json(stats['vector_store'])
            
            with col2:
                st.markdown("### Configuration")
                st.json(stats['config'])
            
            # Query history analysis
            if st.session_state.query_history:
                st.subheader("📜 Query History Analysis")
                
                history_df = pd.DataFrame(st.session_state.query_history)
                
                # Display recent queries
                st.markdown("### Recent Queries")
                recent_queries = history_df[['question', 'response_time', 'relevance_score', 'prompt_type']].tail(10)
                st.dataframe(recent_queries)
                
                # Export options
                st.markdown("### Export Data")
                if st.button("Export Query History"):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="query_history.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Initialize the system to see analytics.")

if __name__ == "__main__":
    main()