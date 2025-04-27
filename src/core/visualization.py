# src/core/visualization.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualization tools for RAG system analysis"""
    
    def __init__(self):
        self.color_scheme = {
            'text': '#1f77b4',
            'image': '#ff7f0e',
            'table': '#2ca02c',
            'primary': '#4CAF50',
            'secondary': '#2196F3',
            'accent': '#FF9800'
        }
    
    def create_embedding_visualization(self, embeddings: np.ndarray, labels: List[str], 
                                     method: str = 't-SNE', n_components: int = 2) -> go.Figure:
        """Create visualization of embedding space"""
        if method.lower() == 't-sne':
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=min(30, len(embeddings)-1))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'label': labels
        })
        
        fig = px.scatter(
            df, x='x', y='y', color='label',
            title=f'Embedding Space Visualization ({method})',
            labels={'x': f'{method} Component 1', 'y': f'{method} Component 2'},
            color_discrete_map=self.color_scheme
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        return fig
    
    def create_similarity_heatmap(self, embeddings: np.ndarray, labels: List[str]) -> go.Figure:
        """Create similarity heatmap between embeddings"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity_matrix = cosine_similarity(embeddings)
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Cosine Similarity Heatmap',
            xaxis_title='Documents',
            yaxis_title='Documents'
        )
        
        return fig
    
    def create_performance_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create comprehensive performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Response Time Distribution',
                'Retrieval Metrics',
                'Generation Quality',
                'Query Types Distribution'
            ),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Response time histogram
        if 'response_times' in metrics:
            fig.add_trace(
                go.Histogram(x=metrics['response_times'], nbinsx=20, name='Response Time'),
                row=1, col=1
            )
        
        # Retrieval metrics bar chart
        if 'retrieval_metrics' in metrics:
            retrieval_data = metrics['retrieval_metrics']
            fig.add_trace(
                go.Bar(
                    x=list(retrieval_data.keys()),
                    y=list(retrieval_data.values()),
                    name='Retrieval Metrics'
                ),
                row=1, col=2
            )
        
        # Generation quality metrics
        if 'generation_metrics' in metrics:
            gen_data = metrics['generation_metrics']
            fig.add_trace(
                go.Bar(
                    x=list(gen_data.keys()),
                    y=list(gen_data.values()),
                    name='Generation Metrics'
                ),
                row=2, col=1
            )
        
        # Query type distribution
        if 'prompt_distribution' in metrics:
            prompt_data = metrics['prompt_distribution']
            fig.add_trace(
                go.Pie(
                    labels=list(prompt_data.keys()),
                    values=list(prompt_data.values()),
                    name='Query Types'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="RAG System Performance Dashboard"
        )
        
        return fig
    
    def create_retrieval_visualization(self, query_embedding: np.ndarray, 
                                     chunk_embeddings: np.ndarray, 
                                     retrieved_indices: List[int]) -> go.Figure:
        """Visualize query and retrieved chunks in embedding space"""
        all_embeddings = np.vstack([query_embedding.reshape(1, -1), chunk_embeddings])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        reduced_embeddings = tsne.fit_transform(all_embeddings)
        
        fig = go.Figure()
        
        # Plot all chunks
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[1:, 0],
            y=reduced_embeddings[1:, 1],
            mode='markers',
            name='All Chunks',
            marker=dict(color='lightgray', size=8)
        ))
        
        # Highlight retrieved chunks
        retrieved_embeddings = reduced_embeddings[1:][retrieved_indices]
        fig.add_trace(go.Scatter(
            x=retrieved_embeddings[:, 0],
            y=retrieved_embeddings[:, 1],
            mode='markers',
            name='Retrieved Chunks',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        # Plot query
        fig.add_trace(go.Scatter(
            x=[reduced_embeddings[0, 0]],
            y=[reduced_embeddings[0, 1]],
            mode='markers',
            name='Query',
            marker=dict(color='green', size=15, symbol='diamond')
        ))
        
        fig.update_layout(
            title='Query and Retrieved Chunks in Embedding Space',
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            showlegend=True
        )
        
        return fig
    
    def create_coverage_visualization(self, coverage_metrics: Dict[str, Any]) -> go.Figure:
        """Visualize coverage metrics"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Type Distribution', 'Source Distribution', 'Page Coverage'),
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Type distribution
        if 'type_distribution' in coverage_metrics:
            type_data = coverage_metrics['type_distribution']
            fig.add_trace(
                go.Pie(
                    labels=list(type_data.keys()),
                    values=list(type_data.values()),
                    name='Types'
                ),
                row=1, col=1
            )
        
        # Source distribution
        if 'source_distribution' in coverage_metrics:
            source_data = coverage_metrics['source_distribution']
            fig.add_trace(
                go.Bar(
                    x=list(source_data.keys()),
                    y=list(source_data.values()),
                    name='Sources'
                ),
                row=1, col=2
            )
        
        # Page coverage histogram
        if 'page_distribution' in coverage_metrics:
            page_data = coverage_metrics.get('page_distribution', {})
            fig.add_trace(
                go.Bar(
                    x=list(page_data.keys()),
                    y=list(page_data.values()),
                    name='Pages'
                ),
                row=1, col=3
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Coverage Analysis"
        )
        
        return fig
    
    def create_semantic_drift_plot(self, query_embeddings: List[np.ndarray], 
                                 response_embeddings: List[np.ndarray]) -> go.Figure:
        """Visualize semantic drift between queries and responses"""
        similarities = []
        for q_emb, r_emb in zip(query_embeddings, response_embeddings):
            sim = cosine_similarity(q_emb.reshape(1, -1), r_emb.reshape(1, -1))[0][0]
            similarities.append(sim)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=similarities,
            mode='lines+markers',
            name='Query-Response Similarity',
            line=dict(color=self.color_scheme['primary'])
        ))
        
        fig.add_hline(y=np.mean(similarities), line_dash="dash", 
                     annotation_text=f"Mean: {np.mean(similarities):.3f}")
        
        fig.update_layout(
            title='Semantic Similarity between Queries and Responses',
            xaxis_title='Query Index',
            yaxis_title='Cosine Similarity',
            yaxis_range=[0, 1]
        )
        
        return fig
    
    def create_chunk_distribution_plot(self, chunks: List[Dict]) -> go.Figure:
        """Visualize distribution of chunks across documents"""
        df = pd.DataFrame(chunks)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Chunks per Document', 'Chunks by Type and Document'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Chunks per document
        doc_counts = df['source'].value_counts()
        fig.add_trace(
            go.Bar(
                x=doc_counts.index,
                y=doc_counts.values,
                name='Chunks per Document'
            ),
            row=1, col=1
        )
        
        # Chunks by type and document (stacked)
        type_doc_counts = df.groupby(['source', 'type']).size().unstack(fill_value=0)
        for chunk_type in type_doc_counts.columns:
            fig.add_trace(
                go.Bar(
                    x=type_doc_counts.index,
                    y=type_doc_counts[chunk_type],
                    name=chunk_type,
                    marker_color=self.color_scheme.get(chunk_type, 'gray')
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=500,
            barmode='stack',
            showlegend=True,
            title_text="Chunk Distribution Analysis"
        )
        
        return fig