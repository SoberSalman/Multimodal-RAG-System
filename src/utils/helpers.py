# src/utils/helpers.py

import logging
import sys
import os
from datetime import datetime
import json
from typing import Any, Dict, List

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

def save_query_history(query_result: Dict[str, Any], history_file: str = "query_history.jsonl"):
    """Save query results to history file"""
    try:
        os.makedirs(os.path.dirname(history_file) if os.path.dirname(history_file) else ".", exist_ok=True)
        
        # Add timestamp
        query_result['timestamp'] = datetime.now().isoformat()
        
        with open(history_file, 'a') as f:
            f.write(json.dumps(query_result) + '\n')
            
    except Exception as e:
        logging.error(f"Error saving query history: {e}")

def load_query_history(history_file: str = "query_history.jsonl") -> List[Dict[str, Any]]:
    """Load query history from file"""
    history = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
        except Exception as e:
            logging.error(f"Error loading query history: {e}")
    
    return history

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"

def get_file_size(file_path: str) -> str:
    """Get file size in human readable format"""
    try:
        size_bytes = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    except Exception:
        return "Unknown"

def ensure_directories(directories: List[str]):
    """Ensure directories exist"""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)