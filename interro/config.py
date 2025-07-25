import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_CONFIG = {
    'retrieval': {
        'max_results': 3,  # Always return top 3 for quality
        'force_keyword_only': False,  # Set True to disable semantic search entirely
        'use_semantic_search': True,  # Enable/disable semantic search
        'similarity_threshold': 0.2,  # Lower = more results, higher = more precise
        'keyword_weight': 0.4,  # Weight for keyword/TF-IDF results
        'semantic_weight': 0.3,  # Weight for semantic results
        'pattern_weight': 0.3,  # Weight for pattern-based results
    },

    # Indexing settings
    'indexing': {
        'chunk_size': 800,  # Smaller chunks for better precision
        'chunk_overlap': 100,
        'max_file_size_mb': 5,
        'file_extensions': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h'],
        'exclude_dirs': ['__pycache__', '.git', 'node_modules', '.venv', 'venv'],
        'exclude_files': ['*.pyc', '*.log', '*.tmp'],
    },

    # Performance tuning
    'performance': {
        'enable_caching': True,  # Cache embeddings for speed
        'parallel_workers': 4,   # Number of parallel embedding workers
        'embedding_timeout': 5,  # Timeout for embedding requests
        'max_embed_length': 1000,  # Max text length for embedding
    },

    # Output formatting
    'output': {
        'show_explanations': True,  # Show why each result was selected
        'show_confidence': True,    # Show confidence scores
        'show_highlights': False,   # Disable highlights for cleaner output
        'verbose_search': True,     # Show search progress
    },
    'retrieval.max_results': 3,  # Only show top 3 quality results
    'retrieval.use_semantic_search': True,  # Enable semantic search for better understanding
    'retrieval.similarity_threshold': 0.2,  # Higher threshold for quality
    'retrieval.keyword_weight': 0.6,
    'retrieval.semantic_weight': 0.8,
    
    # === INDEXING SETTINGS ===
    'indexing.chunk_size': 800,  # Smaller chunks for better focus
    'indexing.chunk_overlap': 100,
    'indexing.max_file_size_mb': 3,  # Skip very large files
    'indexing.file_extensions': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h'],
    'indexing.exclude_dirs': ['node_modules', '__pycache__', '.git', 'venv', '.venv'],
    'indexing.exclude_files': ['*.pyc', '*.log', '*.tmp'],
    
    # === PERFORMANCE SETTINGS ===
    # Set to True for maximum speed (keyword + TF-IDF only)
    'retrieval.skip_semantic_for_speed': False,
}

class Config:
    def __init__(self, config_path: Optional[str] = None):
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            # Look for config in common locations
            for path in [".interro.yaml", "interro.yaml", "~/.interro.yaml"]:
                expanded_path = Path(path).expanduser()
                if expanded_path.exists():
                    self.load_config(str(expanded_path))
                    break
    
    def load_config(self, path: str):
        with open(path, 'r') as f:
            user_config = yaml.safe_load(f)
            self._merge_config(self.config, user_config)
    
    def _merge_config(self, base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value