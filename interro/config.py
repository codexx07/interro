import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_CONFIG = {
    "indexing": {
        "file_extensions": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"],
        "exclude_dirs": [
            "__pycache__", ".git", "node_modules", ".venv", "venv", 
            "build", "dist", ".pytest_cache", ".mypy_cache"
        ],
        "exclude_files": ["*.pyc", "*.pyo", "*.pyd", ".DS_Store"],
        "max_file_size_mb": 5,
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "retrieval": {
        "max_results": 10,
        "similarity_threshold": 0.7,
        "use_semantic_search": True,
        "keyword_weight": 0.3,
        "semantic_weight": 0.7
    },
    "llm": {
        "enabled": False,
        "model": "llama3",
        "max_tokens": 500,
        "temperature": 0.1
    },
    "output": {
        "format": "rich",  # rich, json, plain
        "show_line_numbers": True,
        "context_lines": 5,
        "highlight_syntax": True
    }
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