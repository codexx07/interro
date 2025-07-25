"""
Main entry point for interro package.
This can be used for programmatic access.
"""
import os
from .config import Config
from .indexer import CodeIndexer
from .retriever import CodeRetriever
from .llm_agent import LLMAgent
from .formatter import ResultFormatter

class Interro:
    """Main Interro class for programmatic usage."""
    
    def __init__(self, config_path=None):
        self.config = Config(config_path)
        self.indexed_files = []
        self.retriever = None
    
    def index_path(self, path: str):
        """Index a file or directory."""
        indexer = CodeIndexer(self.config)
        
        if os.path.isfile(path):
            self.indexed_files = [indexer.index_file(path)]
            self.indexed_files = [f for f in self.indexed_files if f is not None]
        else:
            self.indexed_files = indexer.index_directory(path)
        
        # Initialize retriever with indexed files
        self.retriever = CodeRetriever(self.config, self.indexed_files)
        
        return len(self.indexed_files)
    
    def ask(self, query: str, use_llm: bool = None):
        """Ask a question about the indexed codebase."""
        if not self.retriever:
            raise ValueError("No codebase indexed. Call index_path() first.")
        
        # Search for results
        results = self.retriever.search(query)
        
        # Get LLM explanation if requested
        explanation = None
        if use_llm or (use_llm is None and self.config.get('llm.enabled', False)):
            llm_agent = LLMAgent(self.config)
            explanation = llm_agent.explain_results(query, results)
        
        return {
            'query': query,
            'results': results,
            'explanation': explanation,
            'total_files': len(self.indexed_files),
            'total_chunks': sum(len(f.chunks) for f in self.indexed_files)
        }
