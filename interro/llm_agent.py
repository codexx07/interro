import subprocess
import json
from typing import Optional, List
from .retriever import SearchResult

class LLMAgent:
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('llm.enabled', False)
        self.model = config.get('llm.model', 'llama3')
        self.max_tokens = config.get('llm.max_tokens', 500)
        self.temperature = config.get('llm.temperature', 0.1)
    
    def explain_results(self, query: str, results: List[SearchResult]) -> Optional[str]:
        """Generate explanation for search results using LLM."""
        if not self.enabled:
            return None
        
        if not self._check_ollama():
            print("Warning: Ollama not available, skipping LLM explanation")
            return None
        
        # Prepare context from results
        context = self._prepare_context(results)
        
        # Create prompt
        prompt = self._create_explanation_prompt(query, context)
        
        # Query LLM
        try:
            return self._query_ollama(prompt)
        except Exception as e:
            print(f"Warning: LLM query failed: {e}")
            return None
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _prepare_context(self, results: List[SearchResult]) -> str:
        """Prepare context from search results."""
        context_parts = []
        
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            chunk = result.chunk
            context_parts.append(f"""
--- File: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}) ---
{chunk.content}
""")
        
        return '\n'.join(context_parts)
    
    def _create_explanation_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM explanation."""
        return f"""You are a code analysis assistant. A user asked: "{query}"

Here are the relevant code segments I found:

{context}

Please provide a clear, concise explanation that answers the user's question based on the code above. 
Focus on:
1. Directly answering their question
2. Explaining what the relevant code does
3. Highlighting key parts that relate to their query

Keep your response under {self.max_tokens} words and be specific to the code shown.

Answer:"""
    
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama with the given prompt."""
        cmd = ['ollama', 'run', self.model]
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=prompt, timeout=30)
            
            if process.returncode != 0:
                raise Exception(f"Ollama returned error: {stderr}")
            
            return stdout.strip()
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("Ollama query timed out")