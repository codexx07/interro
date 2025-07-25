import json
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.columns import Columns
from .retriever import SearchResult

class ResultFormatter:
    def __init__(self, config):
        self.config = config
        self.format_type = config.get('output.format', 'rich')
        self.console = Console()
    
    def format_results(self, query: str, results: List[SearchResult], 
                      explanation: Optional[str] = None) -> str:
        """Format search results according to configuration."""
        if self.format_type == 'json':
            return self._format_json(query, results, explanation)
        elif self.format_type == 'plain':
            return self._format_plain(query, results, explanation)
        else:
            return self._format_rich(query, results, explanation)
    
    def _format_rich(self, query: str, results: List[SearchResult], 
                    explanation: Optional[str] = None) -> str:
        """Format results using Rich for terminal display."""
        if not results:
            self.console.print(f"[yellow]No results found for query: '{query}'[/yellow]")
            return ""
        
        # Show query
        self.console.print(f"\n[bold blue]Query:[/bold blue] {query}")
        self.console.print(f"[dim]Found {len(results)} results[/dim]\n")
        
        # Show LLM explanation if available
        if explanation:
            self.console.print(Panel(explanation, title="[bold green]AI Explanation[/bold green]"))
            self.console.print()
        
        # Show results
        for i, result in enumerate(results, 1):
            self._format_single_result_rich(result, i)
        
        return ""  # Rich prints directly
    
    def _format_single_result_rich(self, result: SearchResult, index: int):
        """Format a single result using Rich."""
        chunk = result.chunk
        
        # Create title
        title_parts = [f"#{index}"]
        if chunk.name:
            title_parts.append(f"{chunk.name} ({chunk.chunk_type})")
        else:
            title_parts.append(chunk.chunk_type)
        
        title_parts.append(f"Score: {result.score:.2f}")
        title = " | ".join(title_parts)
        
        # Create subtitle with file info
        subtitle = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        
        # Format code with syntax highlighting
        if self.config.get('output.highlight_syntax', True):
            try:
                # Detect language from file extension
                file_ext = chunk.file_path.split('.')[-1]
                language_map = {
                    'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                    'java': 'java', 'cpp': 'cpp', 'c': 'c', 'h': 'c'
                }
                language = language_map.get(file_ext, 'text')
                
                syntax = Syntax(chunk.content, language, 
                              line_numbers=self.config.get('output.show_line_numbers', True),
                              start_line=chunk.start_line)
                
                panel = Panel(syntax, title=title, subtitle=subtitle, border_style="blue")
            except Exception:
                # Fall back to plain text if syntax highlighting fails
                panel = Panel(chunk.content, title=title, subtitle=subtitle, border_style="blue")
        else:
            panel = Panel(chunk.content, title=title, subtitle=subtitle, border_style="blue")
        
        self.console.print(panel)
        self.console.print()
    
    def _format_plain(self, query: str, results: List[SearchResult], 
                     explanation: Optional[str] = None) -> str:
        """Format results as plain text."""
        output = []
        output.append(f"Query: {query}")
        output.append(f"Found {len(results)} results\n")
        
        if explanation:
            output.append("AI Explanation:")
            output.append(explanation)
            output.append("")
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            output.append(f"#{i} | {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
            if chunk.name:
                output.append(f"Name: {chunk.name} ({chunk.chunk_type})")
            output.append(f"Score: {result.score:.2f}")
            output.append("Code:")
            output.append(chunk.content)
            output.append("-" * 80)
            output.append("")
        
        return "\n".join(output)
    
    def _format_json(self, query: str, results: List[SearchResult], 
                    explanation: Optional[str] = None) -> str:
        """Format results as JSON."""
        data = {
            "query": query,
            "explanation": explanation,
            "results": []
        }
        
        for result in results:
            chunk = result.chunk
            data["results"].append({
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "content": chunk.content,
                "score": result.score,
                "match_type": result.match_type
            })
        
        return json.dumps(data, indent=2)