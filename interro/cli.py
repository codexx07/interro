import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .indexer import CodeIndexer
from .retriever import CodeRetriever
from .llm_agent import LLMAgent
from .formatter import ResultFormatter

app = typer.Typer(help="🧠 interro - AI-powered code understanding tool")
console = Console()

@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask about the codebase"),
    path: str = typer.Option(".", "--path", "-p", help="Path to analyze"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
    max_results: Optional[int] = typer.Option(None, "--max-results", "-n", help="Maximum results to show"),
    format_type: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (rich/plain/json)"),
    use_llm: bool = typer.Option(False, "--llm", help="Enable LLM explanations"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use")
):
    """Ask a question about your codebase."""
    
    # Load configuration
    config = Config(config_file)
    
    # Override config with CLI options
    if max_results:
        config.config['retrieval']['max_results'] = max_results
    if format_type:
        config.config['output']['format'] = format_type
    if use_llm:
        config.config['llm']['enabled'] = True
    if model:
        config.config['llm']['model'] = model
    
    # Check if path exists
    if not os.path.exists(path):
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        sys.exit(1)
    
    try:
        # Index the codebase
        console.print(f"[blue]Indexing codebase at '{path}'...[/blue]")
        indexer = CodeIndexer(config)
        
        if os.path.isfile(path):
            indexed_files = [indexer.index_file(path)]
            indexed_files = [f for f in indexed_files if f is not None]
        else:
            indexed_files = indexer.index_directory(path)
        
        if not indexed_files:
            console.print("[yellow]No files found to index[/yellow]")
            sys.exit(0)
        
        total_chunks = sum(len(f.chunks) for f in indexed_files)
        console.print(f"[green]Indexed {len(indexed_files)} files with {total_chunks} code chunks[/green]")
        
        # Search for relevant code
        console.print(f"[blue]Searching for: '{query}'...[/blue]")
        retriever = CodeRetriever(config, indexed_files)
        results = retriever.search(query)
        
        if not results:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            sys.exit(0)
        
        # Get LLM explanation if enabled
        explanation = None
        if config.get('llm.enabled', False):
            console.print("[blue]Generating AI explanation...[/blue]")
            llm_agent = LLMAgent(config)
            explanation = llm_agent.explain_results(query, results)
        
        # Format and display results
        formatter = ResultFormatter(config)
        output = formatter.format_results(query, results, explanation)
        
        if output:  # For non-rich formats
            console.print(output)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@app.command()
def index(
    path: str = typer.Option(".", "--path", "-p", help="Path to index"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save index to file")
):
    """Index a codebase and optionally save the index."""
    
    config = Config(config_file)
    
    if not os.path.exists(path):
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        sys.exit(1)
    
    try:
        console.print(f"[blue]Indexing codebase at '{path}'...[/blue]")
        indexer = CodeIndexer(config)
        
        if os.path.isfile(path):
            indexed_files = [indexer.index_file(path)]
            indexed_files = [f for f in indexed_files if f is not None]
        else:
            indexed_files = indexer.index_directory(path)
        
        total_chunks = sum(len(f.chunks) for f in indexed_files)
        console.print(f"[green]Successfully indexed {len(indexed_files)} files with {total_chunks} code chunks[/green]")
        
        # Show summary
        for file in indexed_files[:10]:  # Show first 10 files
            console.print(f"  {file.path}: {len(file.chunks)} chunks ({file.language})")
        
        if len(indexed_files) > 10:
            console.print(f"  ... and {len(indexed_files) - 10} more files")
        
        if output:
            # Save index to file (you could implement serialization here)
            console.print(f"[blue]Saving index to '{output}'...[/blue]")
            # TODO: Implement index serialization
            console.print("[yellow]Index saving not yet implemented[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    init: bool = typer.Option(False, "--init", help="Initialize default config file")
):
    """Manage configuration."""
    
    if init:
        config_path = Path(".interro.yaml")
        if config_path.exists():
            overwrite = typer.confirm(f"Config file {config_path} already exists. Overwrite?")
            if not overwrite:
                console.print("[yellow]Config initialization cancelled[/yellow]")
                return
        
        # Create default config
        import yaml
        from .config import DEFAULT_CONFIG
        
        with open(config_path, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, indent=2)
        
        console.print(f"[green]Default config created at {config_path}[/green]")
        return
    
    if show:
        config = Config()
        console.print("[blue]Current Configuration:[/blue]")
        
        import yaml
        config_str = yaml.dump(config.config, default_flow_style=False, indent=2)
        console.print(config_str)
        return
    
    console.print("Use --show to display config or --init to create default config")

@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"interro version {__version__}")

if __name__ == "__main__":
    app()