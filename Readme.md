"""
# 🧠 interro - AI-Powered Code Understanding Tool

Ask natural language questions about your codebase and get intelligent answers!

## Features

- 🔍 **Smart Code Search**: Combines keyword and semantic search
- 🧠 **AI Explanations**: Optional LLM-powered code explanations via Ollama
- 🎨 **Beautiful Output**: Rich terminal formatting with syntax highlighting
- 🚀 **Fast Indexing**: Efficient AST-based parsing for Python, generic chunking for other languages
- ⚙️ **Configurable**: Extensive configuration options
- 📦 **Easy Install**: Simple pip installation

## Quick Start

```bash
# Install
pip install interro

# Ask questions about your code
interro ask "Where is the main function?"
interro ask "What handles user authentication?"
interro ask "Explain the database connection logic"

# Index a specific directory
interro ask "How does error handling work?" --path ./src

# Use AI explanations (requires Ollama)
interro ask "What does this class do?" --llm --model llama3
```

## Installation

```bash
pip install interro
```

For development:

```bash
git clone <repo>
cd interro
poetry install
```

## Configuration

Create a `.interro.yaml` file in your project root:

```bash
interro config --init
```

This creates a default configuration you can customize:

```yaml
indexing:
  file_extensions: [".py", ".js", ".ts", ".java", ".cpp"]
  exclude_dirs: ["__pycache__", ".git", "node_modules"]
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  max_results: 10
  use_semantic_search: true
  similarity_threshold: 0.7

llm:
  enabled: false
  model: "llama3"
  max_tokens: 500

output:
  format: "rich"  # rich, plain, json
  highlight_syntax: true
  show_line_numbers: true
```

## Usage Examples

### Basic Questions
```bash
# Find specific functionality
interro ask "Where is data loaded from files?"
interro ask "Show me all the API endpoints"
interro ask "What handles authentication?"

# Understand code structure  
interro ask "Explain the main application flow"
interro ask "How are errors handled?"
interro ask "What external libraries are used?"
```

### Advanced Usage
```bash
# Use AI explanations
interro ask "Explain this algorithm" --llm

# Search specific directory
interro ask "Database queries" --path ./backend

# JSON output for tooling
interro ask "Find all classes" --format json

# Limit results
interro ask "HTTP handlers" --max-results 5
```

### Programmatic Usage
```python
from interro import Interro

# Initialize
interro = Interro()

# Index codebase
interro.index_path("./my_project")

# Ask questions
result = interro.ask("Where is the config loaded?")
print(f"Found {len(result['results'])} matches")

for match in result['results']:
    print(f"{match.chunk.file_path}:{match.chunk.start_line}")
    print(match.chunk.content)
```

## LLM Integration

Interro supports local LLM explanations via [Ollama](https://ollama.ai/):

1. Install Ollama
2. Pull a model: `ollama pull llama3`
3. Enable in config or use `--llm` flag

```bash
# Enable LLM explanations
interro ask "Explain this function" --llm --model llama3
```

## Supported Languages

- **Python**: Full AST parsing for functions, classes, imports
- **JavaScript/TypeScript**: Generic chunking with smart boundaries  
- **Java, C++, C**: Generic chunking
- **Others**: Basic text chunking

## Architecture

```
interro/
├── cli.py           # Command-line interface
├── indexer.py       # Code parsing and chunking
├── retriever.py     # Search (keyword + semantic)
├── llm_agent.py     # LLM integration via Ollama
├── formatter.py     # Output formatting
└── config.py        # Configuration management
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## FAQ

**Q: How does semantic search work?**
A: Uses sentence-transformers to create embeddings of code chunks, enabling similarity-based matching beyond keywords.

**Q: Can I use it without LLM features?**
A: Yes! Keyword and semantic search work without any LLM integration.

**Q: What LLM models are supported?**
A: Any model available through Ollama (llama3, gemma, phi3, etc.)

**Q: How large codebases can it handle?**
A: Tested on codebases up to 100k+ lines. Uses efficient indexing and configurable limits.
"""