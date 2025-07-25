import ast
import os
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import tiktoken

@dataclass
class CodeChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'import', 'block'
    name: Optional[str] = None
    embedding: Optional[List[float]] = None  # ✅ For semantic search

@dataclass
class IndexedFile:
    path: str
    chunks: List[CodeChunk]
    total_lines: int
    language: str

class CodeIndexer:
    def __init__(self, config):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def index_directory(self, directory: str) -> List[IndexedFile]:
        indexed_files = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in self.config.get('indexing.exclude_dirs', [])]
            for file in files:
                if self._should_index_file(file):
                    file_path = os.path.join(root, file)
                    try:
                        indexed_file = self.index_file(file_path)
                        if indexed_file:
                            indexed_files.append(indexed_file)
                    except Exception as e:
                        print(f"Warning: Could not index {file_path}: {e}")
        return indexed_files

    def index_file(self, file_path: str) -> Optional[IndexedFile]:
        if not os.path.exists(file_path):
            return None
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config.get('indexing.max_file_size_mb', 5):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return None

        language = self._detect_language(file_path)
        chunks = self._create_chunks(content, file_path, language)

        return IndexedFile(
            path=file_path,
            chunks=chunks,
            total_lines=len(content.splitlines()),
            language=language
        )

    def _should_index_file(self, filename: str) -> bool:
        extensions = self.config.get('indexing.file_extensions', [])
        exclude_patterns = self.config.get('indexing.exclude_files', [])
        if not any(filename.endswith(ext) for ext in extensions):
            return False
        for pattern in exclude_patterns:
            if re.match(pattern.replace('*', '.*'), filename):
                return False
        return True

    def _detect_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp'
        }
        return language_map.get(ext, 'text')

    def _create_chunks(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        if language == 'python':
            return self._chunk_python_file(content, file_path)
        else:
            return self._chunk_generic_file(content, file_path)

    def _chunk_python_file(self, content: str, file_path: str) -> List[CodeChunk]:
        chunks = []
        try:
            tree = ast.parse(content)

            # Imports
            imports = [
                (node.lineno, ast.get_source_segment(content, node))
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            if imports:
                chunks.append(CodeChunk(
                    content='\n'.join(line for _, line in imports if line),
                    file_path=file_path,
                    start_line=imports[0][0],
                    end_line=imports[-1][0],
                    chunk_type='imports',
                    embedding=None
                ))

            # Functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    chunk_content = ast.get_source_segment(content, node)
                    if chunk_content:
                        chunks.append(CodeChunk(
                            content=chunk_content,
                            file_path=file_path,
                            start_line=node.lineno,
                            end_line=getattr(node, 'end_lineno', node.lineno),
                            chunk_type='function',
                            name=node.name,
                            embedding=None
                        ))
                elif isinstance(node, ast.ClassDef):
                    chunk_content = ast.get_source_segment(content, node)
                    if chunk_content:
                        chunks.append(CodeChunk(
                            content=chunk_content,
                            file_path=file_path,
                            start_line=node.lineno,
                            end_line=getattr(node, 'end_lineno', node.lineno),
                            chunk_type='class',
                            name=node.name,
                            embedding=None
                        ))

        except SyntaxError:
            return self._chunk_generic_file(content, file_path)

        return chunks or self._chunk_generic_file(content, file_path)

    def _chunk_generic_file(self, content: str, file_path: str) -> List[CodeChunk]:
        lines = content.splitlines()
        chunks = []
        chunk_size = self.config.get('indexing.chunk_size', 1000)
        chunk_overlap = self.config.get('indexing.chunk_overlap', 200)

        start_line = 1
        while start_line <= len(lines):
            end_line = min(start_line + chunk_size - 1, len(lines))
            chunk_lines = lines[start_line - 1:end_line]
            chunk_content = '\n'.join(chunk_lines)
            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type='block',
                    embedding=None
                ))
            start_line = end_line - chunk_overlap + 1
            if end_line >= len(lines):
                break

        return chunks
