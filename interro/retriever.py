import re
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .indexer import CodeChunk, IndexedFile

@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    match_type: str  # 'keyword', 'semantic', 'hybrid'
    highlights: List[Tuple[int, int]] = None  # character positions for highlighting

class CodeRetriever:
    def __init__(self, config, indexed_files: List[IndexedFile]):
        self.config = config
        self.indexed_files = indexed_files
        self.all_chunks = []
        
        # Flatten all chunks
        for file in indexed_files:
            self.all_chunks.extend(file.chunks)
        
        # Initialize semantic search if enabled
        self.embeddings = None
        if config.get('retrieval.use_semantic_search', True):
            self._initialize_semantic_search()
    
    def _initialize_semantic_search(self):
        """Initialize semantic search using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            print("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create embeddings for all chunks
            chunk_texts = [chunk.content for chunk in self.all_chunks]
            self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)
            print(f"Created embeddings for {len(self.all_chunks)} code chunks")
            
        except ImportError:
            print("sentence-transformers not available, using keyword search only")
            self.model = None
            self.embeddings = None
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Search for code chunks matching the query."""
        if max_results is None:
            max_results = self.config.get('retrieval.max_results', 10)
        
        results = []
        
        # Keyword search
        keyword_results = self._keyword_search(query)
        
        # Semantic search (if available)
        semantic_results = []
        if self.embeddings is not None and self.model is not None:
            semantic_results = self._semantic_search(query)
        
        # Combine results
        if semantic_results and keyword_results:
            results = self._combine_results(keyword_results, semantic_results)
        elif semantic_results:
            results = semantic_results
        else:
            results = keyword_results
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _keyword_search(self, query: str) -> List[SearchResult]:
        """Perform keyword-based search."""
        results = []
        query_terms = query.lower().split()
        
        for chunk in self.all_chunks:
            score = self._calculate_keyword_score(chunk, query_terms)
            if score > 0:
                highlights = self._find_highlights(chunk.content, query_terms)
                results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    match_type='keyword',
                    highlights=highlights
                ))
        
        return results
    
    def _semantic_search(self, query: str) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        if self.model is None or self.embeddings is None:
            return []
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Embed the query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        results = []
        threshold = self.config.get('retrieval.similarity_threshold', 0.7)
        
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append(SearchResult(
                    chunk=self.all_chunks[i],
                    score=float(similarity),
                    match_type='semantic'
                ))
        
        return results
    
    def _combine_results(self, keyword_results: List[SearchResult], 
                        semantic_results: List[SearchResult]) -> List[SearchResult]:
        """Combine keyword and semantic search results."""
        keyword_weight = self.config.get('retrieval.keyword_weight', 0.3)
        semantic_weight = self.config.get('retrieval.semantic_weight', 0.7)
        
        # Create lookup for semantic scores
        semantic_lookup = {(r.chunk.file_path, r.chunk.start_line): r.score 
                          for r in semantic_results}
        
        combined_results = []
        processed_chunks = set()
        
        # Process keyword results
        for kr in keyword_results:
            key = (kr.chunk.file_path, kr.chunk.start_line)
            semantic_score = semantic_lookup.get(key, 0)
            
            combined_score = (keyword_weight * kr.score + 
                            semantic_weight * semantic_score)
            
            combined_results.append(SearchResult(
                chunk=kr.chunk,
                score=combined_score,
                match_type='hybrid',
                highlights=kr.highlights
            ))
            processed_chunks.add(key)
        
        # Add remaining semantic results
        for sr in semantic_results:
            key = (sr.chunk.file_path, sr.chunk.start_line)
            if key not in processed_chunks:
                combined_results.append(SearchResult(
                    chunk=sr.chunk,
                    score=semantic_weight * sr.score,
                    match_type='semantic'
                ))
        
        return combined_results
    
    def _calculate_keyword_score(self, chunk: CodeChunk, query_terms: List[str]) -> float:
        """Calculate keyword matching score for a chunk."""
        content_lower = chunk.content.lower()
        name_lower = (chunk.name or '').lower()
        
        score = 0
        for term in query_terms:
            # Count occurrences in content
            content_matches = len(re.findall(re.escape(term), content_lower))
            # Boost for matches in function/class names
            name_matches = len(re.findall(re.escape(term), name_lower)) * 2
            
            term_score = content_matches + name_matches
            score += term_score
        
        # Normalize by content length
        if score > 0:
            score = score / math.log(len(chunk.content) + 1)
        
        return score
    
    def _find_highlights(self, content: str, query_terms: List[str]) -> List[Tuple[int, int]]:
        """Find character positions of query terms for highlighting."""
        highlights = []
        content_lower = content.lower()
        
        for term in query_terms:
            start = 0
            while True:
                pos = content_lower.find(term, start)
                if pos == -1:
                    break
                highlights.append((pos, pos + len(term)))
                start = pos + 1
        
        return sorted(highlights)