import re
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .indexer import CodeChunk, IndexedFile
import numpy as np
import subprocess
import json
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
import os

@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    match_type: str  # 'keyword', 'semantic', 'hybrid'
    highlights: List[Tuple[int, int]] = None

class CodeRetriever:
    def __init__(self, config, indexed_files: List[IndexedFile]):
        self.config = config
        self.indexed_files = indexed_files
        self.all_chunks = []

        for file in indexed_files:
            self.all_chunks.extend(file.chunks)

        self.embeddings = None
        self.embedding_cache_file = ".embeddings_cache.pkl"
        
        # Fast initialization - only do embeddings if explicitly enabled and not cached
        if config.get('retrieval.use_semantic_search', True):
            self._initialize_semantic_search_fast()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_embedding_cache(self) -> dict:
        """Load cached embeddings"""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def _save_embedding_cache(self, cache: dict):
        """Save embeddings to cache"""
        try:
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")

    def _embed_text_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts for efficiency"""
        if not texts:
            return []
            
        try:
            # For now, we'll use individual requests but with connection pooling
            # In future, could implement true batch API if Ollama supports it
            session = requests.Session()
            session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=10))
            
            embeddings = []
            for text in texts:
                clean_text = text.strip()[:1500]  # Smaller limit for speed
                if not clean_text:
                    embeddings.append([])
                    continue
                    
                response = session.post(
                    'http://localhost:11434/api/embeddings',
                    json={
                        'model': 'nomic-embed-text',
                        'prompt': clean_text
                    },
                    timeout=5  # Shorter timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings.append(result.get('embedding', []))
                else:
                    embeddings.append([])
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Batch embedding failed: {e}")
            return [[] for _ in texts]

    def _embed_text_parallel(self, texts: List[str], max_workers: int = 4) -> List[List[float]]:
        """Parallel embedding with thread pool"""
        def embed_single(text):
            try:
                clean_text = text.strip()[:1500]
                if not clean_text:
                    return []
                    
                response = requests.post(
                    'http://localhost:11434/api/embeddings',
                    json={
                        'model': 'nomic-embed-text',
                        'prompt': clean_text
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('embedding', [])
                return []
            except:
                return []
        
        embeddings = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(embed_single, text): i 
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embeddings[index] = future.result()
                except:
                    embeddings[index] = []
        
        return embeddings

    def _initialize_semantic_search_fast(self):
        """Fast initialization with caching and batching"""
        print("🚀 Fast semantic search initialization...")
        
        # Check if we should skip semantic search for speed
        if self.config.get('retrieval.skip_semantic_for_speed', False):
            print("⚡ Skipping semantic search for maximum speed")
            return
        
        # Test Ollama connection quickly
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code != 200:
                print("⚡ Ollama not responding - using keyword-only search for speed")
                return
        except:
            print("⚡ Cannot connect to Ollama - using keyword-only search")
            return
        
        # Load existing cache
        cache = self._load_embedding_cache()
        print(f"📦 Loaded {len(cache)} cached embeddings")
        
        # Identify chunks that need embedding
        texts_to_embed = []
        chunk_indices = []
        
        for i, chunk in enumerate(self.all_chunks):
            if chunk.embedding:
                continue  # Already has embedding
                
            embed_text = self._create_embed_text(chunk)
            cache_key = self._get_cache_key(embed_text)
            
            if cache_key in cache:
                chunk.embedding = cache[cache_key]
            else:
                texts_to_embed.append(embed_text)
                chunk_indices.append(i)
        
        print(f"🔄 Need to embed {len(texts_to_embed)} new chunks")
        
        if texts_to_embed:
            # Batch/parallel embedding for speed
            if len(texts_to_embed) > 10:
                print("⚡ Using parallel embedding...")
                new_embeddings = self._embed_text_parallel(texts_to_embed, max_workers=4)
            else:
                print("⚡ Using batch embedding...")
                new_embeddings = self._embed_text_batch(texts_to_embed)
            
            # Update chunks and cache
            cache_updated = False
            for i, embedding in enumerate(new_embeddings):
                if embedding:
                    chunk_idx = chunk_indices[i]
                    self.all_chunks[chunk_idx].embedding = embedding
                    
                    # Update cache
                    embed_text = texts_to_embed[i]
                    cache_key = self._get_cache_key(embed_text)
                    cache[cache_key] = embedding
                    cache_updated = True
            
            if cache_updated:
                self._save_embedding_cache(cache)
                print("💾 Updated embedding cache")
        
        # Build embedding matrix
        valid_embeddings = []
        self.semantic_chunks = []
        
        for chunk in self.all_chunks:
            if chunk.embedding and len(chunk.embedding) > 0:
                valid_embeddings.append(chunk.embedding)
                self.semantic_chunks.append(chunk)
        
        if valid_embeddings:
            self.embeddings = np.array(valid_embeddings, dtype=np.float32)
            print(f"✅ Ready with {len(valid_embeddings)} embeddings")
        else:
            print("⚡ No embeddings available - using keyword-only search")
            self.embeddings = None

    def _create_embed_text(self, chunk: CodeChunk) -> str:
        """Create concise text for embedding"""
        parts = []
        
        # Prioritize most important info
        if chunk.name:
            parts.append(f"{chunk.name}")
        
        # Add key content lines (not full content for speed)
        lines = chunk.content.strip().split('\n')
        if len(lines) > 10:
            # Take first 5 and last 2 lines for functions/classes
            key_lines = lines[:5] + lines[-2:]
        else:
            key_lines = lines
        
        parts.append('\n'.join(key_lines))
        
        return '\n'.join(parts)

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Fast search implementation"""
        if max_results is None:
            max_results = self.config.get('retrieval.max_results', 10)

        # Always do keyword search (it's fast)
        keyword_results = self._enhanced_keyword_search(query)
        
        # Only do semantic search if embeddings are ready and it's worth it
        semantic_results = []
        if (self.embeddings is not None and 
            len(self.embeddings) > 0 and 
            len(keyword_results) < max_results * 2):  # Only if keyword search didn't find enough
            semantic_results = self._semantic_search_fast(query)

        # Quick result combination
        if semantic_results and keyword_results:
            results = self._combine_results_fast(keyword_results, semantic_results)
        elif keyword_results:
            results = keyword_results
        else:
            results = semantic_results

        # Fast filtering and sorting
        if results:
            # Quick score-based filter
            min_score = max(0.1, max(r.score for r in results) * 0.1)
            results = [r for r in results if r.score >= min_score]
            
            # Sort and limit
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_results]
        
        return []

    def _semantic_search_fast(self, query: str) -> List[SearchResult]:
        """Fast semantic search with optimizations"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        query_embedding = self._embed_text_simple(query)
        if not query_embedding:
            return []

        try:
            # Fast vectorized cosine similarity
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Normalize vectors for faster cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return []
            query_embedding = query_embedding / query_norm
            
            # Normalize embedding matrix (do this once and cache if needed)
            embedding_norms = np.linalg.norm(self.embeddings, axis=1)
            valid_indices = embedding_norms > 0
            
            if not np.any(valid_indices):
                return []
            
            normalized_embeddings = self.embeddings[valid_indices]
            normalized_embeddings = normalized_embeddings / embedding_norms[valid_indices].reshape(-1, 1)
            
            # Fast dot product for cosine similarity
            similarities = np.dot(normalized_embeddings, query_embedding)
            
            # Quick threshold filter
            threshold = 0.15  # Lower threshold for speed
            good_indices = similarities >= threshold
            
            results = []
            valid_chunk_indices = np.where(valid_indices)[0]
            
            for i, sim in enumerate(similarities):
                if good_indices[i]:
                    original_idx = valid_chunk_indices[i]
                    results.append(SearchResult(
                        chunk=self.semantic_chunks[original_idx],
                        score=float(sim),
                        match_type="semantic"
                    ))
            
            return results
            
        except Exception as e:
            print(f"❌ Fast semantic search failed: {e}")
            return []

    def _embed_text_simple(self, text: str) -> List[float]:
        """Simple single text embedding"""
        try:
            clean_text = text.strip()[:1000]
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': 'nomic-embed-text',
                    'prompt': clean_text
                },
                timeout=3
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('embedding', [])
            return []
        except:
            return []

    def _enhanced_keyword_search(self, query: str) -> List[SearchResult]:
        """Optimized keyword search"""
        query_lower = query.lower()
        key_terms = self._extract_key_terms_fast(query_lower)
        
        if not key_terms:
            return []
        
        results = []
        for chunk in self.all_chunks:
            score = self._calculate_smart_score_fast(chunk, query_lower, key_terms)
            if score > 0.5:  # Higher threshold for speed
                results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    match_type="keyword",
                    highlights=[]  # Skip highlights for speed
                ))
        
        return results

    def _extract_key_terms_fast(self, query: str) -> List[str]:
        """Fast term extraction"""
        # Simple but effective approach
        stop_words = {'is', 'the', 'a', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'where', 'what', 'how'}
        words = re.findall(r'\b\w{3,}\b', query)  # Only words 3+ chars
        return [w for w in words if w not in stop_words]

    def _calculate_smart_score_fast(self, chunk: CodeChunk, query: str, key_terms: List[str]) -> float:
        """Fast scoring algorithm"""
        content_lower = chunk.content.lower()
        name_lower = (chunk.name or "").lower()
        
        score = 0.0
        
        # Quick exact matches
        if query in content_lower:
            score += 20
        
        # Function/class name matches (most important)
        for term in key_terms:
            if term in name_lower:
                score += 15
        
        # Quick content scan (first 500 chars only)
        content_start = content_lower[:500]
        for term in key_terms:
            score += content_start.count(term) * 2
        
        # File name bonus
        if key_terms:
            file_name = chunk.file_path.lower().split('/')[-1]
            for term in key_terms:
                if term in file_name:
                    score += 5
        
        return score

    def _combine_results_fast(self, keyword_results: List[SearchResult], 
                             semantic_results: List[SearchResult]) -> List[SearchResult]:
        """Fast result combination"""
        # Simple combination - just merge and let sorting handle it
        kw_weight = 0.7
        sem_weight = 0.3
        
        # Create quick lookup
        sem_dict = {(r.chunk.file_path, r.chunk.start_line): r.score for r in semantic_results}
        
        combined = []
        seen = set()
        
        # Process keyword results
        for kr in keyword_results:
            key = (kr.chunk.file_path, kr.chunk.start_line)
            sem_score = sem_dict.get(key, 0)
            
            final_score = kw_weight * kr.score + sem_weight * sem_score
            combined.append(SearchResult(
                chunk=kr.chunk,
                score=final_score,
                match_type="hybrid" if sem_score > 0 else "keyword",
                highlights=kr.highlights
            ))
            seen.add(key)
        
        # Add remaining semantic results
        for sr in semantic_results:
            key = (sr.chunk.file_path, sr.chunk.start_line)
            if key not in seen:
                combined.append(SearchResult(
                    chunk=sr.chunk,
                    score=sr.score * sem_weight,
                    match_type="semantic"
                ))
        
        return combined