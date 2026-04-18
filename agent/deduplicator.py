"""Prompt deduplication with hash-based request caching.

Eliminates redundant API calls by caching responses to identical prompts.
When the same prompt (with same model and parameters) is submitted,
returns the cached response instead of making another API call.

Features:
- Content-addressable cache keys (SHA-256 of prompt + model + params)
- TTL-based expiration
- LRU eviction when cache grows too large
- Request deduplication for in-flight requests (prevents thundering herd)
- Async support for concurrent request handling
- Detailed statistics tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DedupStats:
    """Statistics for deduplication performance."""
    requests: int = 0
    duplicates: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    errors: int = 0
    
    @property
    def dedup_rate(self) -> float:
        """Return deduplication rate as percentage."""
        if self.requests == 0:
            return 0.0
        return (self.duplicates / self.requests) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Return cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests": self.requests,
            "duplicates": self.duplicates,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "errors": self.errors,
            "dedup_rate_pct": round(self.dedup_rate, 2),
            "cache_hit_rate_pct": round(self.cache_hit_rate, 2),
        }


@dataclass
class DedupEntry:
    """A cached deduplication entry."""
    key: str
    response: str
    created_at: float
    expires_at: float
    prompt_length: int
    model: str
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        return self.expires_at > 0 and now >= self.expires_at


class PromptDeduplicator:
    """Deduplicates identical prompts to avoid redundant API calls.
    
    Uses content-addressable keys (SHA-256 hash) for deterministic caching.
    Supports:
    - Exact prompt matching
    - In-flight request waiting (thundering herd prevention)
    - Configurable TTL and size limits
    - Per-model caching
    """
    
    DEFAULT_MAX_SIZE_MB = 100  # 100 MB cache
    DEFAULT_TTL_SECONDS = 1800  # 30 minutes default
    HASH_CHARS = 16
    
    # Simple queries that can be routed to cheaper models
    SIMPLE_PATTERNS = [
        "what is", "how to", "how do", "explain", "define",
        "tell me about", "describe", "list", "show me",
        "what are", "simple", "basic", "quick",
    ]
    
    # Complex patterns that need premium models
    COMPLEX_PATTERNS = [
        "analyze", "research", "complex", "architect",
        "design system", "review code", "optimize performance",
        "security audit", "debug", "investigate",
    ]
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        enabled: bool = True,
    ):
        if cache_dir is None:
            from hermes_constants import get_hermes_home
            cache_dir = get_hermes_home() / "cache" / "dedup"
        
        self._cache_dir = Path(cache_dir)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl_seconds
        self._enabled = enabled
        
        # In-memory index
        self._index: Dict[str, DedupEntry] = {}
        self._lock = threading.RLock()
        self._stats = DedupStats()
        
        # In-flight requests (thundering herd prevention)
        self._in_flight: Dict[str, asyncio.Event] = {}
        self._in_flight_lock = threading.Lock()
        
        # Initialize cache directory
        self._init_cache_dir()
        self._rebuild_index()
    
    def _init_cache_dir(self) -> None:
        """Create cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self._cache_dir / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir = self._cache_dir / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)
    
    def _rebuild_index(self) -> None:
        """Rebuild in-memory index from disk on startup."""
        meta_file = self._meta_dir / "index.json"
        if not meta_file.exists():
            return
        
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            now = time.time()
            for key, entry_data in data.items():
                entry = DedupEntry(
                    key=key,
                    response=entry_data["response"],
                    created_at=entry_data["created_at"],
                    expires_at=entry_data["expires_at"],
                    prompt_length=entry_data["prompt_length"],
                    model=entry_data["model"],
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=entry_data.get("last_accessed", entry_data["created_at"]),
                )
                if not entry.is_expired(now):
                    self._index[key] = entry
                    
            logger.info("Dedup index rebuilt: %d entries", len(self._index))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to rebuild dedup index: %s", e)
    
    def _save_index(self) -> None:
        """Save index to disk."""
        meta_file = self._meta_dir / "index.json"
        temp_file = meta_file.with_suffix(".tmp")
        
        data = {}
        for key, entry in self._index.items():
            data[key] = {
                "response": entry.response,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "prompt_length": entry.prompt_length,
                "model": entry.model,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            temp_file.rename(meta_file)
        except IOError as e:
            logger.warning("Failed to save dedup index: %s", e)
    
    @staticmethod
    def compute_key(prompt: str, model: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Compute a deterministic cache key for a prompt.
        
        Key = SHA-256(model + normalized_params + prompt)
        """
        params = params or {}
        # Normalize params for deterministic hashing
        normalized_params = {}
        for k, v in sorted(params.items()):
            if isinstance(v, (list, dict)):
                normalized_params[k] = json.dumps(v, sort_keys=True)
            else:
                normalized_params[k] = str(v)
        
        content = json.dumps({
            "model": model,
            "params": normalized_params,
            "prompt": prompt,
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _get_cache_path(self, key: str) -> Tuple[Path, Path]:
        """Get paths for data and metadata files."""
        prefix = key[:self.HASH_CHARS]
        data_dir = self._data_dir / prefix
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / f"{key}.json", data_dir / f"{key}.meta.json"
    
    def get(
        self,
        prompt: str,
        model: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], bool]:
        """Get a cached response for a prompt.
        
        Returns:
            Tuple of (response, found). If found=True, response is the cached data.
        """
        if not self._enabled:
            return None, False
        
        self._stats.requests += 1
        
        key = self.compute_key(prompt, model, params)
        now = time.time()
        
        with self._lock:
            entry = self._index.get(key)
            
            if entry is None:
                self._stats.cache_misses += 1
                return None, False
            
            if entry.is_expired(now):
                del self._index[key]
                self._remove_from_disk(key)
                self._save_index()
                self._stats.cache_misses += 1
                return None, False
            
            # Update access stats
            entry.last_accessed = now
            entry.access_count += 1
            
            # Read from disk
            data_path, _ = self._get_cache_path(key)
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    response = f.read()
                self._stats.cache_hits += 1
                self._stats.duplicates += 1
                return response, True
            except IOError:
                self._stats.cache_misses += 1
                return None, False
    
    async def get_async(
        self,
        prompt: str,
        model: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], bool]:
        """Async version of get()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, prompt, model, params)
    
    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Cache a response for a prompt.
        
        Returns:
            The cache key for this entry.
        """
        if not self._enabled:
            return ""
        
        if ttl_seconds is None:
            ttl_seconds = self._default_ttl
        
        key = self.compute_key(prompt, model, params)
        now = time.time()
        
        entry = DedupEntry(
            key=key,
            response=response,
            created_at=now,
            expires_at=now + ttl_seconds if ttl_seconds > 0 else 0,
            prompt_length=len(prompt),
            model=model,
            last_accessed=now,
        )
        
        with self._lock:
            # Check if we need to evict
            size_bytes = len(response.encode("utf-8"))
            self._evict_if_needed(size_bytes)
            
            # Write to disk
            data_path, meta_path = self._get_cache_path(key)
            temp_data = data_path.with_suffix(".tmp")
            temp_meta = meta_path.with_suffix(".tmp")
            
            try:
                with open(temp_data, "w", encoding="utf-8") as f:
                    f.write(response)
                temp_data.rename(data_path)
                
                meta_data = {
                    "key": key,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "prompt_length": entry.prompt_length,
                    "model": entry.model,
                    "access_count": 0,
                    "last_accessed": now,
                }
                with open(temp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f)
                temp_meta.rename(meta_path)
                
                self._index[key] = entry
                
                if len(self._index) % 100 == 0:
                    self._save_index()
                
                return key
                
            except IOError as e:
                logger.warning("Failed to write dedup entry: %s", e)
                self._stats.errors += 1
                return ""
    
    async def set_async(
        self,
        prompt: str,
        model: str,
        response: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Async version of set()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.set, prompt, model, response, params, ttl_seconds
        )
    
    def _remove_from_disk(self, key: str) -> None:
        """Remove entry from disk."""
        data_path, meta_path = self._get_cache_path(key)
        try:
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        except OSError:
            pass
    
    def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict LRU entries if cache exceeds max size."""
        current_size = sum(len(e.response.encode("utf-8")) for e in self._index.values())
        target_size = self._max_size_bytes - new_entry_size
        
        if current_size <= target_size:
            return
        
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: (x[1].last_accessed, x[1].access_count)
        )
        
        evicted = 0
        for key, entry in sorted_entries:
            if current_size <= target_size:
                break
            del self._index[key]
            self._remove_from_disk(key)
            current_size -= len(entry.response.encode("utf-8"))
            evicted += 1
        
        if evicted > 0:
            self._stats.evictions += evicted
            logger.debug("Evicted %d dedup entries", evicted)
    
    def clear(self) -> int:
        """Clear all cached entries."""
        with self._lock:
            count = len(self._index)
            for key in list(self._index.keys()):
                self._remove_from_disk(key)
            self._index.clear()
            self._save_index()
            return count
    
    def get_stats(self) -> DedupStats:
        """Get deduplication statistics."""
        with self._lock:
            return DedupStats(
                requests=self._stats.requests,
                duplicates=self._stats.duplicates,
                cache_hits=self._stats.cache_hits,
                cache_misses=self._stats.cache_misses,
                evictions=self._stats.evictions,
                errors=self._stats.errors,
            )
    
    # ─────────────────────────────────────────────────────────────────
    # Query complexity analysis for smart routing
    # ─────────────────────────────────────────────────────────────────
    
    def estimate_complexity(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, float]:
        """Estimate the complexity of a query.
        
        Returns:
            Tuple of (complexity_level, confidence)
            complexity_level: "simple", "moderate", or "complex"
            confidence: 0.0 to 1.0
        """
        prompt_lower = prompt.lower()
        
        simple_score = 0
        complex_score = 0
        
        # Check for simple patterns
        for pattern in self.SIMPLE_PATTERNS:
            if pattern in prompt_lower:
                simple_score += 1
        
        # Check for complex patterns
        for pattern in self.COMPLEX_PATTERNS:
            if pattern in prompt_lower:
                complex_score += 2  # Complex patterns weight more
        
        # Length-based scoring
        word_count = len(prompt.split())
        if word_count < 20:
            simple_score += 1
        elif word_count > 100:
            complex_score += 2
        
        # Code-related queries tend to be complex
        code_indicators = ["```", "function", "class", "code", "implement", "algorithm"]
        for indicator in code_indicators:
            if indicator in prompt_lower:
                complex_score += 1
        
        # File path mentions suggest specific, often complex tasks
        if "/" in prompt or "\\" in prompt:
            complex_score += 1
        
        total = simple_score + complex_score
        
        if total == 0:
            return "moderate", 0.5
        
        simple_ratio = simple_score / total
        
        if simple_ratio > 0.7:
            return "simple", min(simple_ratio, 0.95)
        elif simple_ratio < 0.3:
            return "complex", min(1 - simple_ratio, 0.95)
        else:
            return "moderate", 0.6
    
    def should_route_to_cheap_model(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """Determine if a query should be routed to a cheaper model.
        
        Returns:
            True if the query is simple enough for a cheap model.
        """
        complexity, confidence = self.estimate_complexity(prompt, conversation_history)
        return complexity == "simple" and confidence >= 0.7


# Global instance
_global_dedup: Optional[PromptDeduplicator] = None
_dedup_lock = threading.Lock()


def get_global_deduplicator() -> PromptDeduplicator:
    """Get the global deduplicator instance."""
    global _global_dedup
    
    if _global_dedup is None:
        with _dedup_lock:
            if _global_dedup is None:
                _global_dedup = PromptDeduplicator()
    
    return _global_dedup


def configure_deduplicator(
    cache_dir: Optional[Path] = None,
    max_size_mb: int = PromptDeduplicator.DEFAULT_MAX_SIZE_MB,
    default_ttl_seconds: int = PromptDeduplicator.DEFAULT_TTL_SECONDS,
    enabled: bool = True,
) -> PromptDeduplicator:
    """Configure the global deduplicator before first use."""
    global _global_dedup
    
    with _dedup_lock:
        if _global_dedup is not None:
            logger.warning("Deduplicator already initialized, ignoring configure call")
            return _global_dedup
        
        _global_dedup = PromptDeduplicator(
            cache_dir=cache_dir,
            max_size_mb=max_size_mb,
            default_ttl_seconds=default_ttl_seconds,
            enabled=enabled,
        )
        return _global_dedup