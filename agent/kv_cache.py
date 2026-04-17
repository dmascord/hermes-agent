"""Persistent disk-based KV cache with content-addressable storage.

Reduces API costs and improves response times by caching LLM responses
and tool results. Uses SHA-256 content hashing for deterministic cache keys.

Features:
- Content-addressable storage (hash of prompt + model + params)
- TTL support with automatic expiration
- LRU eviction when max_size is exceeded
- Atomic writes (write to temp file, then rename)
- Thread-safe operation
- Cache statistics tracking (hits, misses, evictions)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writes: int = 0
    errors: int = 0
    total_latency_saved_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "writes": self.writes,
            "errors": self.errors,
            "hit_rate_pct": round(self.hit_rate, 2),
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 2),
        }


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: str
    created_at: float
    expires_at: float
    size_bytes: int
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        return self.expires_at > 0 and now >= self.expires_at
    
    def touch(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        self.last_accessed = now
        self.access_count += 1


class PersistentKVCache:
    """Thread-safe persistent KV cache with TTL and LRU eviction.
    
    Cache key is deterministic: SHA-256(prompt + model + json(params)).
    This means identical requests will always hit the cache.
    """
    
    DEFAULT_MAX_SIZE_MB = 500  # 500 MB default max cache size
    DEFAULT_TTL_SECONDS = 3600  # 1 hour default TTL
    HASH_CHARS = 16  # Use first 16 chars of hash for directory splitting
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        _test_mode: bool = False,
    ):
        if cache_dir is None:
            from hermes_constants import get_hermes_home
            cache_dir = get_hermes_home() / "cache" / "kv"
        
        self._cache_dir = Path(cache_dir)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl_seconds
        self._test_mode = _test_mode
        
        # In-memory index for fast lookups
        self._index: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Background cleanup task handle
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize cache directory
        self._init_cache_dir()
        
        # Load existing entries into index
        self._rebuild_index()
    
    def _init_cache_dir(self) -> None:
        """Create cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectory for hash prefix splitting (2 levels for better distribution)
        self._hash_dir = self._cache_dir / "data"
        self._hash_dir.mkdir(parents=True, exist_ok=True)
        # Metadata directory
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
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    created_at=entry_data["created_at"],
                    expires_at=entry_data["expires_at"],
                    size_bytes=entry_data["size_bytes"],
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=entry_data.get("last_accessed", entry_data["created_at"]),
                )
                # Only index non-expired entries
                if not entry.is_expired(now):
                    self._index[key] = entry
                    
            logger.info("KV cache index rebuilt: %d entries", len(self._index))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to rebuild KV cache index: %s", e)
    
    def _save_index(self) -> None:
        """Atomically save index to disk."""
        meta_file = self._meta_dir / "index.json"
        temp_file = meta_file.with_suffix(".tmp")
        
        data = {}
        for key, entry in self._index.items():
            data[key] = {
                "value": entry.value,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "size_bytes": entry.size_bytes,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            temp_file.rename(meta_file)
        except IOError as e:
            logger.warning("Failed to save KV cache index: %s", e)
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
    
    @staticmethod
    def compute_key(prompt: str, model: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Compute a deterministic cache key from prompt + model + params."""
        params = params or {}
        # Normalize params for deterministic hashing
        normalized_params = {}
        for k, v in sorted(params.items()):
            if isinstance(v, (list, dict)):
                normalized_params[k] = json.dumps(v, sort_keys=True)
            else:
                normalized_params[k] = str(v)
        
        content = json.dumps({
            "prompt": prompt,
            "model": model,
            "params": normalized_params,
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
    
    def _get_cache_path(self, key: str) -> Tuple[Path, Path]:
        """Get paths for data and metadata files."""
        prefix = key[:self.HASH_CHARS]
        data_dir = self._hash_dir / prefix
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / f"{key}.json", data_dir / f"{key}.meta.json"
    
    def get(self, key: str) -> Tuple[Optional[str], bool]:
        """Get a cached value.
        
        Returns:
            Tuple of (value, found). If found=True, value is the cached data.
            If found=False, value is None (cache miss or expired).
        """
        now = time.time()
        
        with self._lock:
            entry = self._index.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None, False
            
            if entry.is_expired(now):
                # Entry expired, remove it
                del self._index[key]
                self._remove_from_disk(key)
                self._stats.misses += 1
                self._save_index()
                return None, False
            
            # Update access stats
            entry.touch(now)
            
            # Read value from disk
            data_path, _ = self._get_cache_path(key)
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    value = f.read()
                
                self._stats.hits += 1
                return value, True
            except IOError:
                self._stats.misses += 1
                return None, False
    
    async def get_async(self, key: str) -> Tuple[Optional[str], bool]:
        """Async version of get()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)
    
    def set(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache a value.
        
        Args:
            key: Cache key (from compute_key or custom)
            value: String value to cache
            ttl_seconds: Time-to-live in seconds. None uses default.
            
        Returns:
            True if successfully cached, False on error.
        """
        if ttl_seconds is None:
            ttl_seconds = self._default_ttl
        
        now = time.time()
        size_bytes = len(value.encode("utf-8"))
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl_seconds if ttl_seconds > 0 else 0,
            size_bytes=size_bytes,
            last_accessed=now,
        )
        
        with self._lock:
            # Check if we need to evict entries
            self._evict_if_needed(size_bytes)
            
            # Write to disk atomically
            data_path, meta_path = self._get_cache_path(key)
            temp_data = data_path.with_suffix(".tmp")
            temp_meta = meta_path.with_suffix(".tmp")
            
            try:
                # Write data file
                with open(temp_data, "w", encoding="utf-8") as f:
                    f.write(value)
                temp_data.rename(data_path)
                
                # Write metadata file
                meta_data = {
                    "key": key,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "size_bytes": size_bytes,
                    "access_count": 0,
                    "last_accessed": now,
                }
                with open(temp_meta, "w", encoding="utf-8") as f:
                    json.dump(meta_data, f)
                temp_meta.rename(meta_path)
                
                # Update in-memory index
                self._index[key] = entry
                self._stats.writes += 1
                
                # Periodically save index (not every write)
                if self._stats.writes % 100 == 0:
                    self._save_index()
                
                return True
                
            except IOError as e:
                logger.warning("Failed to write KV cache entry: %s", e)
                self._stats.errors += 1
                # Clean up temp files
                try:
                    temp_data.unlink(missing_ok=True)
                    temp_meta.unlink(missing_ok=True)
                except OSError:
                    pass
                return False
    
    async def set_async(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Async version of set()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.set, key, value, ttl_seconds)
    
    def _remove_from_disk(self, key: str) -> None:
        """Remove entry files from disk."""
        data_path, meta_path = self._get_cache_path(key)
        try:
            data_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("Error removing cache file: %s", e)
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        with self._lock:
            if key in self._index:
                del self._index[key]
                self._remove_from_disk(key)
                self._save_index()
                return True
            return False
    
    def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict LRU entries if cache exceeds max size."""
        current_size = sum(e.size_bytes for e in self._index.values())
        target_size = self._max_size_bytes - new_entry_size
        
        if current_size <= target_size:
            return
        
        # Sort by last_accessed (oldest first) for LRU eviction
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
            current_size -= entry.size_bytes
            evicted += 1
            self._stats.evictions += 1
        
        if evicted > 0:
            logger.debug("Evicted %d entries from KV cache", evicted)
    
    def clear(self) -> int:
        """Clear all cache entries.
        
        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._index)
            
            for key in list(self._index.keys()):
                self._remove_from_disk(key)
            
            self._index.clear()
            self._save_index()
            
            logger.info("Cleared %d entries from KV cache", count)
            return count
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                writes=self._stats.writes,
                errors=self._stats.errors,
                total_latency_saved_ms=self._stats.total_latency_saved_ms,
            )
    
    def get_size_bytes(self) -> int:
        """Get current cache size in bytes."""
        with self._lock:
            return sum(e.size_bytes for e in self._index.values())
    
    def start_cleanup_task(self, interval_seconds: int = 300) -> None:
        """Start background task to clean expired entries.
        
        Args:
            interval_seconds: How often to run cleanup (default: 5 minutes)
        """
        if self._cleanup_task is not None:
            return
        
        async def _cleanup_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.debug("Cache cleanup error: %s", e)
        
        loop = asyncio.get_event_loop()
        self._cleanup_task = loop.create_task(_cleanup_loop())
        logger.debug("Started KV cache cleanup task (interval: %ds)", interval_seconds)
    
    def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.debug("Stopped KV cache cleanup task")
    
    def _cleanup_expired(self) -> int:
        """Remove all expired entries.
        
        Returns:
            Number of entries removed.
        """
        now = time.time()
        removed = 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._index.items()
                if entry.is_expired(now)
            ]
            
            for key in expired_keys:
                del self._index[key]
                self._remove_from_disk(key)
                removed += 1
            
            if removed > 0:
                self._save_index()
                logger.debug("Removed %d expired KV cache entries", removed)
        
        return removed


# Global cache instance (lazy initialization)
_global_cache: Optional[PersistentKVCache] = None
_cache_lock = threading.Lock()


def get_global_kv_cache() -> PersistentKVCache:
    """Get the global KV cache instance (singleton)."""
    global _global_cache
    
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = PersistentKVCache()
    
    return _global_cache


def configure_kv_cache(
    cache_dir: Optional[Path] = None,
    max_size_mb: int = PersistentKVCache.DEFAULT_MAX_SIZE_MB,
    default_ttl_seconds: int = PersistentKVCache.DEFAULT_TTL_SECONDS,
) -> PersistentKVCache:
    """Configure the global KV cache before first use.
    
    Must be called before any get/set operations.
    """
    global _global_cache
    
    with _cache_lock:
        if _global_cache is not None:
            logger.warning("KV cache already initialized, ignoring configure call")
            return _global_cache
        
        _global_cache = PersistentKVCache(
            cache_dir=cache_dir,
            max_size_mb=max_size_mb,
            default_ttl_seconds=default_ttl_seconds,
        )
        return _global_cache