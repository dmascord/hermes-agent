"""Semantic code search index cache with mtime-based invalidation.

Caches LSP symbol indexes and file search results for frequently accessed files.
Automatically invalidates when file modification times change.

Features:
- Content hash + mtime-based cache keys
- Per-file caching of search results
- Automatic invalidation on file modification
- LRU eviction when cache grows too large
- Thread-safe operation
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
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FileIndexEntry:
    """Cache entry for a file's search index."""
    file_path: str
    content_hash: str
    mtime: float
    indexed_at: float
    symbols: List[Dict[str, Any]]  # LSP symbols or search results
    size_bytes: int
    access_count: int = 0
    last_accessed: float = 0.0


class CodeSearchIndexCache:
    """Cache for code search indexes and file symbols.
    
    Uses content hash + mtime for cache validation. If either changes,
    the cache is invalidated and the file is re-indexed.
    
    This dramatically speeds up repeated searches in the same files
    by avoiding redundant LSP queries and file reads.
    """
    
    DEFAULT_MAX_FILES = 10000  # Max number of files to cache
    DEFAULT_MAX_SIZE_MB = 200  # Max cache size in MB
    HASH_SAMPLE_SIZE = 8192  # Bytes to sample from file for quick hash
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_files: int = DEFAULT_MAX_FILES,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
    ):
        if cache_dir is None:
            from hermes_constants import get_hermes_home
            cache_dir = get_hermes_home() / "cache" / "code_search"
        
        self._cache_dir = Path(cache_dir)
        self._max_files = max_files
        self._max_size_bytes = max_size_mb * 1024 * 1024
        
        # In-memory index
        self._index: Dict[str, FileIndexEntry] = {}
        self._lock = threading.RLock()
        
        # Track which files are cached (for LRU)
        self._access_order: List[str] = []
        
        # Initialize cache directory
        self._init_cache_dir()
        
        # Load existing index
        self._load_index()
    
    def _init_cache_dir(self) -> None:
        """Create cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self._cache_dir / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir = self._cache_dir / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self) -> None:
        """Load index from disk on startup."""
        meta_file = self._meta_dir / "index.json"
        if not meta_file.exists():
            return
        
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            now = time.time()
            for file_path, entry_data in data.items():
                # Verify file still exists and mtime matches
                path = Path(file_path)
                if not path.exists():
                    continue
                
                current_mtime = path.stat().st_mtime
                if abs(current_mtime - entry_data["mtime"]) > 0.001:
                    # File modified since caching, skip
                    continue
                
                entry = FileIndexEntry(
                    file_path=file_path,
                    content_hash=entry_data["content_hash"],
                    mtime=entry_data["mtime"],
                    indexed_at=entry_data["indexed_at"],
                    symbols=entry_data.get("symbols", []),
                    size_bytes=entry_data.get("size_bytes", 0),
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=entry_data.get("last_accessed", entry_data["indexed_at"]),
                )
                
                self._index[file_path] = entry
                self._access_order.append(file_path)
            
            # Trim access order to only include cached files
            self._access_order = [f for f in self._access_order if f in self._index]
            
            logger.info("Code search index loaded: %d files", len(self._index))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load code search index: %s", e)
    
    def _save_index(self) -> None:
        """Save index metadata to disk."""
        meta_file = self._meta_dir / "index.json"
        temp_file = meta_file.with_suffix(".tmp")
        
        data = {}
        for file_path, entry in self._index.items():
            data[file_path] = {
                "content_hash": entry.content_hash,
                "mtime": entry.mtime,
                "indexed_at": entry.indexed_at,
                "symbols": entry.symbols,
                "size_bytes": entry.size_bytes,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            temp_file.rename(meta_file)
        except IOError as e:
            logger.warning("Failed to save code search index: %s", e)
    
    @staticmethod
    def compute_file_key(file_path: Path, sample_content: bytes) -> Tuple[str, str]:
        """Compute cache key for a file.
        
        Returns:
            Tuple of (content_hash, cache_key)
        """
        # Use sample-based hash for speed on large files
        if len(sample_content) > CodeSearchIndexCache.HASH_SAMPLE_SIZE:
            # Sample beginning, middle, and end
            half = len(sample_content) // 2
            sample = (
                sample_content[:CodeSearchIndexCache.HASH_SAMPLE_SIZE // 2] +
                sample_content[half:half + CodeSearchIndexCache.HASH_SAMPLE_SIZE // 4] +
                sample_content[-CodeSearchIndexCache.HASH_SAMPLE_SIZE // 4:]
            )
        else:
            sample = sample_content
        
        content_hash = hashlib.sha256(sample).hexdigest()
        
        # Include mtime in key so we don't need to validate separately
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0
        
        key_data = json.dumps({
            "path": str(file_path),
            "content_hash": content_hash,
            "mtime": mtime,
        }, sort_keys=True)
        
        cache_key = hashlib.sha256(key_data.encode("utf-8")).hexdigest()
        return content_hash, cache_key
    
    def get_file_index(self, file_path: str) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        """Get cached symbols for a file.
        
        Returns:
            Tuple of (symbols_list, found). If found=True, symbols is the cached data.
            If found=False, symbols is None (cache miss or file modified).
        """
        now = time.time()
        path = Path(file_path)
        
        with self._lock:
            entry = self._index.get(file_path)
            
            if entry is None:
                return None, False
            
            # Check if file still exists
            if not path.exists():
                del self._index[file_path]
                self._access_order.remove(file_path)
                self._save_index()
                return None, False
            
            # Check if file was modified (mtime drift)
            try:
                current_mtime = path.stat().st_mtime
                if abs(current_mtime - entry.mtime) > 0.001:
                    # File modified, invalidate
                    del self._index[file_path]
                    if file_path in self._access_order:
                        self._access_order.remove(file_path)
                    self._save_index()
                    return None, False
            except OSError:
                return None, False
            
            # Update access stats
            entry.last_accessed = now
            entry.access_count += 1
            
            # Move to end of LRU list
            if file_path in self._access_order:
                self._access_order.remove(file_path)
            self._access_order.append(file_path)
            
            return entry.symbols, True
    
    def set_file_index(
        self,
        file_path: str,
        content: bytes,
        symbols: List[Dict[str, Any]],
    ) -> bool:
        """Cache symbols for a file.
        
        Args:
            file_path: Path to the file
            content: File content bytes
            symbols: List of symbol/search result dicts
            
        Returns:
            True if successfully cached, False on error.
        """
        path = Path(file_path)
        
        if not path.exists():
            return False
        
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        
        content_hash, cache_key = self.compute_file_key(path, content)
        
        now = time.time()
        size_bytes = len(content) + len(json.dumps(symbols))
        
        entry = FileIndexEntry(
            file_path=file_path,
            content_hash=content_hash,
            mtime=mtime,
            indexed_at=now,
            symbols=symbols,
            size_bytes=size_bytes,
            last_accessed=now,
        )
        
        with self._lock:
            # Check if we need to evict
            self._evict_if_needed()
            
            # Remove old entry if exists
            if file_path in self._index:
                old_entry = self._index[file_path]
                if file_path in self._access_order:
                    self._access_order.remove(file_path)
            else:
                old_entry = None
            
            self._index[file_path] = entry
            self._access_order.append(file_path)
            
            self._save_index()
            return True
    
    def _evict_if_needed(self) -> None:
        """Evict LRU entries if cache exceeds limits."""
        # Check file count limit
        while len(self._index) > self._max_files:
            if not self._access_order:
                break
            oldest = self._access_order.pop(0)
            if oldest in self._index:
                del self._index[oldest]
                logger.debug("Evicted %s (file count limit)", oldest)
        
        # Check size limit
        current_size = sum(e.size_bytes for e in self._index.values())
        target_size = self._max_size_bytes
        
        while current_size > target_size and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._index:
                entry = self._index[oldest]
                current_size -= entry.size_bytes
                del self._index[oldest]
                logger.debug("Evicted %s (size limit)", oldest)
    
    def invalidate_file(self, file_path: str) -> bool:
        """Invalidate cache for a specific file.
        
        Returns:
            True if file was cached, False otherwise.
        """
        with self._lock:
            if file_path in self._index:
                del self._index[file_path]
                if file_path in self._access_order:
                    self._access_order.remove(file_path)
                self._save_index()
                return True
            return False
    
    def invalidate_directory(self, dir_path: str) -> int:
        """Invalidate cache for all files in a directory.
        
        Returns:
            Number of files invalidated.
        """
        dir_path_obj = Path(dir_path)
        invalidated = 0
        
        with self._lock:
            to_remove = []
            for file_path in self._index.keys():
                path = Path(file_path)
                try:
                    # Check if file is under the directory
                    if path.resolve().is_relative_to(dir_path_obj.resolve()):
                        to_remove.append(file_path)
                except (ValueError, OSError):
                    pass
            
            for file_path in to_remove:
                del self._index[file_path]
                if file_path in self._access_order:
                    self._access_order.remove(file_path)
                invalidated += 1
            
            if invalidated > 0:
                self._save_index()
                logger.debug("Invalidated %d files under %s", invalidated, dir_path)
        
        return invalidated
    
    def clear(self) -> int:
        """Clear all cached indexes.
        
        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._index)
            self._index.clear()
            self._access_order.clear()
            self._save_index()
            logger.info("Cleared %d entries from code search index", count)
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(e.size_bytes for e in self._index.values())
            total_accesses = sum(e.access_count for e in self._index.values())
            
            return {
                "cached_files": len(self._index),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "max_files": self._max_files,
                "max_size_mb": self._max_size_bytes // (1024 * 1024),
                "total_accesses": total_accesses,
            }


# Global cache instance
_global_search_cache: Optional[CodeSearchIndexCache] = None
_cache_lock = threading.Lock()


def get_global_search_cache() -> CodeSearchIndexCache:
    """Get the global code search index cache instance."""
    global _global_search_cache
    
    if _global_search_cache is None:
        with _cache_lock:
            if _global_search_cache is None:
                _global_search_cache = CodeSearchIndexCache()
    
    return _global_search_cache