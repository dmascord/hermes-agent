"""Tool result cache for deterministic tool operations.

Caches results of idempotent tools like grep, glob, and file reads.
Cache keys are based on file contents + search parameters + timestamps.

Features:
- Fast cache lookups using content hash + mtime
- Automatic invalidation on file changes
- Supports glob patterns, content searches, and file reads
- LRU eviction with size limits
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolCacheEntry:
    """A cached tool result."""
    key: str
    result: str
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


class ToolResultCache:
    """Cache for deterministic tool results.
    
    Supports:
    - File reads (with mtime-based invalidation)
    - Glob searches (pattern + directory + file mtimes)
    - Content searches (grep-style with file mtimes)
    - Generic key-value caching
    
    Cache keys are deterministic: hash(contents + params + mtimes)
    """
    
    DEFAULT_MAX_SIZE_MB = 500
    DEFAULT_TTL_SECONDS = 3600  # 1 hour
    GLOB_MODE = "glob"
    GREP_MODE = "grep"
    READ_MODE = "read"
    GENERIC_MODE = "generic"
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        if cache_dir is None:
            from hermes_constants import get_hermes_home
            cache_dir = get_hermes_home() / "cache" / "tool_results"
        
        self._cache_dir = Path(cache_dir)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl_seconds
        
        # In-memory index
        self._index: Dict[str, ToolCacheEntry] = {}
        self._lock = threading.RLock()
        
        # Initialize
        self._init_cache_dir()
        self._load_index()
    
    def _init_cache_dir(self) -> None:
        """Create cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = self._cache_dir / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir = self._cache_dir / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self) -> None:
        """Load index from disk."""
        meta_file = self._meta_dir / "index.json"
        if not meta_file.exists():
            return
        
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            now = time.time()
            for key, entry_data in data.items():
                entry = ToolCacheEntry(
                    key=key,
                    result=entry_data["result"],
                    created_at=entry_data["created_at"],
                    expires_at=entry_data["expires_at"],
                    size_bytes=entry_data["size_bytes"],
                    access_count=entry_data.get("access_count", 0),
                    last_accessed=entry_data.get("last_accessed", entry_data["created_at"]),
                )
                if not entry.is_expired(now):
                    self._index[key] = entry
            
            logger.info("Tool result cache loaded: %d entries", len(self._index))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load tool result cache: %s", e)
    
    def _save_index(self) -> None:
        """Save index to disk."""
        meta_file = self._meta_dir / "index.json"
        temp_file = meta_file.with_suffix(".tmp")
        
        data = {}
        for key, entry in self._index.items():
            data[key] = {
                "result": entry.result,
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
            logger.warning("Failed to save tool result cache index: %s", e)
    
    def _get_cache_path(self, key: str) -> Tuple[Path, Path]:
        """Get paths for data and metadata files."""
        prefix = key[:4]  # Use first 4 chars for directory
        data_dir = self._data_dir / prefix
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / f"{key}.json", data_dir / f"{key}.meta.json"
    
    @staticmethod
    def _compute_file_mtimes(dir_path: Path, pattern: str = "*") -> Dict[str, float]:
        """Get modification times for all matching files."""
        mtimes = {}
        try:
            for p in Path(dir_path).rglob(pattern):
                if p.is_file():
                    try:
                        mtimes[str(p)] = p.stat().st_mtime
                    except OSError:
                        pass
        except (OSError, ValueError):
            pass
        return mtimes
    
    @staticmethod
    def _compute_dir_state(dir_path: Path) -> str:
        """Compute a hash representing the current state of a directory.
        
        Combines file count and mtimes into a single hash.
        """
        files_info = []
        try:
            for p in sorted(Path(dir_path).rglob("*")):
                if p.is_file():
                    try:
                        stat = p.stat()
                        files_info.append(f"{p}:{stat.st_mtime}:{stat.st_size}")
                    except OSError:
                        pass
        except (OSError, ValueError):
            pass
        
        content = json.dumps(files_info, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
    
    # ─────────────────────────────────────────────────────────────────
    # Cache key computation methods
    # ─────────────────────────────────────────────────────────────────
    
    @staticmethod
    def compute_read_key(file_path: str, offset: int = 0, limit: Optional[int] = None) -> str:
        """Compute cache key for a file read.
        
        Key factors: file path, content hash, offset, limit
        """
        path = Path(file_path)
        
        try:
            mtime = path.stat().st_mtime
            size = path.stat().st_size
        except OSError:
            return ""
        
        # Read sample for content hash
        try:
            with open(path, "rb") as f:
                if offset > 0:
                    f.seek(offset)
                sample = f.read(1024 * 1024)  # 1MB sample
            content_hash = hashlib.sha256(sample).hexdigest()[:16]
        except IOError:
            content_hash = ""
        
        key_data = {
            "type": ToolResultCache.READ_MODE,
            "path": str(path),
            "mtime": mtime,
            "size": size,
            "content_hash": content_hash,
            "offset": offset,
            "limit": limit,
        }
        
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    @staticmethod
    def compute_glob_key(
        dir_path: str,
        pattern: str,
        include_hidden: bool = False,
        respect_ignore: bool = True,
    ) -> str:
        """Compute cache key for a glob search.
        
        Key factors: directory state hash, glob pattern, options
        """
        dir_state = ToolResultCache._compute_dir_state(Path(dir_path))
        
        key_data = {
            "type": ToolResultCache.GLOB_MODE,
            "dir": str(Path(dir_path).resolve()),
            "pattern": pattern,
            "dir_state": dir_state,
            "include_hidden": include_hidden,
            "respect_ignore": respect_ignore,
        }
        
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    @staticmethod
    def compute_grep_key(
        dir_path: str,
        pattern: str,
        file_pattern: str = "*",
        context_lines: int = 0,
        regex: bool = True,
        case_sensitive: bool = True,
    ) -> str:
        """Compute cache key for a content search (grep).
        
        Key factors: directory state hash, search pattern, file pattern, options
        """
        dir_state = ToolResultCache._compute_dir_state(Path(dir_path))
        
        key_data = {
            "type": ToolResultCache.GREP_MODE,
            "dir": str(Path(dir_path).resolve()),
            "pattern": pattern,
            "file_pattern": file_pattern,
            "context_lines": context_lines,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "dir_state": dir_state,
        }
        
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    # ─────────────────────────────────────────────────────────────────
    # Cache operations
    # ─────────────────────────────────────────────────────────────────
    
    def get(self, key: str) -> Tuple[Optional[str], bool]:
        """Get a cached tool result.
        
        Returns:
            Tuple of (result, found)
        """
        now = time.time()
        
        with self._lock:
            entry = self._index.get(key)
            
            if entry is None:
                return None, False
            
            if entry.is_expired(now):
                del self._index[key]
                self._remove_from_disk(key)
                self._save_index()
                return None, False
            
            entry.touch(now)
            
            # Read from disk
            data_path, _ = self._get_cache_path(key)
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    result = f.read()
                return result, True
            except IOError:
                return None, False
    
    def set(
        self,
        key: str,
        result: str,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """Cache a tool result.
        
        Returns:
            True if successfully cached.
        """
        if ttl_seconds is None:
            ttl_seconds = self._default_ttl
        
        now = time.time()
        size_bytes = len(result.encode("utf-8"))
        
        entry = ToolCacheEntry(
            key=key,
            result=result,
            created_at=now,
            expires_at=now + ttl_seconds if ttl_seconds > 0 else 0,
            size_bytes=size_bytes,
            last_accessed=now,
        )
        
        with self._lock:
            # Evict if needed
            self._evict_if_needed(size_bytes)
            
            # Write to disk
            data_path, meta_path = self._get_cache_path(key)
            temp_data = data_path.with_suffix(".tmp")
            temp_meta = meta_path.with_suffix(".tmp")
            
            try:
                with open(temp_data, "w", encoding="utf-8") as f:
                    f.write(result)
                temp_data.rename(data_path)
                
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
                
                self._index[key] = entry
                
                if len(self._index) % 100 == 0:
                    self._save_index()
                
                return True
                
            except IOError as e:
                logger.warning("Failed to write tool cache entry: %s", e)
                return False
    
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
        current_size = sum(e.size_bytes for e in self._index.values())
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
            current_size -= entry.size_bytes
            evicted += 1
        
        if evicted > 0:
            logger.debug("Evicted %d tool cache entries", evicted)
    
    def invalidate_file(self, file_path: str) -> int:
        """Invalidate all cached results for a file.
        
        Returns:
            Number of entries invalidated.
        """
        file_path_str = str(Path(file_path).resolve())
        invalidated = 0
        
        with self._lock:
            to_remove = []
            
            for key in self._index.keys():
                # Check if this key relates to the file
                if key.startswith(file_path_str) or file_path_str in key:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._index[key]
                self._remove_from_disk(key)
                invalidated += 1
            
            if invalidated > 0:
                self._save_index()
        
        return invalidated
    
    def clear(self) -> int:
        """Clear all cached tool results.
        
        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._index)
            
            for key in list(self._index.keys()):
                self._remove_from_disk(key)
            
            self._index.clear()
            self._save_index()
            
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(e.size_bytes for e in self._index.values())
            total_accesses = sum(e.access_count for e in self._index.values())
            
            return {
                "cached_entries": len(self._index),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "max_size_mb": self._max_size_bytes // (1024 * 1024),
                "total_accesses": total_accesses,
            }


# Global instance
_global_tool_cache: Optional[ToolResultCache] = None
_cache_lock = threading.Lock()


def get_global_tool_cache() -> ToolResultCache:
    """Get the global tool result cache instance."""
    global _global_tool_cache
    
    if _global_tool_cache is None:
        with _cache_lock:
            if _global_tool_cache is None:
                _global_tool_cache = ToolResultCache()
    
    return _global_tool_cache