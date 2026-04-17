"""Cache statistics tracker and reporting.

Aggregates statistics from all cache systems and provides
reporting for cost optimization and cache tuning.

Features:
- Aggregates stats from KV cache, code search cache, tool result cache
- Tracks token savings from prompt caching
- Provides human-readable reports
- Supports JSON export for monitoring
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.kv_cache import get_global_kv_cache, PersistentKVCache
from agent.code_search_cache import get_global_search_cache, CodeSearchIndexCache
from agent.tool_result_cache import get_global_tool_cache, ToolResultCache


@dataclass
class AggregatedCacheStats:
    """Aggregated statistics from all cache systems."""
    kv_cache: Dict[str, Any] = field(default_factory=dict)
    code_search_cache: Dict[str, Any] = field(default_factory=dict)
    tool_result_cache: Dict[str, Any] = field(default_factory=dict)
    prompt_cache_hits: int = 0
    prompt_cache_misses: int = 0
    prompt_cache_tokens_saved: int = 0
    total_api_calls_saved: int = 0
    estimated_cost_saved_cents: float = 0.0
    period_start: float = 0.0
    period_end: float = 0.0
    session_id: str = ""
    
    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate across all caches."""
        total_hits = (
            self.kv_cache.get("hits", 0) +
            self.code_search_cache.get("cached_files", 0) +
            self.tool_result_cache.get("cached_entries", 0)
        )
        total_requests = (
            self.kv_cache.get("hits", 0) + self.kv_cache.get("misses", 0) +
            self.code_search_cache.get("cached_files", 0) +
            self.tool_result_cache.get("cached_entries", 0)
        )
        return (total_hits / total_requests * 100) if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kv_cache": self.kv_cache,
            "code_search_cache": self.code_search_cache,
            "tool_result_cache": self.tool_result_cache,
            "prompt_cache": {
                "hits": self.prompt_cache_hits,
                "misses": self.prompt_cache_misses,
                "tokens_saved": self.prompt_cache_tokens_saved,
            },
            "totals": {
                "api_calls_saved": self.total_api_calls_saved,
                "estimated_cost_saved_cents": round(self.estimated_cost_saved_cents, 2),
                "overall_hit_rate_pct": round(self.overall_hit_rate, 2),
            },
            "period": {
                "start": self.period_start,
                "end": self.period_end,
                "duration_seconds": round(self.period_end - self.period_start, 2),
            },
            "session_id": self.session_id,
        }
    
    def to_readable_report(self) -> str:
        """Generate a human-readable cache performance report."""
        lines = [
            "═" * 60,
            "  CACHE PERFORMANCE REPORT",
            "═" * 60,
            "",
            "┌─ KV Cache ─────────────────────────────────────────",
        ]
        
        kv = self.kv_cache
        lines.append(f"│  Entries:     {kv.get('hits', 0) + kv.get('misses', 0)}")
        lines.append(f"│  Hits:       {kv.get('hits', 0)}")
        lines.append(f"│  Misses:     {kv.get('misses', 0)}")
        lines.append(f"│  Hit Rate:   {kv.get('hit_rate_pct', 0):.1f}%")
        lines.append(f"│  Writes:     {kv.get('writes', 0)}")
        lines.append(f"│  Evictions:  {kv.get('evictions', 0)}")
        lines.append(f"│  Errors:     {kv.get('errors', 0)}")
        lines.append("└─────────────────────────────────────────────────────")
        lines.append("")
        
        lines.append("┌─ Code Search Index Cache ─────────────────────────")
        cs = self.code_search_cache
        lines.append(f"│  Cached Files:     {cs.get('cached_files', 0)}")
        lines.append(f"│  Total Size:       {cs.get('total_size_mb', 0):.2f} MB")
        lines.append(f"│  Total Accesses:   {cs.get('total_accesses', 0)}")
        lines.append(f"│  Max Files:        {cs.get('max_files', 0)}")
        lines.append(f"│  Max Size:         {cs.get('max_size_mb', 0)} MB")
        lines.append("└─────────────────────────────────────────────────────")
        lines.append("")
        
        lines.append("┌─ Tool Result Cache ────────────────────────────────")
        tr = self.tool_result_cache
        lines.append(f"│  Cached Entries:   {tr.get('cached_entries', 0)}")
        lines.append(f"│  Total Size:       {tr.get('total_size_mb', 0):.2f} MB")
        lines.append(f"│  Total Accesses:   {tr.get('total_accesses', 0)}")
        lines.append(f"│  Max Size:         {tr.get('max_size_mb', 0)} MB")
        lines.append("└─────────────────────────────────────────────────────")
        lines.append("")
        
        lines.append("┌─ Prompt Cache (Anthropic) ─────────────────────────")
        pc = self.prompt_cache
        lines.append(f"│  Hits:             {pc.get('hits', 0)}")
        lines.append(f"│  Misses:           {pc.get('misses', 0)}")
        lines.append(f"│  Tokens Saved:     {pc.get('tokens_saved', 0):,}")
        lines.append("└─────────────────────────────────────────────────────")
        lines.append("")
        
        totals = self.totals
        lines.append("┌─ Overall Impact ───────────────────────────────────")
        lines.append(f"│  API Calls Saved:  {totals.get('api_calls_saved', 0)}")
        lines.append(f"│  Est. Cost Saved:   ${totals.get('estimated_cost_saved_cents', 0) / 100:.4f}")
        lines.append(f"│  Overall Hit Rate: {totals.get('overall_hit_rate_pct', 0):.1f}%")
        lines.append("└─────────────────────────────────────────────────────")
        lines.append("")
        
        duration = self.period_end - self.period_start
        if duration > 0:
            lines.append(f"Period: {duration:.0f} seconds")
        lines.append(f"Session: {self.session_id or 'unknown'}")
        lines.append("═" * 60)
        
        return "\n".join(lines)


class CacheStatsTracker:
    """Tracks and aggregates cache statistics.
    
    Collects stats from all cache systems and provides
    reporting for monitoring and optimization.
    """
    
    # Token pricing (approximate, in cents per 1M tokens)
    # These should be updated periodically
    TOKEN_PRICING = {
        # Anthropic Claude models
        "claude-opus-4": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku": {"input": 0.8, "output": 4.0},
        # OpenAI models
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        # Defaults
        "default": {"input": 5.0, "output": 15.0},
    }
    
    def __init__(self, session_id: str = ""):
        self._session_id = session_id
        self._start_time = time.time()
        self._lock = threading.Lock()
        
        # Accumulated stats for this session
        self._prompt_cache_hits = 0
        self._prompt_cache_misses = 0
        self._prompt_cache_tokens_saved = 0
        self._api_calls_saved = 0
    
    def record_prompt_cache_hit(self, tokens: int) -> None:
        """Record a prompt cache hit and estimated tokens saved."""
        with self._lock:
            self._prompt_cache_hits += 1
            # With cache hit, we save re-sending the cached tokens
            # The actual savings depends on the cache TTL
            self._prompt_cache_tokens_saved += tokens
    
    def record_prompt_cache_miss(self) -> None:
        """Record a prompt cache miss."""
        with self._lock:
            self._prompt_cache_misses += 1
    
    def record_api_call_saved(self) -> None:
        """Record when a full API call was avoided via caching."""
        with self._lock:
            self._api_calls_saved += 1
    
    def get_stats(self) -> AggregatedCacheStats:
        """Get aggregated cache statistics."""
        # Get individual cache stats
        try:
            kv_cache = get_global_kv_cache().get_stats().to_dict()
        except Exception:
            kv_cache = {"hits": 0, "misses": 0, "writes": 0, "evictions": 0, "errors": 0, "hit_rate_pct": 0}
        
        try:
            code_search = get_global_search_cache().get_stats()
        except Exception:
            code_search = {"cached_files": 0, "total_size_mb": 0, "total_accesses": 0}
        
        try:
            tool_cache = get_global_tool_cache().get_stats()
        except Exception:
            tool_cache = {"cached_entries": 0, "total_size_mb": 0, "total_accesses": 0}
        
        with self._lock:
            # Estimate cost savings
            # Each cache hit saves roughly the cost of an API call
            # Average API call is ~500 tokens input, ~200 tokens output
            avg_tokens_saved_per_hit = 500
            cost_per_1k_tokens = 0.003  # Default estimate
            estimated_savings = (
                self._api_calls_saved * avg_tokens_saved_per_hit * cost_per_1k_tokens / 100
            )
            
            return AggregatedCacheStats(
                kv_cache=kv_cache,
                code_search_cache=code_search,
                tool_result_cache=tool_cache,
                prompt_cache_hits=self._prompt_cache_hits,
                prompt_cache_misses=self._prompt_cache_misses,
                prompt_cache_tokens_saved=self._prompt_cache_tokens_saved,
                total_api_calls_saved=self._api_calls_saved,
                estimated_cost_saved_cents=estimated_savings,
                period_start=self._start_time,
                period_end=time.time(),
                session_id=self._session_id,
            )
    
    def get_report(self) -> str:
        """Get a human-readable cache performance report."""
        stats = self.get_stats()
        return stats.to_readable_report()
    
    def get_json(self) -> Dict[str, Any]:
        """Get statistics as a JSON-serializable dict."""
        return self.get_stats().to_dict()
    
    @staticmethod
    def estimate_cost_savings(
        cache_read_tokens: int,
        cache_write_tokens: int,
        model: str = "default",
    ) -> Dict[str, float]:
        """Estimate cost savings from prompt caching.
        
        Anthropic charges ~90% less for cache reads vs normal input tokens.
        
        Returns:
            Dict with cost_normal, cost_with_cache, and savings in cents
        """
        pricing = CacheStatsTracker.TOKEN_PRICING.get(
            model, CacheStatsTracker.TOKEN_PRICING["default"]
        )
        
        # Cache write tokens are charged at normal input rate
        # Cache read tokens are charged at ~10% of normal input rate
        cache_read_discount = 0.10  # 90% savings on cache reads
        
        cost_normal = (
            (cache_read_tokens + cache_write_tokens) * pricing["input"] / 1_000_000 +
            cache_write_tokens * pricing["input"] / 1_000_000 * 0
            # (cache writes are included in normal pricing)
        )
        
        cost_with_cache = (
            cache_write_tokens * pricing["input"] / 1_000_000 +
            cache_read_tokens * pricing["input"] * cache_read_discount / 1_000_000
        )
        
        savings = cost_normal - cost_with_cache
        
        return {
            "cost_normal_cents": cost_normal * 100,
            "cost_with_cache_cents": cost_with_cache * 100,
            "savings_cents": savings * 100,
            "savings_percent": (savings / cost_normal * 100) if cost_normal > 0 else 0,
        }


# Global tracker instance
_global_tracker: Optional[CacheStatsTracker] = None
_tracker_lock = threading.Lock()


def get_global_cache_stats_tracker(session_id: str = "") -> CacheStatsTracker:
    """Get the global cache statistics tracker."""
    global _global_tracker
    
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = CacheStatsTracker(session_id=session_id)
    
    return _global_tracker


def reset_cache_stats_tracker() -> None:
    """Reset the global tracker (for new session)."""
    global _global_tracker
    with _tracker_lock:
        _global_tracker = None