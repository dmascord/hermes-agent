"""Enhanced local memory provider with disk-backed caching.

This provider provides persistent cross-session memory using local disk storage.
It complements the built-in MEMORY.md/USER.md with structured, searchable storage.

Features:
- Disk-backed storage for long-term memory
- Content-addressable storage for deduplication
- Automatic cleanup of old entries
- Integration with the new cache systems
- Full-text search capability
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


class LocalMemoryProvider(MemoryProvider):
    """Local disk-backed memory provider with content-addressable storage.
    
    Provides:
    - Persistent cross-session memory
    - Content hashing for deduplication
    - Automatic TTL-based cleanup
    - Full-text search on memory entries
    """
    
    NAME = "local_memory"
    DEFAULT_MEMORY_LIMIT = 5000  # Max chars per memory entry
    DEFAULT_TTL_DAYS = 30  # Entries expire after 30 days by default
    MAX_ENTRIES = 1000  # Max number of memory entries
    
    def __init__(self):
        self._hermes_home: Optional[Path] = None
        self._memory_dir: Optional[Path] = None
        self._index_file: Optional[Path] = None
        self._session_id: str = ""
        self._initialized: bool = False
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._dirty: bool = False
    
    @property
    def name(self) -> str:
        return self.NAME
    
    def is_available(self) -> bool:
        """Always available - local storage doesn't need external services."""
        return True
    
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the local memory provider."""
        self._session_id = session_id
        
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            self._hermes_home = Path(hermes_home)
        else:
            from hermes_constants import get_hermes_home
            self._hermes_home = get_hermes_home()
        
        self._memory_dir = self._hermes_home / "memories" / "local"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file for fast lookups
        self._index_file = self._memory_dir / "index.json"
        
        # Load existing entries
        self._load_index()
        
        self._initialized = True
        logger.debug("LocalMemoryProvider initialized: %d entries", len(self._entries))
    
    def _load_index(self) -> None:
        """Load memory index from disk."""
        if not self._index_file or not self._index_file.exists():
            self._entries = {}
            return
        
        try:
            with open(self._index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            now = time.time()
            self._entries = {}
            
            for key, entry in data.items():
                # Check TTL
                if entry.get("ttl_days", self.DEFAULT_TTL_DAYS) > 0:
                    age_days = (now - entry.get("created_at", now)) / 86400
                    if age_days > entry.get("ttl_days", self.DEFAULT_TTL_DAYS):
                        continue
                
                self._entries[key] = entry
            
            # Clean up expired entries
            self._save_index()
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load memory index: %s", e)
            self._entries = {}
    
    def _save_index(self) -> None:
        """Save memory index to disk."""
        if not self._index_file:
            return
        
        temp_file = self._index_file.with_suffix(".tmp")
        
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2)
            temp_file.rename(self._index_file)
            self._dirty = False
        except IOError as e:
            logger.warning("Failed to save memory index: %s", e)
    
    def _compute_key(self, content: str, entry_type: str = "memory") -> str:
        """Compute a content-based key for deduplication."""
        import hashlib
        key_data = json.dumps({
            "type": entry_type,
            "content": content,
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:32]
    
    def _add_entry(
        self,
        content: str,
        entry_type: str = "memory",
        tags: Optional[List[str]] = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> str:
        """Add a memory entry."""
        key = self._compute_key(content, entry_type)
        now = time.time()
        
        entry = {
            "id": key,
            "type": entry_type,
            "content": content[:self.DEFAULT_MEMORY_LIMIT],
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
            "ttl_days": ttl_days,
            "access_count": 0,
            "last_accessed": now,
            "session_id": self._session_id,
        }
        
        # Evict old entries if at capacity
        if len(self._entries) >= self.MAX_ENTRIES:
            self._evict_oldest()
        
        self._entries[key] = entry
        self._dirty = True
        
        # Save periodically
        if len(self._entries) % 10 == 0:
            self._save_index()
        
        return key
    
    def _evict_oldest(self) -> int:
        """Evict oldest entries to make room for new ones."""
        if not self._entries:
            return 0
        
        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: (x[1].get("last_accessed", 0), x[1].get("access_count", 0))
        )
        
        # Remove oldest 10%
        to_remove = max(1, len(sorted_entries) // 10)
        removed = 0
        
        for key, _ in sorted_entries[:to_remove]:
            del self._entries[key]
            removed += 1
        
        return removed
    
    def _search_entries(
        self,
        query: str,
        limit: int = 10,
        entry_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search memory entries by content or tags."""
        query_lower = query.lower()
        results = []
        
        for entry in self._entries.values():
            if entry_type and entry.get("type") != entry_type:
                continue
            
            # Update access stats
            entry["access_count"] = entry.get("access_count", 0) + 1
            entry["last_accessed"] = time.time()
            self._dirty = True
            
            # Simple relevance scoring
            score = 0
            content_lower = entry.get("content", "").lower()
            
            # Exact match bonus
            if query_lower in content_lower:
                score = 10
            # Word match
            query_words = query_lower.split()
            for word in query_words:
                if word in content_lower:
                    score += 2
            
            # Tag match bonus
            tags = entry.get("tags", [])
            for tag in tags:
                if query_lower in tag.lower():
                    score += 5
            
            if score > 0:
                results.append((score, entry))
        
        # Sort by score (highest first), then by recency
        results.sort(key=lambda x: (x[0], x[1].get("updated_at", 0)), reverse=True)
        
        return [entry for _, entry in results[:limit]]
    
    # ─────────────────────────────────────────────────────────────────
    # MemoryProvider interface implementation
    # ─────────────────────────────────────────────────────────────────
    
    def system_prompt_block(self) -> str:
        """Return system prompt text for this provider."""
        if not self._initialized or not self._entries:
            return ""
        
        entry_count = len(self._entries)
        if entry_count == 0:
            return ""
        
        return (
            f"[Local Memory: {entry_count} persistent memory entries available. "
            f"Use the memory tool to recall or add to cross-session memory.]"
        )
    
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memory entries for the query."""
        if not self._initialized:
            return ""
        
        if not query or len(query) < 3:
            return ""
        
        results = self._search_entries(query, limit=5)
        
        if not results:
            return ""
        
        parts = ["[Local Memory Recall]"]
        for entry in results:
            content = entry.get("content", "")[:500]
            tags = entry.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            age = self._get_age_string(entry.get("created_at", 0))
            parts.append(f"- {content}{tag_str} (added {age})")
        
        return "\n".join(parts)
    
    def _get_age_string(self, timestamp: float) -> str:
        """Get human-readable age string."""
        if not timestamp:
            return "unknown"
        
        now = time.time()
        age_days = (now - timestamp) / 86400
        
        if age_days < 1:
            hours = int(age_days * 24)
            return f"{hours}h ago" if hours > 0 else "just now"
        elif age_days < 30:
            return f"{int(age_days)}d ago"
        elif age_days < 365:
            return f"{int(age_days / 30)}mo ago"
        else:
            return f"{int(age_days / 365)}y ago"
    
    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Store key information from the turn."""
        if not self._initialized:
            return
        
        # Extract potential memory-worthy content
        combined = f"User: {user_content}\nAssistant: {assistant_content}"
        
        # Check if content is memory-worthy (has significant length and not just a greeting)
        if len(user_content) > 50 and not self._is_greeting(user_content):
            self._add_entry(
                content=combined[:2000],
                entry_type="conversation",
                tags=["conversation"],
            )
    
    @staticmethod
    def _is_greeting(text: str) -> bool:
        """Check if text is likely a greeting."""
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "howdy"]
        text_lower = text.lower()
        return any(g in text_lower for g in greetings) and len(text) < 50
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for memory operations."""
        return [
            {
                "name": "memory_recall",
                "description": "Search cross-session memory for relevant past context. Use when user asks about something that was discussed before or wants to recall information from previous sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant memories",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of memories to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_add",
                "description": "Add a persistent memory entry that will be available in future sessions. Use for important information the user wants to remember across sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags to help with later search",
                        },
                        "ttl_days": {
                            "type": "integer",
                            "description": "Days until this memory expires (0 = never)",
                            "default": 30,
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_list",
                "description": "List all cross-session memory entries. Use to see what information is stored in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of memories to return",
                            "default": 20,
                        },
                    },
                },
            },
            {
                "name": "memory_stats",
                "description": "Get statistics about the local memory store.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]
    
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle memory tool calls."""
        if not self._initialized:
            return tool_error("Memory provider not initialized")
        
        if tool_name == "memory_recall":
            query = args.get("query", "")
            limit = args.get("limit", 5)
            results = self._search_entries(query, limit=limit)
            
            if not results:
                return json.dumps({"success": True, "memories": [], "message": "No memories found"})
            
            memories = []
            for entry in results:
                memories.append({
                    "content": entry.get("content", ""),
                    "tags": entry.get("tags", []),
                    "created_at": entry.get("created_at", 0),
                    "type": entry.get("type", "memory"),
                })
            
            return json.dumps({"success": True, "memories": memories})
        
        elif tool_name == "memory_add":
            content = args.get("content", "")
            if not content:
                return tool_error("content is required for memory_add")
            
            tags = args.get("tags", [])
            ttl_days = args.get("ttl_days", self.DEFAULT_TTL_DAYS)
            
            key = self._add_entry(content=content, tags=tags, ttl_days=ttl_days)
            
            return json.dumps({"success": True, "id": key, "message": "Memory added"})
        
        elif tool_name == "memory_list":
            limit = args.get("limit", 20)
            
            entries = list(self._entries.values())
            entries.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
            
            memories = []
            for entry in entries[:limit]:
                memories.append({
                    "id": entry.get("id", ""),
                    "content": entry.get("content", ""),
                    "tags": entry.get("tags", []),
                    "created_at": entry.get("created_at", 0),
                    "type": entry.get("type", "memory"),
                })
            
            return json.dumps({"success": True, "memories": memories})
        
        elif tool_name == "memory_stats":
            total = len(self._entries)
            now = time.time()
            
            ages = []
            for entry in self._entries.values():
                age_days = (now - entry.get("created_at", now)) / 86400
                ages.append(age_days)
            
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return json.dumps({
                "success": True,
                "total_entries": total,
                "max_entries": self.MAX_ENTRIES,
                "average_age_days": round(avg_age, 1),
                "storage_dir": str(self._memory_dir) if self._memory_dir else None,
            })
        
        return tool_error(f"Unknown memory tool: {tool_name}")
    
    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Save any pending changes at session end."""
        if self._dirty:
            self._save_index()
    
    def shutdown(self) -> None:
        """Clean shutdown - save any pending changes."""
        if self._dirty:
            self._save_index()
        self._initialized = False
        logger.debug("LocalMemoryProvider shutdown")


# Register the provider
_provider_instance: Optional[LocalMemoryProvider] = None


def get_provider() -> LocalMemoryProvider:
    """Get or create the LocalMemoryProvider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = LocalMemoryProvider()
    return _provider_instance


def register() -> None:
    """Register the LocalMemoryProvider with the memory system."""
    from agent.memory_manager import MemoryManager
    # This will be called by the memory plugin system
    pass