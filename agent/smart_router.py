"""Smart model routing based on query complexity.

Automatically routes queries to appropriate models based on task complexity:
- Simple queries (explanations, definitions, simple generation) → cheap/fast models
- Complex queries (code analysis, architecture, research) → premium models

Features:
- Query complexity analysis
- Provider/model routing rules
- Cost-based routing decisions
- Manual override support
- Statistics tracking per routing decision
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.deduplicator import PromptDeduplicator

logger = logging.getLogger(__name__)


# Model pricing per 1M tokens (input, output) in cents
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4": {"input": 15.0, "output": 75.0, "provider": "anthropic"},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3-5-haiku": {"input": 0.8, "output": 4.0, "provider": "anthropic"},
    "claude-3-opus": {"input": 15.0, "output": 75.0, "provider": "anthropic"},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3-haiku": {"input": 0.8, "output": 4.0, "provider": "anthropic"},
    
    # OpenAI
    "gpt-4o": {"input": 5.0, "output": 15.0, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openai"},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0, "provider": "openai"},
    "gpt-4": {"input": 30.0, "output": 60.0, "provider": "openai"},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5, "provider": "openai"},
    
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0, "provider": "google"},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "provider": "google"},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15, "provider": "google"},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40, "provider": "google"},
    
    # OpenRouter (varies by provider)
    "openrouter/anthropic/claude-3-haiku": {"input": 0.8, "output": 4.0, "provider": "openrouter"},
    "openrouter/openai/gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openrouter"},
    
    # Deepseek
    "deepseek-chat": {"input": 0.14, "output": 0.28, "provider": "deepseek"},
    "deepseek-coder": {"input": 0.14, "output": 0.28, "provider": "deepseek"},
}

# Tier definitions
MODEL_TIERS = {
    "ultra_premium": ["claude-opus-4", "gpt-4"],
    "premium": ["claude-sonnet-4", "claude-3-5-sonnet", "gpt-4o", "gpt-4-turbo", "gemini-1.5-pro"],
    "standard": ["claude-3-5-haiku", "gpt-4o-mini", "gemini-1.5-flash", "deepseek-chat"],
    "budget": ["gpt-3.5-turbo", "gemini-1.5-flash-8b"],
}

# Routing rules: complexity → acceptable tiers
ROUTING_RULES = {
    "simple": ["budget", "standard", "premium", "ultra_premium"],
    "moderate": ["standard", "premium", "ultra_premium"],
    "complex": ["premium", "ultra_premium"],
}


@dataclass
class RoutingStats:
    """Statistics for routing decisions."""
    total_requests: int = 0
    simple_routed_to_cheap: int = 0
    complex_routed_to_premium: int = 0
    overridden: int = 0
    cost_savings_cents: float = 0.0
    errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "simple_routed_to_cheap": self.simple_routed_to_cheap,
            "complex_routed_to_premium": self.complex_routed_to_premium,
            "overridden": self.overridden,
            "cost_savings_cents": round(self.cost_savings_cents, 2),
            "errors": self.errors,
        }


@dataclass
class ModelPreference:
    """Preference settings for a model."""
    name: str
    provider: str
    input_cost: float
    output_cost: float
    tier: str
    supportsstreaming: bool = True
    supports_function_calls: bool = True
    max_tokens: int = 128000
    context_length: int = 200000
    
    @property
    def avg_cost_per_1k(self) -> float:
        return (self.input_cost + self.output_cost) / 2


class SmartRouter:
    """Routes queries to appropriate models based on complexity and cost.
    
    Decision flow:
    1. Analyze query complexity (simple/moderate/complex)
    2. Determine acceptable model tiers
    3. Select cheapest model in acceptable tiers
    4. Consider user overrides and preferences
    5. Return selected model + routing reason
    """
    
    def __init__(
        self,
        primary_model: str = "claude-3-5-sonnet",
        budget_model: str = "claude-3-5-haiku",
        enable_routing: bool = True,
        routing_threshold: float = 0.7,
    ):
        self._primary_model = primary_model
        self._budget_model = budget_model
        self._enable_routing = enable_routing
        self._routing_threshold = routing_threshold
        
        self._lock = threading.Lock()
        self._stats = RoutingStats()
        self._deduplicator = PromptDeduplicator()
        
        # User-specified model preferences
        self._preferred_models: Dict[str, ModelPreference] = {}
        
        # Build model registry from pricing
        self._model_registry = self._build_model_registry()
    
    def _build_model_registry(self) -> Dict[str, ModelPreference]:
        """Build model preferences from pricing table."""
        registry = {}
        
        for model_name, pricing in MODEL_PRICING.items():
            # Determine tier
            tier = "standard"
            for t, models in MODEL_TIERS.items():
                if model_name in models or model_name.lower() in models:
                    tier = t
                    break
            
            registry[model_name] = ModelPreference(
                name=model_name,
                provider=pricing.get("provider", "unknown"),
                input_cost=pricing["input"],
                output_cost=pricing["output"],
                tier=tier,
            )
        
        return registry
    
    def register_model(
        self,
        name: str,
        provider: str,
        input_cost: float,
        output_cost: float,
        tier: str = "standard",
        **kwargs,
    ) -> None:
        """Register a custom model with pricing."""
        self._model_registry[name] = ModelPreference(
            name=name,
            provider=provider,
            input_cost=input_cost,
            output_cost=output_cost,
            tier=tier,
            **kwargs,
        )
    
    def get_model_preference(self, model_name: str) -> Optional[ModelPreference]:
        """Get preference settings for a model."""
        return self._model_registry.get(model_name)
    
    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for a request in cents."""
        pref = self._model_registry.get(model)
        if not pref:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * pref.input_cost
        output_cost = (output_tokens / 1_000_000) * pref.output_cost
        return input_cost + output_cost
    
    def analyze_complexity(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, float]:
        """Analyze query complexity using deduplicator's complexity estimation."""
        return self._deduplicator.estimate_complexity(prompt, conversation_history)
    
    def route(
        self,
        prompt: str,
        preferred_model: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        force_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a query to an appropriate model.
        
        Returns:
            Dict with:
            - model: Selected model name
            - provider: Model provider
            - complexity: "simple", "moderate", or "complex"
            - confidence: Confidence in routing decision
            - reason: Human-readable explanation
            - cost_estimate: Estimated cost in cents
            - savings_vs_primary: Estimated savings vs primary model
        """
        if not self._enable_routing:
            return self._route_to_primary(prompt, preferred_model, force_model)
        
        # Check for user override
        if force_model:
            with self._lock:
                self._stats.overridden += 1
            return self._route_with_model(prompt, force_model, "user_override")
        
        # Analyze complexity
        complexity, confidence = self.analyze_complexity(prompt, conversation_history)
        
        with self._lock:
            self._stats.total_requests += 1
        
        # Get acceptable tiers based on complexity
        acceptable_tiers = ROUTING_RULES.get(complexity, ["standard", "premium"])
        
        # Select cheapest acceptable model
        selected_model = None
        best_cost = float("inf")
        
        for tier in acceptable_tiers:
            for model_name, pref in self._model_registry.items():
                if pref.tier == tier and pref.avg_cost_per_1k < best_cost:
                    # Prefer current primary model for moderate tasks
                    if complexity == "moderate" and model_name == self._primary_model:
                        best_cost = pref.avg_cost_per_1k
                        selected_model = model_name
                    elif complexity != "moderate":
                        best_cost = pref.avg_cost_per_1k
                        selected_model = model_name
        
        # Fallback to primary if no model found
        if not selected_model:
            selected_model = self._primary_model
        
        # Calculate cost estimates
        primary_pref = self._model_registry.get(self._primary_model)
        selected_pref = self._model_registry.get(selected_model)
        
        if primary_pref and selected_pref:
            avg_primary = primary_pref.avg_cost_per_1k
            avg_selected = selected_pref.avg_cost_per_1k
            savings = avg_primary - avg_selected
        else:
            savings = 0.0
        
        # Update stats
        with self._lock:
            if complexity == "simple" and selected_pref and selected_pref.tier in ["budget", "standard"]:
                self._stats.simple_routed_to_cheap += 1
                self._stats.cost_savings_cents += savings
            elif complexity == "complex":
                self._stats.complex_routed_to_premium += 1
        
        return {
            "model": selected_model,
            "provider": selected_pref.provider if selected_pref else "unknown",
            "complexity": complexity,
            "confidence": confidence,
            "reason": self._get_routing_reason(complexity, selected_model, confidence),
            "cost_estimate": selected_pref.avg_cost_per_1k if selected_pref else 0,
            "savings_vs_primary": savings,
            "tier": selected_pref.tier if selected_pref else "unknown",
        }
    
    def _route_to_primary(
        self,
        prompt: str,
        preferred_model: Optional[str],
        force_model: Optional[str],
    ) -> Dict[str, Any]:
        """Route to primary or preferred model without smart routing."""
        model = force_model or preferred_model or self._primary_model
        pref = self._model_registry.get(model)
        
        return {
            "model": model,
            "provider": pref.provider if pref else "unknown",
            "complexity": "unknown",
            "confidence": 1.0,
            "reason": "primary_model",
            "cost_estimate": pref.avg_cost_per_1k if pref else 0,
            "savings_vs_primary": 0,
            "tier": pref.tier if pref else "unknown",
        }
    
    def _route_with_model(
        self,
        prompt: str,
        model: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Route to a specific model with explanation."""
        pref = self._model_registry.get(model)
        
        return {
            "model": model,
            "provider": pref.provider if pref else "unknown",
            "complexity": "unknown",
            "confidence": 1.0,
            "reason": reason,
            "cost_estimate": pref.avg_cost_per_1k if pref else 0,
            "savings_vs_primary": 0,
            "tier": pref.tier if pref else "unknown",
        }
    
    def _get_routing_reason(
        self,
        complexity: str,
        model: str,
        confidence: float,
    ) -> str:
        """Generate human-readable routing explanation."""
        pref = self._model_registry.get(model)
        tier = pref.tier if pref else "unknown"
        
        if complexity == "simple":
            return f"Routed to {tier} model ({model}) for simple query (confidence: {confidence:.0%})"
        elif complexity == "moderate":
            return f"Routed to {tier} model ({model}) for moderate query (confidence: {confidence:.0%})"
        else:
            return f"Routed to {tier} model ({model}) for complex query (confidence: {confidence:.0%})"
    
    def get_stats(self) -> RoutingStats:
        """Get routing statistics."""
        with self._lock:
            return RoutingStats(
                total_requests=self._stats.total_requests,
                simple_routed_to_cheap=self._stats.simple_routed_to_cheap,
                complex_routed_to_premium=self._stats.complex_routed_to_premium,
                overridden=self._stats.overridden,
                cost_savings_cents=self._stats.cost_savings_cents,
                errors=self._stats.errors,
            )
    
    def get_available_models(self, tier: Optional[str] = None) -> List[ModelPreference]:
        """Get all available models, optionally filtered by tier."""
        if tier:
            return [m for m in self._model_registry.values() if m.tier == tier]
        return list(self._model_registry.values())
    
    def get_cheapest_model(self, min_tier: str = "standard") -> Optional[ModelPreference]:
        """Get cheapest model above a minimum tier."""
        tier_order = ["budget", "standard", "premium", "ultra_premium"]
        min_idx = tier_order.index(min_tier) if min_tier in tier_order else 1
        
        candidates = [
            m for m in self._model_registry.values()
            if tier_order.index(m.tier) >= min_idx
        ]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda m: m.avg_cost_per_1k)


# Global instance
_global_router: Optional[SmartRouter] = None
_router_lock = threading.Lock()


def get_global_router() -> SmartRouter:
    """Get the global smart router instance."""
    global _global_router
    
    if _global_router is None:
        with _router_lock:
            if _global_router is None:
                _global_router = SmartRouter()
    
    return _global_router


def configure_router(
    primary_model: str = "claude-3-5-sonnet",
    budget_model: str = "claude-3-5-haiku",
    enable_routing: bool = True,
    routing_threshold: float = 0.7,
) -> SmartRouter:
    """Configure the global router."""
    global _global_router
    
    with _router_lock:
        _global_router = SmartRouter(
            primary_model=primary_model,
            budget_model=budget_model,
            enable_routing=enable_routing,
            routing_threshold=routing_threshold,
        )
        return _global_router