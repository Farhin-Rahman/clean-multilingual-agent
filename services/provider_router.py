# services/provider_router.py - Provider routing and health checks

import os
import time
import logging
from typing import Dict, Optional
from services.llm import get_provider_status

logger = logging.getLogger(__name__)

def get_active_provider() -> str:
    """
    Determine the active provider based on configuration and availability.
    Returns 'groq' or 'ollama'.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "ollama":
        status = get_provider_status()
        if status.get("ollama") == "online":
            return "ollama"
        else:
            logger.warning("Ollama configured but not reachable, falling back to Groq")
            return "groq"
    
    return "groq"

def healthcheck() -> Dict[str, str]:
    """
    Non-blocking health check for all providers.
    Returns status dict with 'ready|missing_key' for Groq and 'online|offline' for Ollama.
    """
    return get_provider_status()

def routing_mode() -> str:
    """
    Get the current routing mode from environment.
    Returns 'auto', 'single', or 'consensus'.
    """
    mode = os.getenv("LLM_ROUTING_MODE", "auto").lower()
    if mode in ["auto", "single", "consensus"]:
        return mode
    return "auto"

def should_use_consensus(high_fidelity: bool = False) -> bool:
    """
    Determine if consensus mode should be used based on configuration and context.
    """
    mode = routing_mode()
    if mode == "consensus":
        return True
    elif mode == "single":
        return False
    else:  # auto
        return high_fidelity

def get_consensus_voters() -> list:
    """
    Get list of voters for consensus mode based on availability.
    """
    status = healthcheck()
    voters = []
    
    if status.get("groq") == "ready":
        voters.append(("groq", "llama-3.3-70b-versatile"))
    
    if status.get("ollama") == "online":
        voters.append(("ollama", "llama3:8b"))
    
    return voters

def get_consensus_judge() -> tuple:
    """
    Get the judge for consensus mode.
    """
    status = healthcheck()
    
    # Prefer Groq as judge if available
    if status.get("groq") == "ready":
        return ("groq", "llama-3.3-70b-versatile")
    
    # Fallback to Ollama if Groq not available
    if status.get("ollama") == "online":
        return ("ollama", "llama3:8b")
    
    # Should not happen in practice
    return ("groq", "llama-3.3-70b-versatile")
