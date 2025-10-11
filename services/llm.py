# services/llm.py - LLM provider abstraction with Groq primary, Ollama optional, consensus mode

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None

class GroqProvider:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        self.timeout = 30
        
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict[str, str]], model: str = "llama-3.3-70b-versatile", 
             json_mode: bool = False, temperature: float = 0.2) -> LLMResponse:
        if not self.is_available():
            return LLMResponse("Groq API key not configured", "groq", model)
        
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            max_toks = int(os.getenv("LLM_MAX_TOKENS", "3000"))
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_toks,
                "stream": False
            }
            
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            latency_ms = (time.time() - start_time) * 1000
            
            response.raise_for_status()
            data = response.json()
            
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens")
            
            return LLMResponse(content, "groq", model, tokens_used, latency_ms)
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return LLMResponse(f"Groq API error: {str(e)}", "groq", model)

class OllamaProvider:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.timeout = 30
        
    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=0.2)
            return response.status_code == 200
        except Exception:
            return False
    
    def chat(self, messages: List[Dict[str, str]], model: str = "llama3:8b", 
             json_mode: bool = False, temperature: float = 0.2) -> LLMResponse:
        if not self.is_available():
            return LLMResponse("Ollama server not reachable", "ollama", model)
        
        try:
            import requests
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 4000
                }
            }
            
            if json_mode:
                payload["format"] = "json"
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            latency_ms = (time.time() - start_time) * 1000
            
            response.raise_for_status()
            data = response.json()
            
            content = data["message"]["content"]
            
            return LLMResponse(content, "ollama", model, None, latency_ms)
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return LLMResponse(f"Ollama API error: {str(e)}", "ollama", model)

# Global provider instances
_groq_provider = GroqProvider()
_ollama_provider = OllamaProvider()

def chat(messages: List[Dict[str, str]], model: Optional[str] = None, 
         json_mode: bool = False, temperature: float = 0.2) -> str:
    """
    Main chat function with backward compatibility.
    Uses Groq by default, falls back to Ollama if configured.
    """
    # Determine active provider
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    
    if provider == "groq" and _groq_provider.is_available():
        response = _groq_provider.chat(messages, model or "llama-3.3-70b-versatile", json_mode, temperature)
        return response.content
    elif provider == "ollama" and _ollama_provider.is_available():
        response = _ollama_provider.chat(messages, model or "llama3:8b", json_mode, temperature)
        return response.content
    elif _groq_provider.is_available():
        # Fallback to Groq if configured
        response = _groq_provider.chat(messages, model or "llama-3.3-70b-versatile", json_mode, temperature)
        return response.content
    else:
        return "No LLM provider available. Please configure GROQ_API_KEY or start Ollama server."

def chat_once(messages: List[Dict[str, str]], provider: str = "groq", 
              model: Optional[str] = None, json_mode: bool = False, 
              temperature: float = 0.2) -> str:
    """
    Call a specific provider directly.
    """
    if provider == "groq":
        if not _groq_provider.is_available():
            return "Groq provider unreachable - API key not configured"
        response = _groq_provider.chat(messages, model or "llama-3.3-70b-versatile", json_mode, temperature)
        return response.content
    elif provider == "ollama":
        if not _ollama_provider.is_available():
            return "Ollama provider unreachable - server not running"
        response = _ollama_provider.chat(messages, model or "llama3:8b", json_mode, temperature)
        return response.content
    else:
        return f"Unknown provider: {provider}"

def chat_consensus(messages: List[Dict[str, str]], 
                  voters: List[Tuple[str, str]] = None,
                  judge: Tuple[str, str] = None) -> Dict[str, Any]:
    """
    Get consensus from multiple providers with a judge selecting the best response.
    """
    if voters is None:
        voters = [("groq", "llama-3.3-70b-versatile")]
        if _ollama_provider.is_available():
            voters.append(("ollama", "llama3:8b"))
    
    if judge is None:
        judge = ("groq", "llama-3.3-70b-versatile")
    
    candidates = []
    
    # Get responses from voters
    for provider, model in voters:
        try:
            if provider == "groq" and _groq_provider.is_available():
                response = _groq_provider.chat(messages, model)
                candidates.append({
                    "content": response.content,
                    "provider": provider,
                    "model": model
                })
            elif provider == "ollama" and _ollama_provider.is_available():
                response = _ollama_provider.chat(messages, model)
                candidates.append({
                    "content": response.content,
                    "provider": provider,
                    "model": model
                })
        except Exception as e:
            logger.warning(f"Voter {provider} failed: {e}")
            continue
    
    if not candidates:
        return {
            "final": "No consensus available - all providers unreachable",
            "candidates": [],
            "judge": "No judge available"
        }
    
    if len(candidates) == 1:
        return {
            "final": candidates[0]["content"],
            "candidates": candidates,
            "judge": "Single response (no consensus needed)"
        }
    
    # Judge selects the best response
    try:
        judge_provider, judge_model = judge
        judge_prompt = f"""
You are a judge selecting the best financial analysis response. Choose the most accurate, comprehensive, and well-reasoned response.

Candidates:
{chr(10).join([f"{i+1}. Provider: {c['provider']} ({c['model']}){chr(10)}Response: {c['content'][:500]}..." for i, c in enumerate(candidates)])}

Return JSON with:
- "choice": number (1-based index)
- "rationale": 1-2 line explanation

JSON:
"""
        
        judge_messages = [{"role": "user", "content": judge_prompt}]
        
        if judge_provider == "groq" and _groq_provider.is_available():
            judge_response = _groq_provider.chat(judge_messages, judge_model, json_mode=True)
        elif judge_provider == "ollama" and _ollama_provider.is_available():
            judge_response = _ollama_provider.chat(judge_messages, judge_model, json_mode=True)
        else:
            # Fallback: pick first candidate
            return {
                "final": candidates[0]["content"],
                "candidates": candidates,
                "judge": "Judge unavailable - using first response"
            }
        
        try:
            judge_data = json.loads(judge_response.content)
            choice_idx = int(judge_data.get("choice", 1)) - 1
            rationale = judge_data.get("rationale", "No rationale provided")
            
            if 0 <= choice_idx < len(candidates):
                selected = candidates[choice_idx]
            else:
                selected = candidates[0]
                rationale = "Invalid choice index - using first response"
            
            return {
                "final": selected["content"],
                "candidates": candidates,
                "judge": f"{rationale} (selected {selected['provider']})"
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Judge response parsing failed: {e}")
            return {
                "final": candidates[0]["content"],
                "candidates": candidates,
                "judge": "Judge response parsing failed - using first response"
            }
            
    except Exception as e:
        logger.error(f"Consensus judge failed: {e}")
        return {
            "final": candidates[0]["content"],
            "candidates": candidates,
            "judge": f"Judge error: {str(e)}"
        }

def get_provider_status() -> Dict[str, str]:
    """Get status of all providers."""
    status = {}
    
    if _groq_provider.is_available():
        status["groq"] = "ready"
    else:
        status["groq"] = "missing_key"
    
    if _ollama_provider.is_available():
        status["ollama"] = "online"
    else:
        status["ollama"] = "offline"
    
    return status
