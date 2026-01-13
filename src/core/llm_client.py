"""
LLM client implementations for different providers (Ollama, vLLM, Mock)
"""

import requests
import httpx
import json
import asyncio
from typing import Dict, Any, Optional
from src.core.protocols import LLMProvider
from src.core.retry import retry_async

class OllamaClient(LLMProvider):
    """
    Concrete implementation of LLMProvider for the Ollama inference server.
    Handles HTTP communication and error mapping.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 30):
        self.base_url = base_url
        self.generate_endpoint = f"{base_url}/api/generate"
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}

    def get_response(self, expert_name: str, query: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Connects to the local Ollama instance to fetch a completion.
        
        Args:
            expert_name: internal ID mapped to a model name in config.
            query: The prompt.
            config: Dictionary containing model mapping (e.g., {'professor': 'llama3.2:3b'})
        
        Returns:
            The text response or an error string.
        """
        if not config:
            return "Error: No expert configuration provided."

        model_id = config.get("model_name")
        system_prompt = config.get("system_prompt", "")

        if not model_id:
            return f"Error: No model mapped for expert '{expert_name}'"

        payload = {
            "model": model_id,
            "prompt": query,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 2048
            }
        }

        try:
            response = requests.post(
                self.generate_endpoint, 
                json=payload, 
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse the Ollama NDJSON/JSON response
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            return f"[System Error]: Could not connect to Ollama at {self.base_url}. Is 'ollama serve' running?"
        except requests.exceptions.Timeout:
            return f"[System Error]: Inference timed out after {self.timeout}s."
        except Exception as e:
            return f"[System Error]: {str(e)}"

    async def _make_request_with_retry(self, payload: Dict[str, Any]) -> str:
        """
        Internal method to make HTTP request with retry logic.

        Args:
            payload: The request payload

        Returns:
            The LLM response text

        Raises:
            Exception: If request fails after all retries
        """
        async def _do_request():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.generate_endpoint,
                    json=payload,
                    headers=self.headers,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()

        # Retry on connection errors and timeouts
        # Use exponential backoff: 1s, 2s, 4s
        try:
            return await retry_async(
                _do_request,
                max_retries=3,
                initial_delay=1.0,
                max_delay=5.0,
                exponential_base=2.0,
                exceptions=(httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)
            )
        except httpx.ConnectError:
            raise Exception(f"Could not connect to Ollama at {self.base_url}. Is 'ollama serve' running?")
        except httpx.TimeoutException:
            raise Exception(f"Inference timed out after {self.timeout}s.")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

    async def get_response_async(self, expert_name: str, query: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Async version: Connects to the local Ollama instance to fetch a completion.
        Includes automatic retry logic with exponential backoff for transient errors.

        Args:
            expert_name: internal ID mapped to a model name in config.
            query: The prompt.
            config: Dictionary containing model mapping (e.g., {'professor': 'llama3.2:3b'})

        Returns:
            The text response or an error string.
        """
        if not config:
            return "Error: No expert configuration provided."

        model_id = config.get("model_name")
        system_prompt = config.get("system_prompt", "")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", None)

        if not model_id:
            return f"Error: No model mapped for expert '{expert_name}'"

        payload = {
            "model": model_id,
            "prompt": query,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 2048
            }
        }

        # Add max_tokens if specified
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            return await self._make_request_with_retry(payload)
        except Exception as e:
            return f"[System Error]: {str(e)}"