"""
LLM client implementations for different providers (Ollama, vLLM, Mock)
"""

import requests
import json
from typing import Dict, Any, Optional
from src.core.protocols import LLMProvider

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