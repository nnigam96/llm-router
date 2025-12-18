"""
Protocol definitions for Protocol-Oriented Programming
"""

from typing import Protocol, Tuple, runtime_checkable

@runtime_checkable
class LLMProvider(Protocol):
    """
    Interface for any Backend that serves LLM inference.
    Could be Ollama, OpenAI, vLLM, or a Mock.
    """
    def get_response(self, expert_name: str, query: str) -> str:
        """
        Args:
            expert_name: The internal ID of the expert (e.g., 'professor', 'zoomer').
            query: The user's prompt.
        
        Returns:
            The raw text response from the model.
        """
        ...

@runtime_checkable
class RoutingStrategy(Protocol):
    """
    Interface for any logic that routes a query to an expert.
    Could be Keyword, Semantic, Hybrid, or Random.
    """
    def route(self, query: str) -> Tuple[str, float, float]:
        """
        Args:
            query: The user's prompt.
            
        Returns:
            Tuple containing:
            1. expert_name (str): The selected expert ID.
            2. confidence (float): 0.0 to 1.0.
            3. latency_ms (float): The time taken to make the decision.
        """
        ...