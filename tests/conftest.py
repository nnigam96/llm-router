"""
Pytest configuration and fixtures for LLM Router tests
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_technical_queries():
    """Fixture providing sample technical queries"""
    return [
        "explain the def keyword in Python",
        "what is a class in object-oriented programming",
        "how do I debug a segmentation fault",
        "calculate the derivative of x^2",
        "explain quantum entanglement",
    ]


@pytest.fixture
def sample_casual_queries():
    """Fixture providing sample casual queries"""
    return [
        "yo what's up",
        "tell me a joke lol",
        "the vibes are off today",
        "no cap that's actually true",
        "bet I'll finish this",
    ]


@pytest.fixture
def sample_neutral_queries():
    """Fixture providing queries that don't clearly match either expert"""
    return [
        "what is the weather today",
        "tell me about world history",
        "how are you doing",
        "what's for dinner",
    ]
