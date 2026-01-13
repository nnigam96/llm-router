"""
Router strategy implementations (Hybrid, Semantic, Keyword)
"""

import time
import re
import numpy as np
from typing import Tuple, List, Dict, Optional
from sentence_transformers import SentenceTransformer, util
from src.core.protocols import RoutingStrategy

class KeywordRoutingStrategy(RoutingStrategy):
    """
    L1 Strategy: Deterministic Regex Matching.
    Extremely low latency (<1ms).
    Patterns can be loaded from configuration files for easy customization.
    """
    def __init__(self, patterns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the keyword routing strategy.

        Args:
            patterns: Optional dictionary mapping expert IDs to lists of regex patterns.
                     If None, uses default patterns.
        """
        # Default patterns (used if no config provided)
        default_patterns = {
            "professor": [
                r'\bdef\b', r'\bclass\b', r'\bimport\b', r'\bmath\b',
                r'\bderivative\b', r'\bquantum\b', r'\bcode\b', r'\bdebug\b',
                r'\bfunction\b', r'\bapi\b', r'\bcompiler\b'
            ],
            "zoomer": [
                r'\byo\b', r'\blol\b', r'\bvibes\b', r'\bcap\b',
                r'\bbet\b', r'\bfinna\b', r'\bbruh\b', r'\bmeme\b',
                r'\bjoke\b', r'\blit\b', r'\bfam\b'
            ]
        }

        # Use provided patterns or defaults
        patterns = patterns or default_patterns

        # Compile patterns into regex objects
        self.patterns = {}
        for expert, pattern_list in patterns.items():
            # Combine all patterns for this expert into one regex
            combined_pattern = '|'.join(pattern_list)
            self.patterns[expert] = re.compile(combined_pattern, re.IGNORECASE)

    def route(self, query: str) -> Tuple[str, float, float]:
        start_time = time.time()

        # Check each expert's patterns
        # Priority order is determined by dictionary order (Python 3.7+ maintains insertion order)
        for expert, pattern in self.patterns.items():
            if pattern.search(query):
                latency = (time.time() - start_time) * 1000
                return expert, 1.0, latency

        # No match found
        latency = (time.time() - start_time) * 1000
        return "nomatch", 0.0, latency


class SemanticRoutingStrategy(RoutingStrategy):
    """
    L2 Strategy: Vector Similarity Search.
    Higher latency (~30ms), Higher Recall.
    Uses a quantized BERT model (all-MiniLM-L6-v2) for embeddings.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.0):
        self.encoder = SentenceTransformer(model_name)
        self.threshold = threshold
        
        # Defining the "Centroids" for our expert domains
        self.routes: Dict[str, List[str]] = {
            "professor": [
                "explain the quantum physics of light",
                "debug this python stack trace",
                "what is the derivative of log x",
                "architectural patterns for microservices",
                "mathematical proof of concept",
                "software engineering best practices"
            ],
            "zoomer": [
                "yo what is up",
                "tell me a joke bro",
                "write a tweet about pizza",
                "vibes are off today",
                "explain like i am 5",
                "that movie was mid no cap"
            ]
        }
        
        # Pre-compute embeddings on initialization (The "Index")
        self.route_embeddings = {
            k: self.encoder.encode(v) for k, v in self.routes.items()
        }

    def route(self, query: str) -> Tuple[str, float, float]:
        start_time = time.time()
        
        # 1. Embed the incoming query
        query_emb = self.encoder.encode(query)
        
        # 2. Compare against all route clusters
        scores = {}
        for route, embeddings in self.route_embeddings.items():
            # Calculate Cosine Similarity
            sims = util.cos_sim(query_emb, embeddings)
            # Take the MAX similarity to any example in the cluster
            scores[route] = float(sims.max())
            
        # 3. Decision Logic
        best_route = max(scores, key=scores.get)
        confidence = scores[best_route]
        
        latency = (time.time() - start_time) * 1000
        
        if confidence < self.threshold:
            return "nomatch", confidence, latency
            
        return best_route, confidence, latency


class HybridRoutingStrategy(RoutingStrategy):
    """
    Production Strategy: Cascading Router.
    Tries Keyword (Fast) -> Fallback to Semantic (Slow).
    """
    def __init__(self, semantic_threshold: float = 0.45):
        self.keyword_router = KeywordRoutingStrategy()
        self.semantic_router = SemanticRoutingStrategy(threshold=semantic_threshold)
        self.fallback_expert = "professor" # Safety net

    def route(self, query: str) -> Tuple[str, float, float]:
        start_time = time.time()
        
        # Step 1: Fast Path (Regex)
        expert, conf, k_lat = self.keyword_router.route(query)
        
        if expert != "nomatch":
            total_lat = (time.time() - start_time) * 1000
            return expert, conf, total_lat
            
        # Step 2: Slow Path (Vectors)
        expert, conf, s_lat = self.semantic_router.route(query)
        
        if expert == "nomatch":
            expert = self.fallback_expert
            conf = 0.1 # Low confidence fallback
            
        total_lat = (time.time() - start_time) * 1000
        return expert, conf, total_lat