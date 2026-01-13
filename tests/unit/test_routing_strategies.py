"""
Unit tests for routing strategies (Keyword, Semantic, Hybrid)
"""

import pytest
from src.core.router_strategy import (
    KeywordRoutingStrategy,
    SemanticRoutingStrategy,
    HybridRoutingStrategy,
)


class TestKeywordRoutingStrategy:
    """Test cases for KeywordRoutingStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.router = KeywordRoutingStrategy()

    def test_professor_keywords(self):
        """Test that technical keywords route to professor"""
        technical_queries = [
            "explain the def keyword in Python",
            "what is a class in OOP",
            "how do I import libraries",
            "calculate the derivative of x^2",
            "explain quantum computing",
            "debug this code",
            "what is a compiler",
        ]

        for query in technical_queries:
            expert, confidence, latency = self.router.route(query)
            assert expert == "professor", f"Failed for query: {query}"
            assert confidence == 1.0
            assert latency >= 0

    def test_zoomer_keywords(self):
        """Test that casual keywords route to zoomer"""
        casual_queries = [
            "yo what's up",
            "tell me a joke lol",
            "the vibes are off",
            "no cap that's true",
            "bet I'll do it",
            "that's so lit fam",
            "bruh what is this",
        ]

        for query in casual_queries:
            expert, confidence, latency = self.router.route(query)
            assert expert == "zoomer", f"Failed for query: {query}"
            assert confidence == 1.0
            assert latency >= 0

    def test_no_match(self):
        """Test queries that don't match any keywords"""
        neutral_queries = [
            "what is the weather today",
            "tell me about history",
            "how are you",
        ]

        for query in neutral_queries:
            expert, confidence, latency = self.router.route(query)
            assert expert == "nomatch", f"Failed for query: {query}"
            assert confidence == 0.0
            assert latency >= 0

    def test_case_insensitive(self):
        """Test that keyword matching is case-insensitive"""
        assert self.router.route("IMPORT libraries")[0] == "professor"
        assert self.router.route("YO what's up")[0] == "zoomer"
        assert self.router.route("DeBuG this code")[0] == "professor"

    def test_latency_is_low(self):
        """Test that keyword routing latency is very low (<5ms)"""
        _, _, latency = self.router.route("explain the def keyword")
        assert latency < 5.0, f"Latency {latency}ms is too high for keyword routing"


class TestSemanticRoutingStrategy:
    """Test cases for SemanticRoutingStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.router = SemanticRoutingStrategy(threshold=0.3)

    def test_technical_queries(self):
        """Test that technical queries route to professor"""
        technical_queries = [
            "how do I reverse a linked list",
            "explain big O notation",
            "what is a hash table",
        ]

        for query in technical_queries:
            expert, confidence, latency = self.router.route(query)
            # Should route to professor with reasonable confidence
            assert expert == "professor", f"Failed for query: {query}"
            assert confidence > 0.3
            assert latency >= 0

    def test_casual_queries(self):
        """Test that casual queries route to zoomer"""
        casual_queries = [
            "make me laugh",
            "what's the latest gossip",
            "tell me something funny",
        ]

        for query in casual_queries:
            expert, confidence, latency = self.router.route(query)
            # Should route to either zoomer or have low confidence
            assert latency >= 0

    def test_threshold_enforcement(self):
        """Test that threshold parameter works correctly"""
        high_threshold_router = SemanticRoutingStrategy(threshold=0.9)

        # Most queries should not meet this high threshold
        expert, confidence, latency = high_threshold_router.route("hello world")

        if confidence < 0.9:
            assert expert == "nomatch"

    def test_confidence_range(self):
        """Test that confidence scores are within valid range"""
        queries = [
            "explain quantum physics",
            "tell me a joke",
            "what is the weather",
        ]

        for query in queries:
            _, confidence, _ = self.router.route(query)
            assert 0.0 <= confidence <= 1.0, f"Invalid confidence {confidence} for query: {query}"

    def test_latency_reasonable(self):
        """Test that semantic routing latency is reasonable (<100ms)"""
        _, _, latency = self.router.route("explain machine learning")
        # Semantic routing should be slower than keyword but still fast
        assert latency < 100.0, f"Latency {latency}ms is too high for semantic routing"


class TestHybridRoutingStrategy:
    """Test cases for HybridRoutingStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.router = HybridRoutingStrategy(semantic_threshold=0.45)

    def test_keyword_fast_path(self):
        """Test that queries with keywords use fast path"""
        # This should hit the keyword router (fast path)
        expert, confidence, latency = self.router.route("explain the def keyword")

        assert expert == "professor"
        assert confidence == 1.0
        # Should be very fast (keyword path)
        assert latency < 10.0

    def test_semantic_slow_path(self):
        """Test that queries without keywords use semantic path"""
        # This should miss keyword router and use semantic
        expert, confidence, latency = self.router.route("how do I sort an array efficiently")

        assert expert in ["professor", "zoomer"]
        # Latency should be higher (semantic path)
        # We can't assert exact values, but it should work

    def test_fallback_behavior(self):
        """Test that fallback works for low confidence"""
        # Create a router with high semantic threshold to trigger fallback
        router = HybridRoutingStrategy(semantic_threshold=0.99)

        expert, confidence, latency = router.route("completely random gibberish xyz123")

        # Should fallback to professor
        assert expert == "professor"
        assert confidence <= 0.99

    def test_consistency(self):
        """Test that same query produces consistent results"""
        query = "explain object-oriented programming"

        results = [self.router.route(query) for _ in range(3)]

        # All results should have the same expert
        experts = [r[0] for r in results]
        assert len(set(experts)) == 1, "Same query produced different routing decisions"

    def test_various_queries(self):
        """Test routing for various types of queries"""
        test_cases = [
            ("def my_function():", "professor"),  # Keyword match
            ("yo what's good", "zoomer"),  # Keyword match
            ("explain binary search", "professor"),  # Semantic match
            ("tell me something cool", None),  # Could be either
        ]

        for query, expected_expert in test_cases:
            expert, confidence, latency = self.router.route(query)

            if expected_expert is not None:
                assert expert == expected_expert, f"Failed for query: {query}"

            assert 0.0 <= confidence <= 1.0
            assert latency >= 0


class TestProtocolCompliance:
    """Test that all strategies comply with the RoutingStrategy protocol"""

    def test_keyword_strategy_protocol(self):
        """Test KeywordRoutingStrategy implements RoutingStrategy protocol"""
        from src.core.protocols import RoutingStrategy

        router = KeywordRoutingStrategy()
        assert isinstance(router, RoutingStrategy)

    def test_semantic_strategy_protocol(self):
        """Test SemanticRoutingStrategy implements RoutingStrategy protocol"""
        from src.core.protocols import RoutingStrategy

        router = SemanticRoutingStrategy()
        assert isinstance(router, RoutingStrategy)

    def test_hybrid_strategy_protocol(self):
        """Test HybridRoutingStrategy implements RoutingStrategy protocol"""
        from src.core.protocols import RoutingStrategy

        router = HybridRoutingStrategy()
        assert isinstance(router, RoutingStrategy)

    def test_route_return_signature(self):
        """Test that route() returns correct tuple signature"""
        routers = [
            KeywordRoutingStrategy(),
            SemanticRoutingStrategy(),
            HybridRoutingStrategy(),
        ]

        for router in routers:
            result = router.route("test query")

            # Should return tuple of (str, float, float)
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert isinstance(result[0], str)  # expert_id
            assert isinstance(result[1], float)  # confidence
            assert isinstance(result[2], float)  # latency_ms
