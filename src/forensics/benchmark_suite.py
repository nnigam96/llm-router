"""
Benchmarking suite for measuring router overhead and performance
"""

import pandas as pd
import numpy as np
from src.core.router_strategy import KeywordRoutingStrategy, SemanticRoutingStrategy, HybridRoutingStrategy

# Synthetic "Ground Truth" for Forensic analysis
TEST_SET = [
    ("calculate the integral of x^2", "professor"),
    ("yo this pizza is bussin", "zoomer"),
    ("class MyObject(object): pass", "professor"), # Keyword hit
    "explain quantum entanglement", "professor",
    "no cap that was funny", "zoomer", # Keyword hit
    "help me structure my backend", "professor", # Semantic needed
    "i am feeling sad today", "zoomer", # Semantic needed
    "import numpy as np", "professor",
    "bruh", "zoomer",
    "what is the capital of france", "professor" # Ambiguous
]

def run_benchmark():
    print("Initializing Routers for Forensic Study...")
    
    # We instantiate all strategies to compare them side-by-side
    routers = {
        "Keyword (Regex)": KeywordRoutingStrategy(),
        "Semantic (BERT)": SemanticRoutingStrategy(),
        "Hybrid (Production)": HybridRoutingStrategy()
    }
    
    results = []
    
    print(f"\n{'='*20} RUNNING ROUTING BENCHMARKS {'='*20}")
    
    for name, router in routers.items():
        latencies = []
        correct_count = 0
        
        # Run simulated traffic
        for query, ground_truth in TEST_SET:
            # Note: We unpack the tuple. Protocols guarantee this return signature.
            pred_expert, conf, lat_ms = router.route(query)
            
            latencies.append(lat_ms)
            
            # Simple Accuracy Check
            # In Hybrid, 'nomatch' counts as a failure for the keyword part, 
            # but the Hybrid strategy wrapper handles fallback. 
            # Here we check the final decision.
            if pred_expert == ground_truth:
                correct_count += 1
                
        # Metrics
        avg_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)
        accuracy = (correct_count / len(TEST_SET)) * 100
        
        results.append({
            "Architecture": name,
            "Avg Latency (ms)": round(avg_lat, 3),
            "P99 Latency (ms)": round(p99_lat, 3),
            "Accuracy (%)": round(accuracy, 1)
        })

    # Forensic Report
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()