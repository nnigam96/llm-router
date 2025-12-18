"""
Telemetry middleware for logging router and inference latency
"""

import json
import uuid
import os
from datetime import datetime
from typing import Any, Dict

class DataLogger:
    """
    Handles the 'Data Flywheel'. 
    Captures live traffic and formats it for Offline RL / DPO training.
    
    Design Pattern: Append-Only Log (JSONL)
    Why: Faster than SQL for high-throughput writes; robust against crashes.
    """
    def __init__(self, log_file: str = "data/logs/routing_events.jsonl"):
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file

    def log_interaction(self, 
                        prompt: str, 
                        routed_model: str, 
                        response: str, 
                        feedback_score: int, 
                        latency_ms: float):
        """
        Logs a single inference event. 
        feedback_score: 1 (Like) or 0 (Dislike)
        """
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": routed_model,
            "response": response,
            "feedback": feedback_score,
            "latency_ms": latency_ms
        }
        
        # Atomic append
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_logs(self):
        """Generator to read logs efficiently without loading all into RAM."""
        if not os.path.exists(self.log_file):
            return
            
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue