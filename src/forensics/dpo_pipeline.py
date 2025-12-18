"""
DPO (Direct Preference Optimization) pipeline for data flywheel
"""

import json
import os
from typing import List, Dict

def export_dpo_dataset(input_log: str, output_file: str) -> int:
    """
    CONVERTS LOGS TO DPO TRAINING DATA.
    
    The Critical Interview Concept:
    DPO requires triplets: (Prompt, Chosen_Response, Rejected_Response).
    
    Strategy (Binary Expert System):
    If User Dislikes (0) Model A -> We assume Model B was 'Chosen'.
    If User Likes (1) Model A -> Model A is 'Chosen', Model B is 'Rejected'.
    
    Note: In a real system, 'Rejected' logic is harder without explicit
    negative samples, but this weak supervision is standard for bootstrapped startups.
    """
    dpo_pairs = []
    
    if not os.path.exists(input_log):
        print(f"No logs found at {input_log}")
        return 0

    with open(input_log, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except:
                continue
                
            # Filter for events that have feedback
            if "feedback" not in data:
                continue

            # Construct the DPO Pair
            # Note: Since we don't ALWAYS generate the 'other' response,
            # we rely on the fact that we can't fully construct a (Chosen, Rejected)
            # pair unless we have the TEXT of both.
            
            # Use Case A: User Liked the response
            if data["feedback"] == 1:
                dpo_entry = {
                    "prompt": data["prompt"],
                    "chosen": data["response"], 
                    # We leave rejected empty or use a generic "I don't know" 
                    # In rigorous setups, you would run the other model offline to generate the 'rejected' text.
                    "rejected": "[Offline Generated Bad Response]", 
                    "metadata": {"source": "user_feedback_positive"}
                }
                dpo_pairs.append(dpo_entry)

            # Use Case B: User Disliked the response
            elif data["feedback"] == 0:
                 dpo_entry = {
                    "prompt": data["prompt"],
                    "chosen": "[Offline Generated Good Response]",
                    "rejected": data["response"],
                    "metadata": {"source": "user_feedback_negative"}
                }
                dpo_pairs.append(dpo_entry)

    # Write Output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dpo_pairs:
            f.write(json.dumps(entry) + "\n")
            
    return len(dpo_pairs)