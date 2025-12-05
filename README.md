# LLM-Router: High-Performance Semantic Orchestration

A forensic laboratory for optimizing Latency vs. Semantics in Distributed LLM Systems.

## Mission

Most "routers" are just simple if/else statements wrapping OpenAI calls. LLM-Router is a production-grade microservice designed to benchmark, optimize, and serve query routing logic under strict latency constraints (<50ms overhead).

It implements a Hybrid Routing Architecture (Keyword -> Semantic Fallback) and features a built-in DPO (Direct Preference Optimization) Flywheel to align routing logic with user feedback without a separate Reward Model.

## System Architecture

This project enforces strict Protocol-Oriented Programming and Modularization to simulate a scalable production environment.

### Core Components

#### The Hybrid Router

- **L1 (Fast Path)**: Deterministic Regex/Keyword matching (O(1) latency).
- **L2 (Slow Path)**: Vector-based Semantic Routing using all-MiniLM-L6-v2 (~30ms latency).

#### The Expert Engine

- Abstracted LLMProvider protocol supporting Ollama, vLLM, or Mock endpoints.
- Currently configured for asymmetric deployment: Llama-3.2-3B (Reasoning) + TinyLlama-1.1B (Speed) on consumer hardware.

#### The Forensic Lab

- Telemetry middleware tracing Router Latency vs. Inference Latency.
- Data Flywheel: Asynchronous logging of {prompt, chosen, rejected} triplets for DPO finetuning.

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama (running locally)

### 1. Setup Environment

```bash
git clone https://github.com/your-username/llm-router.git
cd llm-router
pip install -r requirements.txt
```

### 2. Initialize Models

Pull the expert models required for the default configuration:

```bash
ollama pull llama3.2:3b   # "The Professor"
ollama pull tinyllama     # "The Zoomer"
ollama serve              # Start inference server
```

### 3. Run the Microservice

Start the FastAPI server with Hydra configuration:

```bash
python src/api/main.py
```

### 4. Forensic Benchmarking

Query the router to measure overhead:

```bash
curl -X POST "http://localhost:8000/v1/route" \
     -H "Content-Type: application/json" \
     -d '{"query": "explain quantum entanglement"}'
```

## Configuration (Hydra)

Experiments are managed via strict config files in `conf/`. To change the routing threshold dynamically:

```bash
python src/api/main.py router=semantic router.threshold=0.85
```

## Telemetry & DPO

Logs are stored in `data/logs/` in JSONL format. To export a dataset for DPO training:

```python
from src.forensics.dpo_pipeline import export_dpo_dataset
export_dpo_dataset(input_log="data/logs/routing_events.jsonl", output="data/dpo_pairs/train.jsonl")
```

## License

MIT
