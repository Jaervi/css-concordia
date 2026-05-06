# Concordia CSS Simulation Framework

This repository provides a framework for simulating artificial societies using Large Language Models (LLMs) and analyzing their **Cognitive Social Structures (CSS)**. 

The framework allows you to:
1.  **Simulate interactions** in a "social sandbox" (e.g., a small town).
2.  **Conduct interviews** where agents describe their social perceptions in natural language.
3.  **Extract structured networks** from those interviews using a secondary LLM "extractor."
4.  **Analyze and Visualize** the differences between **Objective Reality** (logs) and **Subjective Perception** (mental maps).

## Quick Start

1.  **Clone the repository.**
2.  **Set up the environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Configure API Keys:**
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_google_ai_studio_key
    GEMINI_MODEL_NAME=gemini-1.5-pro
    ```
4.  **Run a test simulation (No API costs):**
    ```bash
    python experiment/concordia_sim.py --max-steps 1 --disable-llm
    ```

---

## Core Workflow

### 1. Running the Simulation
`concordia_sim.py` is the main entry point. It runs the simulation steps and triggers the interview phase at the end.

```bash
python experiment/concordia_sim.py --setup small_town_extended --max-steps 5
```
- **Outputs:** Raw logs and initial `_results.json` are saved in `experiment/outputs/`.

### 2. Extraction & Analysis
If you ran a simulation with natural language interviews, use `extract_and_analyze.py` to turn those qualitative answers into structured data.

```bash
python experiment/extract_and_analyze.py experiment/outputs/YOUR_FILE_results.json
```

### 3. Visualizing Results
The best way to see the data is to generate an interactive HTML report:

```bash
python experiment/generate_report.py experiment/outputs/YOUR_FILE_analyzed.json --open
```

You can also use the various `visualize_*.py` scripts to generate specific PNG plots for your analysis.

---

## Repository Structure

- `experiment/concordia_sim.py`: Main simulation loop.
- `experiment/questionnaires.py`: Interview prompts and CSS data structure.
- `experiment/extract_and_analyze.py`: LLM-based extraction of networks from text.
- `experiment/generate_report.py`: Interactive HTML dashboard generator.
- `experiment/metrics.py`: SNA calculations (Reciprocity, Accuracy, Consensus).
- `experiment/setups/`: JSON configurations for town environments and agents.
- `experiment/visualize_*.py`: Suite of plotting tools for specific social metrics.