"""MLE-STAR: autonomous ML engineering agent for tabular forecasting.

Package layout:
    config     - paths, budgets, seeds, metric configuration
    data       - loading, harness-enforced invariants, temporal features, splits
    profiling  - deterministic EDA profile (B1)
    harness    - candidate evaluation: AST gate, sandboxed runner, telemetry
    runner     - child-process entry executed per candidate (never import-side-effects)
    llm        - LLM clients, retries, token accounting
    prompts    - all prompt templates
    store      - experiment store / solution tree / blacklist / checkpoints
    search     - beam selection, prune, restart policy
    ensemble   - top-K ensembling and final test scoring
    agents     - LangGraph node functions
    graph      - graph wiring
"""

__version__ = "2.0.0"
