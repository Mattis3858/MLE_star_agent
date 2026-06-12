"""D3: structured experiment store + B2 solution-tree nodes + C3 checkpoints.

Every evaluated candidate is a node {id, parent_id, code, scores, telemetry,
strategy, ...} appended to a JSONL file. This one store powers:
  - the beam/frontier queries of the tree search (B2),
  - the strategy blacklist fed back to the Planner,
  - resumability (C3),
  - the final report (instead of parsing .log files).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from . import config


@dataclass
class Node:
    id: int
    parent_id: Optional[int]
    iteration: int
    status: str                       # ok | error | timeout | memory_limit | ...
    strategy: str = ""
    component: str = ""
    citation: str = ""
    reasoning: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    test_scores: Dict[str, float] = field(default_factory=dict)
    telemetry: Dict = field(default_factory=dict)
    best_params: Dict = field(default_factory=dict)
    error: str = ""
    code: str = ""

    def score(self, metric: str) -> float:
        return float(self.scores.get(metric, float("inf")))


class ExperimentStore:
    def __init__(self, path: str = config.STORE_PATH):
        self.path = path
        self.nodes: List[Node] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.nodes.append(Node(**json.loads(line)))

    # -- persistence ---------------------------------------------------------

    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(node), default=str) + "\n")
        return node

    def next_id(self) -> int:
        return max((n.id for n in self.nodes), default=0) + 1

    def get(self, node_id: int) -> Optional[Node]:
        return next((n for n in self.nodes if n.id == node_id), None)

    # -- tree queries (B2) ----------------------------------------------------

    def successful(self) -> List[Node]:
        return [n for n in self.nodes if n.status == "ok" and n.scores]

    def best(self, metric: str) -> Optional[Node]:
        ok = self.successful()
        return min(ok, key=lambda n: n.score(metric)) if ok else None

    def frontier(self, metric: str, k: int = config.BEAM_WIDTH) -> List[Node]:
        """Active beam: top-k successful nodes, best first."""
        return sorted(self.successful(), key=lambda n: n.score(metric))[:k]

    def path_failures(self, node_id: int) -> int:
        """Consecutive failed descendants count along the latest path from
        node_id (used by the prune rule)."""
        count = 0
        children = [n for n in self.nodes if n.parent_id == node_id]
        while children:
            latest = children[-1]
            if latest.status == "ok":
                break
            count += 1
            children = [n for n in self.nodes if n.parent_id == latest.id]
        return count

    def top_k_diverse(self, metric: str, k: int) -> List[Node]:
        """For ensembling (B5): best nodes, preferring distinct lineages."""
        ranked = sorted(self.successful(), key=lambda n: n.score(metric))
        picked: List[Node] = []
        for n in ranked:
            roots = {self._root(p.id) for p in picked}
            if self._root(n.id) not in roots or len(picked) < 1:
                picked.append(n)
            if len(picked) == k:
                return picked
        # Not enough distinct lineages -> fill with next best regardless.
        for n in ranked:
            if n not in picked:
                picked.append(n)
            if len(picked) == k:
                break
        return picked

    def _root(self, node_id: int) -> int:
        n = self.get(node_id)
        while n and n.parent_id is not None:
            n = self.get(n.parent_id)
        return n.id if n else node_id


# ---------------------------------------------------------------------------
# C3: checkpoint (everything not derivable from the store)
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str = config.CHECKPOINT_PATH) -> None:
    serializable = {
        k: v for k, v in state.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)


def load_checkpoint(path: str = config.CHECKPOINT_PATH) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
