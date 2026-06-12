"""LangGraph node functions.

Flow: research (B1 profile + retrieval) -> foundation -> refine loop
(evaluate -> C1 debug sub-loop -> record -> B3 ablation -> B2 planner ->
coder) -> finalize (B6 CV confirm + A2 test-once + B5 ensemble) -> report.
"""

import json
import os
import time
from typing import Optional, TypedDict

import numpy as np

from . import config, data, harness, llm, profiling, prompts, store
from .runner import METRICS

STORE: store.ExperimentStore = None  # set by init_run()
RESEARCH_CACHE = os.path.join(config.CACHE_DIR, "research.json")


class AgentState(TypedDict, total=False):
    task: str
    profile_md: str
    citations: str
    design_spec: str
    feature_table: str
    code: str
    parent_id: Optional[int]
    plan: dict
    iteration: int
    best_id: Optional[int]
    best_score: float
    no_improve_rounds: int
    min_delta: float
    noise: dict
    ablation: dict
    started_at: float
    stop_reason: str
    final: dict
    report: str


def init_run(resume: bool = False) -> dict:
    """Create/load the experiment store and checkpoint. Returns initial state
    overrides when resuming."""
    global STORE
    config.ensure_dirs()
    if not resume:
        for p in (config.STORE_PATH, config.CHECKPOINT_PATH):
            if os.path.exists(p):
                os.remove(p)
    STORE = store.ExperimentStore()
    if resume:
        ckpt = store.load_checkpoint()
        if ckpt:
            print(f">> Resuming at iteration {ckpt.get('iteration', 0)}")
            return ckpt
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ast_repair(code: str) -> str:
    """Part E micro-repair loop: only syntactically valid code may proceed."""
    for _ in range(config.AST_REPAIR_MAX):
        err = harness.ast_gate(code)
        if err is None:
            return code
        print(f">> AST gate failed ({err}); requesting in-memory fix...")
        code = llm.strip_fences(
            llm.call_coder(prompts.SYNTAX_REPAIR.format(error=err, code=code))
        )
    return code


def _feature_table(meta: dict) -> str:
    lines = ["| column | dtype | group |", "|---|---|---|"]
    for g, cols in meta["feature_groups"].items():
        for c in cols:
            lines.append(f"| {c} | {meta['dtypes'][c]} | {g} |")
    return "\n".join(lines)


def _baseline_example() -> str:
    with open(harness.BASELINE_PATH, encoding="utf-8") as f:
        return prompts.EXAMPLE_NOTE.format(baseline_code=f.read())


def _frontier_md(metric: str) -> str:
    rows = []
    for n in STORE.frontier(metric):
        t = n.telemetry or {}
        rows.append(
            f"- node {n.id} (parent {n.parent_id}): {metric}={n.score(metric):.4f}, "
            f"strategy: {n.strategy or 'foundation'}; "
            f"fit {t.get('fit_seconds', '?')}s, {t.get('n_features', '?')} features"
        )
    return "\n".join(rows) if rows else "(no successful candidates yet)"


def _telemetry_md(metric: str, k: int = 5) -> str:
    rows = []
    for n in STORE.nodes[-k:]:
        t = n.telemetry or {}
        rows.append(
            f"- node {n.id}: status={n.status}, {metric}={n.score(metric):.4f}, "
            f"total {t.get('total_seconds', '?')}s, peak {t.get('peak_mb', '?')} MB"
        )
    return "\n".join(rows) if rows else "(none)"


def _blacklist_md() -> str:
    """B2: strategies that never landed on the best node's lineage."""
    best = STORE.best(config.METRIC)
    lineage = set()
    node = best
    while node:
        lineage.add(node.id)
        node = STORE.get(node.parent_id) if node.parent_id else None
    bad = [n.strategy for n in STORE.nodes if n.strategy and n.id not in lineage]
    bad = list(dict.fromkeys(bad))[-config.BLACKLIST_CAP:]
    return "\n".join(f"- {s}" for s in bad) if bad else "(empty)"


def _ablation_md(deltas: dict) -> str:
    if not deltas:
        return "(not yet run)"
    return "\n".join(
        f"- remove {g}: {d:+.4f}" for g, d in sorted(
            deltas.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -9e9)
        )
    )


def _nodes_md(metric: str) -> str:
    lines = [
        f"| node | parent | status | val {metric} | strategy | citation |",
        "|---|---|---|---|---|---|",
    ]
    for n in STORE.nodes:
        score = f"{n.score(metric):.4f}" if n.scores else "-"
        lines.append(
            f"| {n.id} | {n.parent_id} | {n.status} | {score} "
            f"| {(n.strategy or 'foundation')[:90]} | {(n.citation or '')[:60]} |"
        )
    return "\n".join(lines)


def _web_search(query: str, max_results: int = 5) -> str:
    if not os.getenv("TAVILY_API_KEY"):
        return ""
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        results = TavilySearchResults(max_results=max_results).invoke(query)
        return "\n".join(
            f"Source: {r['url']}\n{r['content'][:800]}" for r in results
        )
    except Exception as e:
        print(f">> Tavily search failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Node 1: Research / EDA (B1 + B7-lite)
# ---------------------------------------------------------------------------

def research_node(state: AgentState) -> dict:
    print("\n=== [1] Research & EDA ===")
    df, meta = data.prepare()
    prof = profiling.profile(df, meta)
    profile_md = profiling.render_markdown(prof)
    feature_table = _feature_table(meta)
    print(f">> Profiled {prof['n_rows']:,} rows; "
          f"target skew {prof['target']['skew']} -> log recommended: "
          f"{prof['target']['log_transform_recommended']}")

    if os.path.exists(RESEARCH_CACHE):  # C5: don't re-pay for research
        with open(RESEARCH_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        print(">> Research cache hit")
        return {**cached, "profile_md": profile_md, "feature_table": feature_table}

    search_results = ""
    try:
        from langchain_community.utilities import ArxivAPIWrapper

        search_results += "--- Core paper ---\n" + ArxivAPIWrapper(
            top_k_results=1, doc_content_chars_max=2000
        ).run("2506.15692")
    except Exception as e:
        print(f">> Arxiv fetch failed: {e}")

    task = state["task"]
    web = _web_search(f"{task} kaggle winner solution feature engineering")
    code_web = _web_search(f"{task} github solution feature engineering code pandas")  # B7-lite
    if web:
        search_results += "\n--- Kaggle solutions ---\n" + web
    if code_web:
        search_results += "\n--- Code-grounded snippets (inspiration only; never executed) ---\n" + code_web

    citations = llm.call_reasoner(
        prompts.RESEARCH.format(task=task, profile_md=profile_md, search_results=search_results)
    )

    # D4: ingest cross-run lessons (scoped to this task).
    lessons = ""
    if os.path.exists(config.LESSONS_PATH):
        with open(config.LESSONS_PATH, encoding="utf-8") as f:
            lessons = f.read()[-3000:]
        citations += "\n\n--- Lessons from previous runs on this task ---\n" + lessons

    design_spec = llm.call_reasoner(
        prompts.DESIGN.format(
            task=task, profile_md=profile_md, citations=citations, metric=config.METRIC
        )
    )
    print(">> Design spec ready")

    result = {"citations": citations, "design_spec": design_spec}
    with open(RESEARCH_CACHE, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return {**result, "profile_md": profile_md, "feature_table": feature_table}


# ---------------------------------------------------------------------------
# Node 2: Foundation + noise floor (A3)
# ---------------------------------------------------------------------------

def foundation_node(state: AgentState) -> dict:
    print("\n=== [2] Foundation ===")
    print(">> Estimating noise floor (baseline x seeds)...")
    noise = harness.noise_floor()
    print(f">> Noise floor: std={noise['std']:.4f} -> min_delta={noise['min_delta']:.4f} "
          f"(baseline {config.METRIC} ~ {noise.get('baseline_mean', float('nan')):.4f})")

    code = llm.strip_fences(
        llm.call_coder(
            prompts.FOUNDATION.format(
                task=state["task"],
                design_spec=state["design_spec"],
                feature_table=state["feature_table"],
                contract=prompts.CONTRACT,
                example=_baseline_example(),
            )
        )
    )
    code = _ast_repair(code)
    return {
        "code": code,
        "parent_id": None,
        "plan": {"component": "foundation", "strategy": "initial candidate from design spec",
                 "citation": "", "reasoning": "foundation"},
        "iteration": 0,
        "best_id": None,
        "best_score": float("inf"),
        "no_improve_rounds": 0,
        "noise": noise,
        "min_delta": noise["min_delta"],
        "started_at": time.time(),
        "stop_reason": "",
        "ablation": {},
    }


# ---------------------------------------------------------------------------
# Node 3: Refinement loop (B2 + C1 + B3)
# ---------------------------------------------------------------------------

def _debug_loop(code: str, result: harness.EvalResult, name: str) -> tuple:
    """C1: dedicated self-repair loop with error-driven retrieval."""
    for attempt in range(1, config.DEBUG_MAX_ATTEMPTS + 1):
        print(f">> Debug attempt {attempt}/{config.DEBUG_MAX_ATTEMPTS} ({result.status})")
        search_hint = ""
        if attempt == 1 and result.status == "error":
            first_line = result.error.strip().splitlines()[-1][:200] if result.error else ""
            hits = _web_search(f"python {first_line}", max_results=2)
            if hits:
                search_hint = "Possibly relevant web results:\n" + hits
        code = llm.strip_fences(
            llm.call_coder(
                prompts.DEBUGGER.format(
                    status=result.status, error=result.error[:3000],
                    search_hint=search_hint, code=code, contract=prompts.CONTRACT,
                )
            )
        )
        code = _ast_repair(code)
        result = harness.evaluate(
            code, f"{name}_fix{attempt}", optuna_trials=0
        )
        if result.status == "ok":
            print(f">> Debug succeeded on attempt {attempt}")
            return code, result
    return code, result


def refine_node(state: AgentState) -> dict:
    it = state["iteration"] + 1
    metric = config.METRIC
    print(f"\n=== [3] Refinement {it}/{config.MAX_ITERATIONS} ===")

    updates: dict = {"iteration": it}
    plan = state.get("plan", {})
    code = state["code"]

    try:
        result = harness.evaluate(code, f"iter_{it}", optuna_trials=config.OPTUNA_TRIALS_FAST)
        if result.status != "ok":
            code, result = _debug_loop(code, result, f"iter_{it}")

        node = STORE.add(store.Node(
            id=STORE.next_id(), parent_id=state.get("parent_id"), iteration=it,
            status=result.status, strategy=plan.get("strategy", ""),
            component=plan.get("component", ""), citation=plan.get("citation", ""),
            reasoning=plan.get("reasoning", ""), scores=result.scores,
            telemetry=result.telemetry, best_params=result.best_params,
            error=result.error[:2000], code=code,
        ))

        best_score = state.get("best_score", float("inf"))
        no_improve = state.get("no_improve_rounds", 0)
        min_delta = state.get("min_delta", config.MIN_DELTA_ABS)

        if result.status == "ok":
            s = result.score(metric)
            print(f">> val {metric} = {s:.4f} (best {best_score:.4f}, min_delta {min_delta:.4f})")
            if s < best_score and (
                best_score == float("inf")
                or harness.is_real_improvement(s, best_score, min_delta)
            ):
                print(f">> NEW BEST: node {node.id}")
                updates.update(best_id=node.id, best_score=s, no_improve_rounds=0)
            else:
                updates["no_improve_rounds"] = no_improve + 1
                print(f">> No real improvement ({no_improve + 1} rounds)")
        else:
            updates["no_improve_rounds"] = no_improve + 1
            print(f">> Candidate failed permanently: {result.status}")

        # B3: periodic ablation on the current best.
        best = STORE.best(metric)
        if best and it % config.ABLATION_EVERY == 0:
            print(">> Running ablation on current best...")
            _, meta = data.prepare()
            deltas = harness.ablation(
                best.code, best.score(metric), list(meta["feature_groups"]),
            )
            updates["ablation"] = deltas
            print(f">> Ablation: { {k: round(v, 3) for k, v in deltas.items()} }")

        # Plan the next candidate (unless this was the last iteration).
        if it < config.MAX_ITERATIONS:
            nxt = _plan_next(state, updates, result)
            updates.update(nxt)

    except llm.BudgetExceeded as e:
        print(f">> {e}")
        updates["stop_reason"] = "token_budget"

    if time.time() - state.get("started_at", time.time()) > config.WALL_CLOCK_BUDGET_S:
        updates["stop_reason"] = "wall_clock_budget"

    merged = {**state, **updates}
    store.save_checkpoint({k: v for k, v in merged.items() if k != "report"})
    return updates


def _plan_next(state: AgentState, updates: dict, last: harness.EvalResult) -> dict:
    metric = config.METRIC
    best = STORE.best(metric)
    no_improve = updates.get("no_improve_rounds", state.get("no_improve_rounds", 0))

    # No successful node yet -> regenerate from scratch, no planner needed.
    if best is None:
        print(">> No healthy candidate yet; regenerating foundation")
        code = _ast_repair(llm.strip_fences(llm.call_coder(
            prompts.FOUNDATION.format(
                task=state["task"], design_spec=state["design_spec"],
                feature_table=state["feature_table"], contract=prompts.CONTRACT,
                example=_baseline_example(),
            )
        )))
        return {"code": code, "parent_id": None,
                "plan": {"component": "foundation", "strategy": "regenerated foundation",
                         "citation": "", "reasoning": "no successful candidate yet"}}

    if last.status == "ok":
        last_detail = f"val {metric} = {last.score(metric):.4f}"
    else:
        last_detail = f"error:\n{last.error[:1500]}"

    plan_raw = llm.call_reasoner(prompts.PLANNER.format(
        metric=metric,
        min_delta=state.get("min_delta", config.MIN_DELTA_ABS),
        frontier_md=_frontier_md(metric),
        best_id=best.id, best_score=best.score(metric),
        no_improve=no_improve,
        last_status=last.status, last_detail=last_detail,
        ablation_md=_ablation_md(updates.get("ablation") or state.get("ablation", {})),
        telemetry_md=_telemetry_md(metric),
        blacklist_md=_blacklist_md(),
        profile_md=state["profile_md"][:1500],
        restart_after=config.RESTART_AFTER,
    ))
    try:
        txt = plan_raw.strip()
        if "```" in txt:
            txt = txt.split("```json")[-1].split("```")[1] if "```json" in txt else txt.split("```")[1]
        plan = json.loads(txt)
    except Exception:
        plan = {"branch_from_node_id": best.id, "restart": False,
                "component": "recovery", "strategy": "simplify the model configuration",
                "citation": "", "reasoning": "planner returned invalid JSON"}

    forced_restart = no_improve >= config.RESTART_AFTER
    if plan.get("restart") or forced_restart:
        print(f">> RESTART (planner={bool(plan.get('restart'))}, forced={forced_restart})")
        code = _ast_repair(llm.strip_fences(llm.call_coder(
            prompts.FOUNDATION.format(
                task=state["task"],
                design_spec=state["design_spec"]
                + "\n\nIMPORTANT: previous approaches plateaued. Take a structurally "
                  "DIFFERENT approach (different model family or encoding strategy) than:\n"
                + _blacklist_md(),
                feature_table=state["feature_table"], contract=prompts.CONTRACT,
                example=_baseline_example(),
            )
        )))
        return {"code": code, "parent_id": None,
                "plan": {"component": "restart", "strategy": plan.get("strategy", "random restart"),
                         "citation": plan.get("citation", ""),
                         "reasoning": plan.get("reasoning", "")}}

    # B2: branch from any frontier node; prune dead paths back to best.
    branch = STORE.get(int(plan.get("branch_from_node_id") or best.id)) or best
    if STORE.path_failures(branch.id) >= config.PRUNE_AFTER:
        print(f">> Path from node {branch.id} pruned; branching from best (node {best.id})")
        branch = best

    print(f">> Plan: [{plan.get('component')}] {plan.get('strategy')} (branch from {branch.id})")
    code = _ast_repair(llm.strip_fences(llm.call_coder(prompts.CODER.format(
        strategy=plan.get("strategy", ""), component=plan.get("component", ""),
        reasoning=plan.get("reasoning", ""), base_code=branch.code,
        contract=prompts.CONTRACT,
    ))))
    return {"code": code, "parent_id": branch.id, "plan": plan}


def should_continue(state: AgentState) -> str:
    if state.get("stop_reason"):
        return "finalize"
    if state["iteration"] >= config.MAX_ITERATIONS:
        return "finalize"
    if state.get("no_improve_rounds", 0) >= config.PATIENCE:
        print(f">> Patience exhausted ({config.PATIENCE} non-improving rounds)")
        return "finalize"
    return "refine"


# ---------------------------------------------------------------------------
# Node 4: Finalize (B6 CV confirm + A2 test-once + B5 ensemble)
# ---------------------------------------------------------------------------

def finalize_node(state: AgentState) -> dict:
    print("\n=== [4] Finalize: CV confirmation, test scoring, ensemble ===")
    metric = config.METRIC
    finalists = STORE.top_k_diverse(metric, config.ENSEMBLE_TOP_K)
    if not finalists:
        return {"final": {"error": "no successful candidates"},
                "stop_reason": state.get("stop_reason") or "no_success"}

    # B6: confirm finalists with expanding-window CV (selection still val-only).
    cv_scores = {}
    for n in finalists:
        r = harness.evaluate(n.code, f"final_cv_node{n.id}", mode="cv",
                             best_params=n.best_params)
        cv_scores[n.id] = r.score(metric) if r.status == "ok" else float("inf")
        print(f">> node {n.id}: val={n.score(metric):.4f}  cv={cv_scores[n.id]:.4f}")
    chosen = min(finalists, key=lambda n: cv_scores[n.id])

    # A2: the report-only test slice, touched exactly once per finalist.
    preds, test_scores = {}, {}
    for n in finalists:
        pred_path = os.path.join(config.CACHE_DIR, f"test_pred_node{n.id}.npz")
        r = harness.evaluate(n.code, f"final_test_node{n.id}", mode="test",
                             best_params=n.best_params, predictions_path=pred_path)
        if r.status == "ok":
            test_scores[n.id] = r.test_scores
            preds[n.id] = np.load(pred_path)
            print(f">> node {n.id}: test {metric} = {r.test_scores.get(metric, float('nan')):.4f}")
        else:
            print(f">> node {n.id}: test evaluation failed ({r.status})")

    # B5: ensemble with val-derived weights (never tuned on test).
    ensemble = {}
    if len(preds) >= 2:
        ids = list(preds)
        w = np.array([1.0 / max(STORE.get(i).score(metric), 1e-9) for i in ids])
        w /= w.sum()
        blend = sum(wi * preds[i]["y_pred"] for wi, i in zip(w, ids))
        y_true = preds[ids[0]]["y_true"]
        ensemble = {
            "members": ids,
            "weights": [round(float(x), 4) for x in w],
            "test_scores": {m: fn(y_true, blend) for m, fn in METRICS.items()},
        }
        print(f">> Ensemble test {metric} = {ensemble['test_scores'][metric]:.4f}")

    final = {
        "chosen_node": chosen.id,
        "chosen_val": chosen.score(metric),
        "cv_scores": cv_scores,
        "test_scores": {str(k): v for k, v in test_scores.items()},
        "ensemble": ensemble,
    }

    with open(os.path.join(config.OUTPUT_DIR, "final_best_model.py"), "w",
              encoding="utf-8") as f:
        f.write(chosen.code)
    with open(os.path.join(config.OUTPUT_DIR, "final_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({**final, "noise": state.get("noise", {}),
                   "tokens": llm.LEDGER.summary()}, f, indent=2, default=str)
    return {"final": final,
            "stop_reason": state.get("stop_reason") or "completed"}


# ---------------------------------------------------------------------------
# Node 5: Report (Analyst + D4 lessons)
# ---------------------------------------------------------------------------

def report_node(state: AgentState) -> dict:
    print("\n=== [5] Report ===")
    metric = config.METRIC
    final = state.get("final", {})
    chosen_id = final.get("chosen_node")
    chosen_test = final.get("test_scores", {}).get(str(chosen_id), {})
    ens_test = (final.get("ensemble") or {}).get("test_scores", {})

    try:
        report = llm.call_reasoner(prompts.ANALYST.format(
            task=state["task"],
            test_days=config.TEST_DAYS, horizon=config.HORIZON_DAYS,
            min_delta=state.get("min_delta", config.MIN_DELTA_ABS), metric=metric,
            best_val=final.get("chosen_val", float("nan")),
            best_test=round(chosen_test.get(metric, float("nan")), 4),
            k=config.ENSEMBLE_TOP_K,
            ensemble_test=round(ens_test.get(metric, float("nan")), 4) if ens_test else "n/a",
            nodes_md=_nodes_md(metric),
            ablation_md=_ablation_md(state.get("ablation", {})),
            token_summary=json.dumps(llm.LEDGER.summary()),
            wall_minutes=(time.time() - state.get("started_at", time.time())) / 60,
            stop_reason=state.get("stop_reason", ""),
        ))
    except Exception as e:
        report = f"# Report generation failed\n\n{e}\n\nSee outputs/final_results.json."

    with open(os.path.join(config.OUTPUT_DIR, "analysis_report.md"), "w",
              encoding="utf-8") as f:
        f.write(report)

    # D4: cross-run lessons ledger (dated, task-scoped, capped).
    try:
        summary = (
            f"task: {state['task']}\nbest val {metric}: {final.get('chosen_val')}\n"
            f"test: {chosen_test}\nensemble: {ens_test}\n"
            f"strategies that won: "
            + "; ".join(n.strategy for n in STORE.frontier(metric) if n.strategy)
        )
        lessons = llm.call_reasoner(prompts.LESSONS.format(summary=summary))
        _append_lessons(state["task"], lessons)
    except Exception as e:
        print(f">> Lessons ledger skipped: {e}")

    return {"report": report}


def _append_lessons(task: str, lessons: str, cap: int = 10) -> None:
    import datetime

    header = f"## {datetime.date.today()} | {task}"
    entry = f"{header}\n{lessons.strip()}\n"
    existing = ""
    if os.path.exists(config.LESSONS_PATH):
        with open(config.LESSONS_PATH, encoding="utf-8") as f:
            existing = f.read()
    sections = [s for s in existing.split("\n## ") if s.strip()]
    sections = sections[-(cap - 1):] if len(sections) >= cap else sections
    body = "\n## ".join(sections)
    if body and not body.startswith("##"):
        body = "## " + body
    with open(config.LESSONS_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Learned lessons (auto-maintained, last {cap} runs)\n\n"
                + (body + "\n\n" if body else "") + entry)
