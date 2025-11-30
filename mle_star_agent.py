import os
import re
import subprocess
import sys
import json
from typing import TypedDict, List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
# from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from dotenv import load_dotenv

load_dotenv()
ollama_cloud_headers = {
    "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
}

# =========================
# LLM 設定
# =========================
llm_reasoner = ChatOllama(
    model="deepseek-v3.1:671b-cloud",   # 記得要加 :cloud 或是完整 tag
    base_url="https://ollama.com",
    headers=ollama_cloud_headers,       # 傳送驗證資訊
    temperature=0.1,
    keep_alive="30m"
)
llm_coder = ChatOllama(
    model="qwen3-coder:480b-cloud",
    base_url="https://ollama.com",
    headers=ollama_cloud_headers,
    temperature=0.1
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MAX_ITERATIONS = 10

# =========================
# State 定義（加強 reward / memory）
# =========================
class ExperimentLog(TypedDict):
    iteration: int
    component: str           # 修改了哪個組件 (如: Feature Engineering, Model Params)
    strategy: str            # 具體策略 (如: Entity Embeddings)
    citation: str            # 引用來源 (論文或方法論)
    mape: float
    status: str
    reasoning: str           # 為什麼做這個改動
    reward: float            # 根據 MAPE 推出的 reward（例如 -MAPE）


class AgentState(TypedDict):
    task_description: str      # 任務描述
    design_spec: str
    code: str
    mape_score: float
    iteration_count: int
    best_code: str
    best_mape: float
    execution_log: str
    citations: List[str]
    report: str
    history: List[ExperimentLog]   # 結構化的歷史紀錄，用於生成詳細報告
    no_improve_rounds: int         # 連續未改善的輪數


# =========================
# Node 1: Research / Design
# - 強制讀取題目指定的 Google 論文 (arxiv:2506.15692)
# =========================
def search_node(state: AgentState):
    print("\n=== [Step 1] Research Agent: Deep Researching... ===")

    task_desc = state.get("task_description", "Kaggle Rossmann Store Sales forecasting")

    # Arxiv / Web 搜尋工具
    arxiv_tool = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1000)

    # Tavily 只在有 API key 時啟用，避免沒有 key 就報錯
    tavily_tool = None
    if TAVILY_API_KEY:
        tavily_tool = TavilySearchResults(max_results=5)

    results_text = ""
    core_paper_text = ""

    # 策略 0: 題目指定的 Google 論文 (arxiv:2506.15692)
    print(">> Fetching core Google paper (arxiv:2506.15692)...")
    try:
        # 這裡用 paper id 當作查詢關鍵字
        core_paper_text = arxiv_tool.run("2506.15692")
        results_text += "\n--- Core Paper (Google 2506.15692) ---\n" + core_paper_text + "\n"
    except Exception as e:
        print(f"Core paper fetch failed: {e}")

    # 策略 A: 找其他相關論文
    print(">> Searching Arxiv for additional academic context...")
    try:
        arxiv_result = arxiv_tool.run("machine learning time series forecasting entity embeddings Rossmann")
        results_text += f"\n--- Additional Arxiv Papers ---\n{arxiv_result}\n"
    except Exception as e:
        print(f"Arxiv search failed: {e}")

    # 策略 B: 找 Kaggle 實戰 Code
    if tavily_tool is not None:
        print(">> Searching Web (Tavily) for Kaggle solutions...")
        try:
            tavily_results = tavily_tool.invoke(f"{task_desc} kaggle winner solution feature engineering")
            web_content = "\n".join(
                [f"Source: {res['url']}\nContent: {res['content']}" for res in tavily_results]
            )
            results_text += f"\n--- Web Solutions ---\n{web_content}\n"
        except Exception as e:
            print(f"Tavily search failed: {e}")
    else:
        print(">> Tavily API key not set; skipping web search.")

    # LLM 分析：整理核心引用
    citation_prompt = f"""
You are a Research Scientist.
You are designing an automated ML agent for the task: "{task_desc}".

We have a **core Google paper** that MUST be treated as a primary reference:
- Google ML Sales Forecasting Agent (arxiv:2506.15692)

Core Paper (raw text, possibly truncated):
{core_paper_text}

Other Search Data:
{results_text}

Task:
1. Extract 3-5 key references (paper titles, Kaggle solutions, or method names) that we should implement.
2. One of the references MUST explicitly be "Google ML Sales Forecasting Agent (arxiv:2506.15692)".
3. Focus on:
   - High-Impact Feature Engineering (date parts, lag features, store embeddings, promo effects)
   - ML agent architecture ideas from the Google paper.

Output each reference on its own line, no extra explanation.
"""
    citation_resp = llm_reasoner.invoke([HumanMessage(content=citation_prompt)])
    citations = [line.strip("- \"'") for line in citation_resp.content.splitlines() if line.strip()]


    core_ref = "Google ML Sales Forecasting Agent (arxiv:2506.15692)"
    if not any("2506.15692" in c or "Sales Forecasting Agent" in c for c in citations):
        citations.insert(0, core_ref)

    print(">> Research Agent Citations:")
    for c in citations:
        print(f"   - {c}")


    spec_prompt = f"""
Design a machine learning pipeline for {task_desc} based on these references.

References:
{os.linesep.join(citations)}

You MUST explicitly align the overall architecture with the ideas from:
- Google ML Sales Forecasting Agent (arxiv:2506.15692)

The design should include sections for:
1. Data Preprocessing
   - Merging train.csv and store.csv on 'Store'
   - Handling missing values and outliers
   - Filtering out closed stores and zero-sales days if appropriate

2. Feature Engineering
   - Calendar / date features (e.g., year, month, week, day-of-week, promo periods)
   - Store-level features (e.g., store type, competition distance, competition open date)
   - Promotion / holiday related features
   - Any agentic or automated ideas inspired by the Google paper

3. Model Selection
   - Favor gradient boosting models (XGBoost / LightGBM) as a strong baseline
   - Mention how the agent can switch / compare models

4. Evaluation Metric
   - Use MAPE as the primary metric
   - Mention how to avoid infinite MAPE when ground-truth sales are zero

Also briefly describe how an automated ML agent (MLE-STAR-style) will:
- Search for better features
- Tune hyperparameters
- Iterate based on validation MAPE.
"""
    spec_resp = llm_reasoner.invoke([HumanMessage(content=spec_prompt)])

    return {
        "design_spec": spec_resp.content,
        "citations": citations,
        "history": [],
        "iteration_count": 0,
        "best_mape": float("inf"),
        "best_code": "",
        "no_improve_rounds": 0,
    }


# =========================
# Node 2: Foundation Coder
# - 加強 MAPE 安全性與資料過濾規則
# =========================
def foundation_node(state: AgentState):
    print("\n=== [Step 2] MLE Coder: Writing Base Training Script ===")
    spec = state["design_spec"]

    prompt = f"""
You are an expert ML engineer. Write a COMPLETE Python script for Rossmann Store Sales forecasting.

Design Spec:
{spec}

**CRITICAL RULES (Strictly Follow):**

1. DATA MERGE & FILTERING (CRITICAL):
   - Read 'train.csv' and 'store.csv', then merge them on 'Store' into a single DataFrame named `df`.
   - You MUST filter to keep only rows where:
       - Sales > 0
       - Open == 1
   - This filtering MUST happen **BEFORE** any train/validation/time-series split.
   - Example:
       df = df[(df['Sales'] > 0) & (df['Open'] == 1)]

2. HARDWARE ACCELERATION (IMPORTANT):
   - Detect if a GPU is available (e.g., check CUDA).
   - If using XGBoost:
       - Set `tree_method='gpu_hist'` when GPU is available, otherwise `tree_method='hist'`.
   - If using LightGBM:
       - Set `device='gpu'` when GPU is available, otherwise `device='cpu'`.
   - This is crucial for training speed on this dataset.

3. DATA LEAKAGE PREVENTION:
   - When preprocessing (scaling/encoding), fit transformers ONLY on the training set.
   - Apply the fitted transformers to the validation set.
   - Do NOT use any statistics from the validation set to transform the training data.

4. DATA USAGE:
   - Use 'train.csv' and 'store.csv' files located in the current working directory.
   - Correctly parse dates and use them for feature engineering (e.g. extracting year, month, day-of-week).

5. MAPE IMPLEMENTATION (CRITICAL, TO AVOID INF):
   - Implement MAPE in a numerically safe way that does NOT produce 'inf' even if some targets are zero.
   - One acceptable pattern is:

       import numpy as np

       def safe_mape(y_true, y_pred):
           y_true = np.array(y_true)
           y_pred = np.array(y_pred)
           # Avoid division by zero:
           y_true_safe = np.where(y_true == 0, 1, y_true)
           return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

   - You may instead drop rows where y_true == 0 **before** computing MAPE.
   - At the end of training/validation, compute the final validation MAPE and assign it to a variable `final_mape`.
   - Print the final result EXACTLY as:

       print(f"FINAL_MAPE: {{final_mape}}")

6. OUTPUT & PLOTTING:
   - If you generate plots, call `import matplotlib.pyplot as plt` and use `plt.switch_backend('Agg')` at the top to avoid GUI issues.
   - You may optionally save a few key plots (e.g. feature importance) to disk.

The script must:
- Be fully executable as `python train_iter_X.py` in the current directory.
- Not require any manual input.
- Use reasonable defaults for hyperparameters.

Output ONLY valid Python code. Do NOT wrap it in markdown fences.
"""
    response = llm_coder.invoke([HumanMessage(content=prompt)])
    code = response.content.strip().replace("```python", "").replace("```", "").strip()

    return {
        "code": code,
        "best_code": code,
    }


# =========================
# Node 3: Refinement Loop
# - Planner + Coder + Reward / Memory
# =========================
def refinement_node(state: AgentState):
    current_iteration = state["iteration_count"] + 1
    print(f"\n=== [Step 3] Optimization Agent: Iteration {current_iteration}/{MAX_ITERATIONS} ===")

    current_code = state["code"]
    filename = f"train_iter_{current_iteration}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(current_code)

    print(f">> Executing {filename}...")
    try:
        # 增加 timeout，確保模型訓練跑得完
        result = subprocess.run(
            [sys.executable, filename],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output = result.stdout + "\n" + result.stderr

        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"iter_{current_iteration}.log")

        global_citations = state.get("citations", [])

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== Global Citations from Research Agent ===\n")
            if global_citations:
                for c in global_citations:
                    f.write(f"- {c}\n")
            else:
                f.write("(No citations found in state)\n")

            f.write("\n=== Raw Execution Output ===\n")
            f.write(output)

        m = re.search(r"(FINAL_MAPE|Final MAPE|MAPE)\s*[:=]\s*([0-9eE\.\+\-infINF]+)", output)
        if m:
            mape_str = m.group(2).strip()
            if "inf" in mape_str.lower():
                mape = float("inf")
            else:
                try:
                    mape = float(mape_str)
                except ValueError:
                    mape = float("inf")
        else:
            mape = float("inf")

        print(f">> Result MAPE: {mape}")
    except Exception as e:
        output = str(e)
        mape = float("inf")
        print(f">> Execution Failed: {e}")

    # Reward：MAPE 越小 reward 越高
    reward = -mape if mape != float("inf") else -1e9

    best_mape = state["best_mape"]
    best_code = state["best_code"]
    execution_status = "Success"
    no_improve_rounds = state.get("no_improve_rounds", 0)

    code_to_optimize = current_code  # 預設：基於當前版本繼續優化

    # 情況 A: 找到更好的結果
    if mape < best_mape:
        best_mape = mape
        best_code = current_code
        no_improve_rounds = 0
        print(f">> New Best Found! (MAPE = {mape})")

    # 情況 B: 沒有變好
    else:
        if mape == float("inf"):
            execution_status = "Failed"
            no_improve_rounds += 1

            # 若目前 best 也是 inf，就沒有 rollback 的意義
            if best_mape == float("inf"):
                print(">> Best score is still INF. Not reverting. Forcing Agent to fix current broken code.")
                code_to_optimize = current_code
            else:
                print(">> CRITICAL: Code broke. Reverting to previous BEST code.")
                code_to_optimize = best_code
        else:
            # 結果沒變好但程式有跑完：採保守策略回滾
            no_improve_rounds += 1
            print(f">> No Improvement (Current: {mape}, Best: {best_mape})")
            code_to_optimize = best_code

    # 儲存這一輪的實驗紀錄（先填入共通欄位，稍後由 Planner 補 component/strategy/citation/reasoning）
    new_log: ExperimentLog = {
        "iteration": current_iteration,
        "component": "",
        "strategy": "",
        "citation": "",
        "mape": mape,
        "status": execution_status,
        "reasoning": "",
        "reward": reward,
    }

    history = state["history"] + [new_log]

    # 3. 判斷是否終止（iteration 或連續未改善輪數）
    if current_iteration >= MAX_ITERATIONS or no_improve_rounds >= 3:
        print(f">> Stopping criteria reached. Iterations={current_iteration}, no_improve_rounds={no_improve_rounds}")
        return {
            "iteration_count": current_iteration,
            "mape_score": mape,
            "best_mape": best_mape,
            "execution_log": output,
            "best_code": best_code,
            "history": history,
            "no_improve_rounds": no_improve_rounds,
        }

    # ============================================================
    # Planner Agent
    # ============================================================
    print(">> Generating Refinement Plan...")
    history_text = "\n".join(
        [
            f"Iter {h['iteration']}: MAPE = {h['mape']} (Status: {h['status']}, Reward: {h['reward']})"
            for h in history
        ]
    )

    # 擴大 Log Context 到 2000 字元，以免漏掉 traceback
    log_snippet = output[-2000:]

    plan_prompt = f"""
You are the Planner Agent of an automated ML system for Rossmann Sales forecasting.

Current Status: {execution_status}
Current MAPE: {mape}
Global Best MAPE: {best_mape}
Consecutive Non-Improving Rounds: {no_improve_rounds}

History:
{history_text}

Execution Log Snippet (last 2000 chars):
{log_snippet}

Task:
1. Identify the WEAKEST component of the current pipeline
   (e.g., feature engineering, model choice, hyperparameters, data filtering, MAPE calculation).
2. Propose ONE specific, concrete improvement strategy.
3. If Status is 'Failed', analyze the error and propose a FIX or a simpler alternative.
4. If Status is 'Success' but stuck (no improvement), propose a NEW idea
   (e.g., adding a log-transform of Sales, using more calendar features, or tuning a key hyperparameter).
5. Whenever possible, reference either:
   - Google ML Sales Forecasting Agent (arxiv:2506.15692), or
   - Kaggle Rossmann winner solutions / related literature.

Output strictly in JSON:
{{
  "component": "...",
  "strategy": "...",
  "citation": "...",
  "reasoning": "..."
}}
"""
    try:
        plan_resp = llm_reasoner.invoke([HumanMessage(content=plan_prompt)])
        plan_text = plan_resp.content.strip()
        if "```json" in plan_text:
            plan_text = plan_text.split("```json")[1].split("```")[0]
        elif "```" in plan_text:
            plan_text = plan_text.split("```")[1].split("```")[0]
        plan_data = json.loads(plan_text)
    except Exception:
        plan_data = {
            "component": "Recovery",
            "strategy": "Fix syntax / runtime errors and simplify the model configuration.",
            "citation": "Debugging Best Practices",
            "reasoning": "Previous iteration failed or returned invalid JSON.",
        }

    print(f">> Plan Component: {plan_data['component']}")
    print(f">> Plan Strategy: {plan_data['strategy']}")

    history[-1]["component"] = plan_data["component"]
    history[-1]["strategy"] = plan_data["strategy"]
    history[-1]["citation"] = plan_data["citation"]
    history[-1]["reasoning"] = plan_data["reasoning"]

    # ============================================================
    # Coder Agent
    # ============================================================
    print(">> Implementing Plan...")

    code_prompt = f"""
You are the Coder Agent.
Refine the Python training script for Rossmann Sales forecasting based strictly on this plan.

PLAN: {plan_data['strategy']}
REASON: {plan_data['reasoning']}

**BASE CODE to Modify**:
{code_to_optimize}

**Constraints**:
- Modify the BASE CODE provided above; do not create an entirely new file structure.
- Focus your changes on the component: {plan_data['component']}.
- Ensure the merged DataFrame is still named `df`.
- Ensure the line `df = df[(df['Sales'] > 0) & (df['Open'] == 1)]`
  (or an equivalent boolean mask) exists **exactly once** and is executed
  **before any train/validation split**. Do NOT move or remove this filtering.
- Ensure GPU-related parameters (e.g., `tree_method='gpu_hist'` for XGBoost,
  `device='gpu'` for LightGBM) are preserved if they were previously set.
- Ensure the MAPE implementation is numerically stable and does NOT produce 'inf'
  even if some targets are zero (e.g., use a safe denominator like `y_true_safe = np.where(y_true == 0, 1, y_true)`).
- Ensure the script prints the final validation metric as:
    print(f"FINAL_MAPE: {{final_mape}}")

Output ONLY the full Python code. Do NOT wrap it in markdown fences.
"""
    code_resp = llm_coder.invoke([HumanMessage(content=code_prompt)])
    new_code = code_resp.content.strip().replace("```python", "").replace("```", "").strip()

    return {
        "code": new_code,
        "mape_score": mape,
        "iteration_count": current_iteration,
        "best_mape": best_mape,
        "best_code": best_code,
        "execution_log": output,
        "history": history,
        "no_improve_rounds": no_improve_rounds,
    }


# =========================
# Node 4: Report Writer
# - 說明 multi-agent 架構 & version history
# =========================
def report_node(state: AgentState):
    print("\n=== [Step 4] Analyst Agent: Writing Final Report ===")

    history = state["history"]
    best_mape = state["best_mape"]

    version_table = "| Iter | Strategy | Status | Result (MAPE) |\n|---|---|---|---|\n"
    for h in history:
        version_table += f"| {h['iteration']} | {h['strategy']} | {h['status']} | {h['mape']} |\n"

    history_bullets = "\n".join(
        [
            f"- Iter {h['iteration']}: component = {h['component']}, "
            f"strategy = {h['strategy']}, citation = {h['citation']}, "
            f"MAPE = {h['mape']}, reward = {h['reward']}, status = {h['status']}"
            for h in history
        ]
    )

    prompt = f"""
You are a Senior Data Scientist. Write a **professional Markdown report** for the Rossmann Sales Prediction task.

**Project Context**:
- The entire pipeline was automated by an MLE-STAR style **multi-agent system**.
- Best Validation MAPE achieved: {best_mape}

**Experiment History (raw data)**:
Table:
{version_table}

Detailed logs:
{history_bullets}

Task:
Write a Markdown report including the following sections:

1. **Executive Summary**
   - High-level overview of what was built.
   - Final best MAPE and a short interpretation (e.g., is this acceptable, where it might fail).

2. **Methodology & Agent Architecture**
   - Describe the multi-agent architecture used in this project:
     - Research Agent (searches papers and Kaggle solutions, outputs citations and design_spec)
     - Foundation Coder Agent (writes the first training script)
     - Planner Agent (decides what to change based on logs and history)
     - Coder Agent (edits the code according to the plan)
     - Evaluator/Rewarder (runs the code, measures MAPE, updates best_mape)
     - Analyst Agent (this report writer)
   - Explain how this architecture is inspired by the Google ML Sales Forecasting Agent (arxiv:2506.15692).

3. **Key Improvements and References**
   - Summarize the main strategies that led to MAPE improvements.
   - For each major improvement, mention:
     - What was changed (e.g., new features, model tuning, bug fix).
     - Which reference (paper, Kaggle solution) motivated it.
     - How the MAPE changed relative to previous iterations.

4. **Agent Learning Mechanism (Memory, Knowledge Ingestion, Reward)**
   - Explain:
     - How `history` acts as memory across iterations.
     - How `best_mape` and the derived reward guide the Planner's decisions.
     - How external knowledge (citations, Google paper, Kaggle solutions) is ingested into the system.
   - Emphasize that the agent is not just tuning hyperparameters directly, but refining its own pipeline.

5. **Versioned Experiments**
   - For each iteration in the history, write 1–2 sentences:
     - What changed in that version.
     - Which citation (if any) was used.
     - Whether MAPE improved or not compared to the previous iteration.
   - Present this as "Version 1, Version 2, ..." so that a reviewer can clearly see the evolution.

6. **Challenges and Limitations**
   - Discuss any iterations that failed (e.g., infinite MAPE, runtime errors) and how the agent recovered.
   - Mention practical constraints (e.g., hardware limits, training time, data issues).
   - Suggest future improvements (e.g., deeper models, better holiday features, or additional agent roles).

7. **Business Insights & Future Strategy Suggestions**
   - Briefly translate the modeling results into business language:
     - What patterns in sales were learned (e.g., importance of promo, seasonality).
     - What next steps a retail manager could take based on these insights.

Output ONLY valid Markdown. Do NOT include backticks around the whole report.
"""
    response = llm_reasoner.invoke([HumanMessage(content=prompt)])
    return {"report": response.content}


# =========================
# Workflow Setup
# =========================
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("foundation", foundation_node)
workflow.add_node("refine", refinement_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "foundation")
workflow.add_edge("foundation", "refine")


def should_continue(state: AgentState):
    # 同時考慮 iteration 上限與連續未改善輪數
    if state["iteration_count"] < MAX_ITERATIONS:
        return "refine"
    return "report"


workflow.add_conditional_edges("refine", should_continue, {"refine": "refine", "report": "report"})
workflow.add_edge("report", END)

app = workflow.compile()

if __name__ == "__main__":
    if not (os.path.exists("train.csv") and os.path.exists("store.csv")):
        print("Please ensure 'train.csv' and 'store.csv' are in the current directory.")
    else:
        print("Starting MLE-STAR Agent (Enhanced)...")
        initial_state: Dict[str, Any] = {
            "task_description": "Rossmann Store Sales forecasting using train.csv and store.csv"
        }
        result = app.invoke(initial_state)

        with open("final_best_model.py", "w", encoding="utf-8") as f:
            f.write(result["best_code"])

        with open("analysis_report.md", "w", encoding="utf-8") as f:
            f.write(result["report"])

        print(f"Done. Best MAPE: {result['best_mape']}")