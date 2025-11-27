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
# LLM 設定 (保持不變)
# =========================
llm_reasoner = ChatOllama(
    model="deepseek-v3.1:671b-cloud",   # 記得要加 :cloud 或是完整 tag
    base_url="https://ollama.com",
    headers=ollama_cloud_headers,     # 傳送驗證資訊
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
# State 定義 (大幅增強以符合加分項 #7)
# =========================
class ExperimentLog(TypedDict):
    iteration: int
    component: str          # 修改了哪個組件 (如: Feature Engineering, Model Params)
    strategy: str           # 具體策略 (如: Entity Embeddings)
    citation: str           # 引用來源 (論文或方法論)
    mape: float
    status: str
    reasoning: str          # 為什麼做這個改動

class AgentState(TypedDict):
    task_description: str     # 任務描述
    design_spec: str
    code: str
    mape_score: float
    iteration_count: int
    best_code: str
    best_mape: float
    execution_log: str
    citations: List[str]
    report: str
    # [NEW] 結構化的歷史紀錄，用於生成詳細報告
    history: List[ExperimentLog]


# =========================
# Node 1: Research / Design (加入動態 Task Description)
# =========================
def search_node(state: AgentState):
    print("\n=== [Step 1] Research Agent: Deep Researching... ===")
    
    task_desc = state.get("task_description", "Kaggle Rossmann Store Sales forecasting")
    
    tavily_tool = TavilySearchResults(max_results=5)
    arxiv_tool = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
    
    results_text = ""
    
    # 策略 A: 找論文
    print(">> Searching Arxiv for academic context...")
    try:
        arxiv_result = arxiv_tool.run("machine learning time series forecasting entity embeddings")
        results_text += f"\n--- Arxiv Papers ---\n{arxiv_result}\n"
    except Exception as e:
        print(f"Arxiv search failed: {e}")

    # 策略 B: 找 Kaggle 實戰 Code
    print(">> Searching Web (Tavily) for Kaggle solutions...")
    try:
        tavily_results = tavily_tool.invoke(f"{task_desc} kaggle winner solution feature engineering")
        web_content = "\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in tavily_results])
        results_text += f"\n--- Web Solutions ---\n{web_content}\n"
    except Exception as e:
        print(f"Tavily search failed: {e}")

    # LLM 分析
    citation_prompt = f"""
    You are a Research Scientist. 
    Analyze the following search results for the task: "{task_desc}".
    
    Search Data:
    {results_text}
    
    Output a list of 3-5 key references (Paper titles or Kaggle solution names) that we should implement.
    Focus on High-Impact Feature Engineering (e.g., Date parts, Lag features, Embeddings).
    """
    
    citation_resp = llm_reasoner.invoke([HumanMessage(content=citation_prompt)])
    citations = [line.strip("- \"'") for line in citation_resp.content.splitlines() if line.strip()]

    # 生成 Spec
    spec_prompt = f"""
    Design a machine learning pipeline for {task_desc} based on these references.
    References: {citations}
    
    Include sections for:
    1. Data Preprocessing (Handling missing values, Outliers).
    2. Feature Engineering (Date features, Competitor details).
    3. Model Selection (XGBoost/LightGBM).
    4. Evaluation Metric (MAPE).
    """
    spec_resp = llm_reasoner.invoke([HumanMessage(content=spec_prompt)])

    return {
        "design_spec": spec_resp.content,
        "citations": citations,
        "history": [],
        "iteration_count": 0,
        "best_mape": float("inf"),
        "best_code": ""
    }

# =========================
# Node 2: Foundation Coder (加入 Data Leakage 防護指令)
# =========================
def foundation_node(state: AgentState):
    print("\n=== [Step 2] MLE Coder: Writing Base Training Script ===")
    spec = state["design_spec"]

    # [MODIFIED] 加入了針對 Rossmann 的特定防護規則 (Sales > 0)
    prompt = f"""
    You are an expert ML engineer. Write a COMPLETE Python script for Rossmann Sales.
    
    Design Spec:
    {spec}

    **CRITICAL RULES (Strictly Follow):**
    1. **HANDLING ZEROS (CRITICAL)**: 
        - You MUST filter 'Sales > 0' and 'Open == 1' for BOTH training AND validation sets.
        - Do this filtering BEFORE doing any Train-Test Split.
        - Code pattern: 
            df = df[df['Sales'] > 0]
            train, val = train_test_split(df, ...)
    2. **HARDWARE ACCELERATION (IMPORTANT)**:
       - Detect if a GPU is available (e.g., Check CUDA).
       - If using **XGBoost**: Set `tree_method='gpu_hist'` (if GPU exists) else `tree_method='hist'`.
       - If using **LightGBM**: Set `device='gpu'` (if GPU exists) else `device='cpu'`.
       - This is crucial for training speed on this dataset.
    3. **Data Leakage Prevention**: When preprocessing, DO NOT use statistics from the Validation set to scale Training data.
    4. **Data Usage**: Merge 'train.csv' and 'store.csv' on 'Store'.
    5. **Output**: 
       - Print "FINAL_MAPE: <value>" at the end.
       - Use `plt.switch_backend('Agg')` if plotting.
    
    Output ONLY valid Python code. No markdown text.
    """

    response = llm_coder.invoke([HumanMessage(content=prompt)])
    code = response.content.strip().replace("```python", "").replace("```", "").strip()

    return {
        "code": code,
        "best_code": code,
    }

# =========================
# Node 3: Refinement Loop (核心修改：模擬 Planner + Coder)
# =========================
def refinement_node(state: AgentState):
    current_iteration = state["iteration_count"] + 1
    print(f"\n=== [Step 3] Optimization Agent: Iteration {current_iteration}/{MAX_ITERATIONS} ===")
    
    # 1. 執行與評估
    current_code = state["code"]
    filename = f"train_iter_{current_iteration}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(current_code)

    print(f">> Executing {filename}...")
    try:
        # [MODIFIED] 增加 timeout 時間，確保模型訓練跑得完
        result = subprocess.run([sys.executable, filename], capture_output=True, text=True, timeout=1800)
        output = result.stdout + "\n" + result.stderr
        
        # 抓取 MAPE
        m = re.search(r"(FINAL_MAPE|Final MAPE|MAPE)\s*[:=]\s*([0-9.inf]+)", output, re.IGNORECASE)
        mape = float(m.group(2)) if m else float("inf")
        # 雙重檢查 inf
        if "inf" in str(mape).lower() or ("inf" in output.lower() and "final_mape" in output.lower()):
            mape = float("inf")
        
        print(f">> Result MAPE: {mape}")
    except Exception as e:
        output = str(e)
        mape = float("inf")
        print(f">> Execution Failed: {e}")

    # 2. 更新 Best Model 與 決定 Rollback
    best_mape = state["best_mape"]
    best_code = state["best_code"]
    execution_status = "Success"
    
    code_to_optimize = current_code # 預設：繼續使用當前的 code
    
    # 情況 A: 找到更好的結果
    if mape < best_mape:
        best_mape = mape
        best_code = current_code
        print(f">> New Best Found! ({mape})")
        # 既然變好了，當然基於這版繼續改
        code_to_optimize = current_code 

    # 情況 B: 結果沒變好 (或是 inf)
    else:
        print(f">> No Improvement (Current: {mape}, Best: {best_mape})")
        
        if mape == float("inf"):
            execution_status = "Failed" # 標記失敗，讓 Planner 知道要 Debug
            
            # === [關鍵修改 START] ===
            # 如果目前的最佳紀錄也是 inf，代表我們根本還沒任何能用的版本。
            # 這時候回滾也沒用（因為舊的也是爛的）。
            # 不如讓 AI 看著這次報錯的 code 嘗試修復它。
            if best_mape == float("inf"):
                print(">> Best score is still INF. NOT reverting. Forcing Agent to fix current broken code.")
                code_to_optimize = current_code
            else:
                # 只有當我們「曾經」有過正常的版本時，才執行回滾保護
                print(">> CRITICAL: Code broke. Reverting to previous BEST code.")
                code_to_optimize = best_code
            # === [關鍵修改 END] ===
            
        else:
            # 如果只是沒變好但程式沒壞 (例如 MAPE 0.25 -> 0.26)
            # 這裡可以選擇回滾，也可以選擇不回滾 (Exploration)。
            # 保守策略是回滾：
            code_to_optimize = best_code
    # 3. 判斷是否終止
    if current_iteration >= MAX_ITERATIONS:
        return {
            "iteration_count": current_iteration, 
            "mape_score": mape, 
            "best_mape": best_mape, 
            "execution_log": output
        }

    # ============================================================
    # Planner Agent
    # ============================================================
    print(">> Generating Refinement Plan...")
    history_text = "\n".join([f"Iter {h['iteration']}: {h['strategy']} -> MAPE {h['mape']} ({h['status']})" for h in state["history"]])
    
    # [MODIFIED] 擴大 Log Context 到 2000 字元，以免漏掉 traceback
    log_snippet = output[-2000:] 

    plan_prompt = f"""
    You are the Planner Agent. Analyze the current execution.
    
    Current Status: {execution_status}
    Current MAPE: {mape}
    Global Best MAPE: {best_mape}
    
    History:
    {history_text}
    
    Execution Log Snippet:
    {log_snippet}

    **Task:**
    1. Identify the WEAKEST component.
    2. Propose ONE specific improvement strategy.
    3. **IMPORTANT**: If Status is 'Failed', analyze the error and propose a FIX or a simpler alternative.
       If Status is 'Success' but stuck, propose a NEW feature (e.g., Log transform sales, Add Store Type embedding).
    
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
    except:
        plan_data = {
            "component": "Recovery",
            "strategy": "Fix syntax and simplify model",
            "citation": "Debugging Best Practices",
            "reasoning": "Previous iteration failed parsing."
        }

    print(f">> Plan: {plan_data['strategy']}")

    # ============================================================
    # Coder Agent
    # ============================================================
    print(">> Implementing Plan...")
    
    # [MODIFIED] Coder 現在會明確知道它是在修改哪個版本的 Code (Base Code)
    code_prompt = f"""
    You are the Coder Agent. 
    Refine the Python script based strictly on this plan.
    
    PLAN: {plan_data['strategy']}
    REASON: {plan_data['reasoning']}
    
    **BASE CODE to Modify**:
    {code_to_optimize}
    
    **Constraints**:
    - Modify the BASE CODE provided above.
    - Focus changes on: {plan_data['component']}.
    - Ensure `train = train[train['Sales'] > 0]` is PRESERVED.
    - **Ensure GPU parameters (e.g., `tree_method='gpu_hist'`) are PRESERVED if previously set.**
    - Ensure `FINAL_MAPE` is printed.
    
    Output ONLY the full Python code.
    """
    code_resp = llm_coder.invoke([HumanMessage(content=code_prompt)])
    new_code = code_resp.content.strip().replace("```python", "").replace("```", "").strip()

    new_log: ExperimentLog = {
        "iteration": current_iteration,
        "component": plan_data['component'],
        "strategy": plan_data['strategy'],
        "citation": plan_data['citation'],
        "mape": mape,
        "status": execution_status,
        "reasoning": plan_data['reasoning']
    }
    
    return {
        "code": new_code,
        "mape_score": mape,
        "iteration_count": current_iteration,
        "best_mape": best_mape,
        "best_code": best_code,
        "execution_log": output,
        "history": state["history"] + [new_log]
    }

# =========================
# Node 4: Report Writer (優化以展現加分項 #7)
# =========================
def report_node(state: AgentState):
    print("\n=== [Step 4] Analyst Agent: Writing Final Report ===")
    
    history = state["history"]
    best_mape = state["best_mape"]
    
    version_table = "| Iter | Strategy | Status | Result (MAPE) |\n|---|---|---|---|\n"
    for h in history:
        version_table += f"| {h['iteration']} | {h['strategy']} | {h['status']} | {h['mape']} |\n"

    prompt = f"""
    You are a Senior Data Scientist. Write a final report for the Rossmann Sales Prediction task.
    
    **Project Context**:
    - Automated by MLE-STAR Agent.
    - Best Validation MAPE: {best_mape}
    
    **Experiment History**:
    {version_table}
    
    **Task**:
    Write a Markdown report including:
    1. **Executive Summary**: What was achieved?
    2. **Methodology**: Explain the automated refinement process.
    3. **Key Improvements**: Which strategies worked best? (Refer to the table).
    4. **Challenges**: Discuss any failed iterations (Infinite MAPE, etc.) and how the agent recovered.
    
    Output ONLY Markdown.
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
        initial_state = {"task_description": "Rossmann Store Sales forecasting using train.csv and store.csv"}
        result = app.invoke(initial_state)

        with open("final_best_model.py", "w", encoding="utf-8") as f:
            f.write(result["best_code"])
        
        with open("analysis_report.md", "w", encoding="utf-8") as f:
            f.write(result["report"])
            
        print(f"Done. Best MAPE: {result['best_mape']}")