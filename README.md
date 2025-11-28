# MLE-STAR: Autonomous Machine Learning Optimization Agent
## 針對 Rossmann Store Sales 預測任務的自動化機器學習代理系統
### 1. 專案概述 (Overview): 
MLE-STAR (Machine Learning Engineer - Self-Training & Refining Agent) 是一個基於 LangGraph 與 LangChain 構建的自主代理系統。本專案旨在全自動化 Kaggle 競賽級別的機器學習流程。透過整合 DeepSeek-V3 (Reasoner) 與 Qwen-Coder (Coder) 兩個大型語言模型，系統能夠自主進行文獻探討、程式碼撰寫、模型訓練、錯誤修復以及超參數優化，最終產出最佳化的預測模型與分析報告。目前專案已針對 Rossmann Store Sales 資料集進行了特化配置。

### 2. 核心架構與工作流 (Architecture & Workflow): 
本系統採用 StateGraph 狀態機架構，包含四個主要節點：
1. Research Agent (Search Node)功能：
    - 搜尋學術論文 (Arxiv) 與 Kaggle 優勝方案 (Tavily Search)。
    - 目的：獲取針對時間序列預測與實體嵌入 (Entity Embeddings) 的最新技術與特徵工程策略。
    - 輸出：生成一份包含特徵工程與模型選擇的 Design Spec。
2. Foundation Coder (Foundation Node)功能：
    - 根據設計規範撰寫初代 Python 訓練腳本。
    - 安全防護：內建強制性規則（如過濾 Sales > 0、防止 Data Leakage），確保基礎程式碼的正確性。
    - 硬體感知：自動偵測 GPU 並配置 XGBoost/LightGBM 的加速參數
3. Refinement Loop (Refinement Node) - 核心引擎

    這是一個迭代優化的循環過程（預設 10 次迭代）：
    - 執行 (Execute)：運行當前的 Python 訓練腳本。
    - 評估 (Evaluate)：解析輸出日誌，捕捉 MAPE 分數與錯誤訊息。
    - 決策 (Planner Agent)：
        - 若失敗 (Error/Inf)：分析 Traceback 並提出修復方案。
        - 若成功但未提升：提出新的特徵工程策略（如 Log Transform, Lag Features）。
    - Rollback 機制：若新程式碼導致效能下降或崩潰，系統會自動回滾至上一個最佳版本。- 實作 (Coder Agent)：根據計畫修改程式碼。
4. Analyst Agent (Report Node)功能：
    - 在迭代結束後，彙整所有的實驗紀錄 (ExperimentLog)。
    - 輸出：生成一份 Markdown 格式的總結報告，包含方法論、嘗試過的策略列表以及最佳 MAPE 成績。

### 3. 技術特點 (Key Features):
-  特點說明雙模型協作使用 DeepSeek 進行高層次推理與策略規劃，使用 Qwen-Coder 進行精準的程式碼生成。自我修復 (Self-Healing)當訓練腳本報錯時，Agent 不會中斷，而是閱讀錯誤日誌並嘗試修復語法或邏輯錯誤。防退化機制 (Rollback)具備版本控制概念，確保最終產出的程式碼是歷次迭代中表現最好的版本。防資料洩漏 (Anti-Leakage)系統被植入嚴格指令，禁止在預處理階段使用驗證集的統計數據進行 Scaling。結構化日誌透過 ExperimentLog 追蹤每一次迭代的策略 (Strategy)、組件 (Component) 與結果 (MAPE)。

### 4. 環境需求與安裝 (Prerequisites):
- 系統要求Python 3.9+
- 建議具備 NVIDIA GPU (以加速 XGBoost/LightGBM 訓練)安裝
```bash
pip install langchain-ollama langchain-community langgraph tavily-python arxiv dotenv
```

#### 需確保已安裝機器學習相關庫 (Agent 生成的代碼會用到)
```bash 
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```
API 配置請在專案根目錄建立 .env 檔案，填入以下資訊：程式碼片段# Ollama Cloud 或本地服務設定
```.env
OLLAMA_API_KEY=your_ollama_key_here
TAVILY_API_KEY=your_tavily_key_here
```

#### 資料準備
請確保以下兩個檔案位於專案根目錄：train.csv (Rossmann 訓練資料)store.csv (店鋪資訊)

### 5. 使用說明 (Usage):
直接執行主程式即可啟動自動化流程：Bashpython mle_star_agent.py

執行過程監控系統會在 Console 輸出詳細的步驟：
- [Step 1] Research Agent: 顯示搜尋到的論文與 Kaggle 方案。
- [Step 2] MLE Coder: 生成基礎程式碼。
- [Step 3] Optimization Agent: 顯示每次迭代的 MAPE 分數與優化策略 (例如："New Best Found!")。
- [Step 4] Analyst Agent: 撰寫報告。

>>輸出產物執行完成後，將生成以下檔案：final_best_model.py: 經過多次優化後，MAPE 分數最低的完整 Python 訓練腳本。analysis_report.md: 包含實驗歷程與洞察的完整分析報告。train_iter_X.py: (過程檔案) 各次迭代的臨時腳本。

### 6. 自定義與擴展 (Configuration)您可以在程式碼開頭調整以下參數：
- MAX_ITERATIONS: 預設為 10。增加此數值可讓 Agent 進行更深度的特徵工程探索。
- llm_reasoner: 目前設定為 deepseek-v3.1，可更換為其他具備強大推理能力的模型。
- llm_coder: 目前設定為 qwen3-coder，專職負責寫 Code。

### 7. 限制與未來展望執行時間：
由於包含真實的模型訓練過程，若數據集較大，建議增加 subprocess.run 的 timeout 時間。模型依賴：系統極度依賴 LLM 的 coding 能力，若模型生成錯誤的 Pandas 語法且無法自我修復，可能會導致迭代提早終止。