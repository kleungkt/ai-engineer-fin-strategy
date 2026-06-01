# 🚀 AI 工程師實戰項目集：LLM × 量化金融策略生成平台

> 從自然語言到可執行策略的全鏈路 AI 系統，對齊「AI 應用開發工程師（金融交易策略方向）」核心能力

[![Tests](https://img.shields.io/badge/tests-594%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 📋 項目總覽

本倉庫包含 **5 個相互關聯的實戰項目**，從基礎 LLM 應用到完整的策略生成平台，涵蓋了 AI 工程師在金融科技領域所需的全部核心技能。

| # | 項目 | 狀態 | Tests | 代碼量 | 核心技能 |
|---|------|------|-------|--------|---------|
| 01 | [NL Stock Query](projects/01-nl-stock-query/) | ✅ | 223 | 2,984 行 | Prompt Engineering, Function Calling, 技術指標, 回測 |
| 02 | [RAG Financial KB](projects/02-rag-financial-kb/) | ✅ | 110 | 5,331 行 | RAG, 向量數據庫, 分塊策略, 重排序, 評估 |
| 03 | [AI Strategy Generator](projects/03-ai-strategy-generator/) | ✅ | 128 | 9,221 行 | Agent 架構, 代碼生成, AST 驗證, 參數優化 |
| 04 | [Strategy Diagnostics](projects/04-strategy-diagnostics/) | ✅ | 68 | 3,304 行 | 評分體系, AI 分析報告, 多格式輸出 |
| 05 | [FinLLM Fine-tune](projects/05-finllm-finetune/) | ✅ | 65 | 3,304 行 | 數據管線, LoRA 微調, 量化部署, 評估 |

**合計**: 113 個 Python 文件, 25,000+ 行代碼, **594 個測試全部通過**

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        用戶交互層                                │
│            自然語言輸入 / Streamlit / REST API                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI 策略生成引擎 (P3)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ NLU 解析器   │  │ 代碼生成器   │  │ 代碼驗證器 (AST)     │  │
│  │ (P1 parser)  │  │ (LLM+模板)  │  │ (安全檢查)           │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         ▼                 ▼                      ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               策略執行沙箱 (Backtrader)                    │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  回測引擎 (P1)   │ │ RAG 知識庫   │ │ 策略診斷 (P4)    │
│  數據源 (AKShare)│ │ (P2 ChromaDB)│ │ 評分 + AI 分析   │
└──────────────────┘ └──────────────┘ └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  LLM 微調 (P5)   │
                    │  LoRA / 量化部署  │
                    └──────────────────┘
```

---

## 🎯 JD 能力對齊

本項目集完整對齊「AI 應用開發工程師（金融交易策略方向）」的每一項職責：

### 崗位職責覆蓋

| JD 要求 | 本倉庫對應 | 實現位置 |
|---------|-----------|---------|
| **語義理解與資產意圖識別** | LLM Function Calling + NER | `P1/parser.py`, `P3/strategy_agent.py` |
| **智能指標配置與策略映射** | 6 種技術指標 + 6 種策略模板 | `P1/indicators.py`, `P3/strategy_templates.py` |
| **回測數據工程與模板生成** | AKShare 數據源 + Backtrader 引擎 | `P1/backtester.py`, `P3/backtester.py` |
| **智能診斷與策略建議** | 評分體系 + GPT-4o 專業分析 | `P4/evaluator.py`, `P4/ai_analyst.py` |

### 必備條件覆蓋

| 要求 | 覆蓋 |
|------|------|
| Python 精通 | 全部 113 個 .py 文件 |
| LangChain / LlamaIndex / AutoGen | P2 RAG 系統使用 LangChain 風格架構 |
| Prompt Engineering | P1/P3 的 LLM function calling + 系統提示設計 |
| RAG 實際落地 | P2 完整 RAG pipeline（嵌入 → 檢索 → 重排序 → 生成） |
| Agent 智能體 | P3 策略生成 Agent（解析 → 生成 → 驗證 → 執行） |
| 金融量化知識 | MA/MACD/RSI/Bollinger/KDJ 計算 + 回測指標 |
| Backtrader / Pandas | 全面使用，4 種回測策略 |

### 加分項覆蓋

| 要求 | 覆蓋 |
|------|------|
| 金融垂直領域 LLM | P5 金融 LLM 微調管線（LoRA/QLoRA） |
| 向量數據庫 | P2 ChromaDB + 混合檢索 + 重排序 |

---

## 📁 項目結構

```
ai-engineer-fin-strategy/
│
├── projects/
│   ├── 01-nl-stock-query/              # 📊 自然語言股票查詢器
│   │   ├── src/
│   │   │   ├── parser.py               #   LLM 意圖解析 (OpenAI function calling)
│   │   │   ├── indicators.py           #   6 個技術指標 (MA/EMA/MACD/RSI/Bollinger/KDJ)
│   │   │   ├── screener.py             #   股票篩選引擎 (AND 邏輯)
│   │   │   ├── backtester.py           #   4 種策略回測 (MA/RSI/MACD/Bollinger)
│   │   │   ├── diagnostics.py          #   策略診斷評分 + AI 分析
│   │   │   ├── patterns.py             #   7 種 K 線形態識別
│   │   │   ├── data_fetcher.py         #   AKShare 真實 A 股數據
│   │   │   ├── api.py                  #   FastAPI REST API (4 endpoints)
│   │   │   └── main.py                 #   CLI 互動入口
│   │   └── tests/                      #   223 tests ✅
│   │
│   ├── 02-rag-financial-kb/            # 📚 RAG 金融知識庫問答
│   │   ├── src/
│   │   │   ├── document_loader.py      #   PDF/Word/Excel/HTML 文檔載入
│   │   │   ├── text_splitter.py        #   3 種分塊策略 (遞迴/語義/金融專用)
│   │   │   ├── embedding.py            #   OpenAI/BGE-M3 嵌入封裝
│   │   │   ├── vector_store.py         #   ChromaDB 向量庫操作
│   │   │   ├── retriever.py            #   相似度/MMR/混合檢索
│   │   │   ├── reranker.py             #   Cross-encoder 重排序
│   │   │   ├── rag_engine.py           #   RAG 核心引擎
│   │   │   ├── source_tracker.py       #   引用來源追溯
│   │   │   ├── evaluator.py            #   RAGAS 評估指標
│   │   │   └── api.py                  #   FastAPI REST API
│   │   └── tests/                      #   110 tests ✅
│   │
│   ├── 03-ai-strategy-generator/       # 🤖 AI 策略生成器 ⭐ 核心項目
│   │   ├── src/
│   │   │   ├── strategy_agent.py       #   NL→策略→代碼→回測 全鏈路 Agent
│   │   │   ├── strategy_templates.py   #   6 種預建策略模板
│   │   │   ├── code_validator.py       #   AST 代碼驗證 + 安全檢查
│   │   │   ├── backtester.py           #   回測執行引擎
│   │   │   ├── optimizer/              #   參數優化 (網格/隨機/Walk-Forward/過擬合檢測)
│   │   │   ├── nlu/                    #   意圖分類/實體提取/邏輯解析
│   │   │   ├── generator/              #   代碼生成器/模板引擎/安全檢查
│   │   │   ├── backtest/               #   回測引擎/指標計算/報告生成
│   │   │   ├── data_fetcher.py         #   AKShare 數據 + 樣本生成
│   │   │   ├── api.py                  #   FastAPI REST API
│   │   │   └── main.py                 #   CLI 互動入口
│   │   └── tests/                      #   128 tests ✅
│   │
│   ├── 04-strategy-diagnostics/        # 🏥 策略診斷系統
│   │   ├── src/
│   │   │   ├── models.py               #   BacktestResult/DiagnosticReport 模型
│   │   │   ├── evaluator.py            #   多維度評分 (Sharpe/Return/Drawdown/WinRate)
│   │   │   ├── ai_analyst.py           #   GPT-4o 專業量化分析報告
│   │   │   ├── formatter.py            #   text/markdown/JSON 三種格式
│   │   │   ├── api.py                  #   FastAPI REST API
│   │   │   └── main.py                 #   CLI 入口
│   │   └── tests/                      #   68 tests ✅
│   │
│   └── 05-finllm-finetune/             # 🧠 金融 LLM 微調
│       ├── src/
│       │   ├── data_pipeline.py        #   金融數據管線 + 合成數據生成
│       │   ├── trainer.py              #   LoRA/QLoRA 微調配置
│       │   ├── evaluator.py            #   情感/QA/生成評估 (BLEU/ROUGE)
│       │   ├── quantize.py             #   GGUF/AWQ 量化部署
│       │   ├── api.py                  #   FastAPI 訓練/評估/生成 API
│       │   └── main.py                 #   CLI 入口
│       └── tests/                      #   65 tests ✅
│
├── shared/                             # 共用模組
│   ├── indicators/                     #   技術指標計算庫
│   ├── data_loader/                    #   數據加載器
│   └── llm_utils/                      #   LLM prompt 模板
│
├── ROADMAP.md                          # 14 週學習路線圖
└── requirements.txt                    # Python 依賴
```

---

## 🚀 快速開始

```bash
# 克隆倉庫
git clone https://github.com/kleungkt/ai-engineer-fin-strategy.git
cd ai-engineer-fin-strategy

# 選擇一個項目開始
cd projects/01-nl-stock-query

# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 跑測試
PYTHONPATH=src pytest tests/ -v

# 啟動 API 服務
PYTHONPATH=src python src/api.py
# → 訪問 http://localhost:8000/docs 查看 API 文檔
```

---

## 🔧 技術棧

| 類別 | 技術 |
|------|------|
| **LLM** | OpenAI GPT-4o/4o-mini, Function Calling, Prompt Engineering |
| **RAG** | ChromaDB, OpenAI Embeddings, BGE-M3, Cross-encoder Reranking |
| **量化** | Backtrader, Pandas, NumPy, AKShare (A 股數據) |
| **NLP** | Pydantic 結構化輸出, AST 驗證, NER/意圖識別 |
| **微調** | HuggingFace Transformers, PEFT (LoRA/QLoRA), BitsAndBytes |
| **後端** | FastAPI, Uvicorn, Pydantic v2 |
| **評估** | RAGAS, BLEU, ROUGE, Sharpe Ratio, Max Drawdown |
| **測試** | Pytest, unittest.mock, 594 tests |

---

## 📊 項目詳解

### P1: 自然語言股票查詢器

將自然語言轉化為股票篩選條件，支持技術指標計算、策略回測、K 線形態識別。

```python
# 輸入
"幫我查最近 RSI 低於 30 且 MACD 金叉的 A 股"

# 系統解析
QueryIntent(
    intent_type="indicator_screener",
    indicators=[
        IndicatorCondition(name="RSI", comparison="below", value=30),
        IndicatorCondition(name="MACD", comparison="crossover")
    ],
    stock_scope="A股",
    time_range=30
)

# 輸出: 符合條件的股票列表 + 回測結果 + 策略診斷
```

**亮點**: 6 個技術指標、7 種 K 線形態、4 種回測策略、FastAPI API

### P2: RAG 金融知識庫問答

基於 RAG 技術的金融知識問答系統，支持語義搜索、來源追溯、評估指標。

```
用戶: 什麼是 Basel III 的資本充足率要求？

系統: 根據 Basel III 框架，銀行需滿足以下資本充足率要求：
1. 核心一級資本（CET1）比率 ≥ 4.5%
2. 一級資本比率 ≥ 6%
3. 總資本比率 ≥ 8%

📄 來源: Basel III Framework, Section 3.1 (p.12)
```

**亮點**: 3 種分塊策略、混合檢索（向量+BM25）、Cross-encoder 重排序、RAGAS 評估

### P3: AI 策略生成器 ⭐

核心項目 — 完整的 NL→策略代碼→回測→優化 pipeline。

```python
# 輸入
"幫我寫一個均值回歸策略，用布林通道和 RSI 結合，
 在下軌且 RSI 超賣時買入，到上軌且 RSI 超買時賣出"

# 系統自動生成 Backtrader 策略代碼
class MeanReversionBollingerRSI(bt.Strategy):
    params = {'bb_period': 20, 'bb_std': 2, 'rsi_period': 14, ...}
    
    def next(self):
        # 自動生成的交易邏輯
        ...

# 自動回測 → 返回績效報告 → 參數優化建議
```

**亮點**: LLM 代碼生成、AST 驗證、安全沙箱執行、網格搜索優化、Walk-Forward 驗證

### P4: 策略診斷系統

多維度策略評分 + AI 專業分析報告。

```
📊 策略診斷報告

整體評分: 72/100

✅ 夏普比率: 1.8 (good)
⚠️ 年化收益: 12% (acceptable)  
🔴 最大回撤: 25% (poor)
✅ 勝率: 55% (good)

建議:
1. 最大回撤過高，建議加入動態止損
2. 考慮使用 ATR 自適應止損距離
3. 建議在震盪市減少倉位
```

**亮點**: 加權評分體系、GPT-4o 專業分析、多策略對比、text/markdown/JSON 輸出

### P5: 金融 LLM 微調

完整的微調管線：數據準備 → LoRA 訓練 → 評估 → 量化部署。

**亮點**: 合成數據生成、LoRA/QLoRA 配置、BLEU/ROUGE 評估、GGUF/AWQ 量化

---

## 🧪 測試

```bash
# 跑所有項目的測試
for project in projects/*/; do
    echo "=== Testing $project ==="
    cd "$project"
    PYTHONPATH=src .venv/bin/pytest tests/ -q
    cd ../..
done

# 預期輸出:
# 01-nl-stock-query:     223 passed
# 02-rag-financial-kb:   110 passed
# 03-ai-strategy-generator: 128 passed
# 04-strategy-diagnostics:  68 passed
# 05-finllm-finetune:       65 passed
# Total: 594 passed
```

---

## 📝 簡歷亮點

> - 構建了基於 LLM 的自然語言策略轉換系統，支持用戶用中文描述交易邏輯，自動生成 Backtrader 回測代碼並輸出診斷報告
> - 設計 RAG 金融知識庫，支持語義檢索、混合搜索、Cross-encoder 重排序，問答準確率可量化評估
> - 實現完整 AI Agent 策略生成 pipeline：NLU → 代碼生成 → AST 驗證 → 安全執行 → 參數優化（網格搜索 + Walk-Forward）
> - 開發策略診斷系統，支持 Sharpe/回撤/勝率等多維度評分 + GPT-4o 專業量化分析報告
> - 構建金融 LLM 微調管線：數據準備 → LoRA 訓練 → GGUF/AWQ 量化部署 → BLEU/ROUGE 評估

---

## 📜 License

MIT
