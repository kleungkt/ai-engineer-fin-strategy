# 🚀 AI 工程師養成：LLM × 量化金融策略生成平台

> 從零到一，系統性學習 AI 應用開發（金融交易策略方向）

## 🎯 目標崗位對齊

本倉庫對齊「AI 應用開發工程師（金融交易策略方向）」的核心能力：

| 能力維度 | 本倉庫對應項目 |
|---------|--------------|
| 語義理解與資產意圖識別 | [Project 1: NL Stock Query](projects/01-nl-stock-query/) |
| RAG 金融知識庫 | [Project 2: RAG Financial KB](projects/02-rag-financial-kb/) |
| 智能指標配置與策略映射 | [Project 3: AI Strategy Generator](projects/03-ai-strategy-generator/) |
| 回測數據工程 | [Project 3 + Project 4](projects/03-ai-strategy-generator/) |
| 智能診斷與策略建議 | [Project 4: Strategy Diagnostics](projects/04-strategy-diagnostics/) |
| 模型微調（加分） | [Project 5: FinLLM Fine-tune](projects/05-finllm-finetune/) |

---

## 🛤️ 學習路徑

### 第一層：LLM 應用開發（核心）

```
Prompt Engineering → LangChain/LlamaIndex → RAG → Agent → Fine-tuning
```

- [ ] Prompt Engineering 基礎（Few-shot, CoT, ReAct, 結構化輸出）
- [ ] LangChain Chain 構建與 Tool Use
- [ ] RAG pipeline（Embedding → Vector Store → Retrieval → Generation）
- [ ] Agent 架構（ReAct loop, Multi-agent）
- [ ] Fine-tuning（LoRA/QLoRA, SFT）

### 第二層：量化金融基礎

```
技術指標 → K線形態 → 回測概念 → 回測框架 → 數據處理
```

- [ ] 技術指標計算（MA, MACD, RSI, Bollinger Bands）
- [ ] K 線形態識別
- [ ] 回測核心概念（Return, Sharpe, Max Drawdown, Win Rate）
- [ ] Backtrader / vectorbt 框架
- [ ] Pandas 金融數據操作

### 第三層：工程能力

```
向量數據庫 → FastAPI → 數據管道 → 系統集成
```

- [ ] 向量數據庫（Chroma 入門 → Milvus 生產）
- [ ] FastAPI 構建 API 服務
- [ ] 歷史行情數據清洗與標準化

---

## 📁 項目結構

```
ai-engineer-fin-strategy/
├── README.md                          # 本文件
├── ROADMAP.md                         # 詳細學習路線圖
├── projects/
│   ├── 01-nl-stock-query/             # 自然語言股票查詢器
│   │   ├── README.md
│   │   ├── src/
│   │   ├── tests/
│   │   └── examples/
│   ├── 02-rag-financial-kb/           # RAG 金融知識庫問答
│   │   ├── README.md
│   │   ├── src/
│   │   ├── docs/                      # 金融知識文檔
│   │   └── tests/
│   ├── 03-ai-strategy-generator/      # AI 策略生成器 ⭐ 核心項目
│   │   ├── README.md
│   │   ├── src/
│   │   ├── strategies/
│   │   ├── backtest/
│   │   └── tests/
│   ├── 04-strategy-diagnostics/       # 智能策略診斷助手
│   │   ├── README.md
│   │   ├── src/
│   │   └── templates/
│   └── 05-finllm-finetune/           # 金融微調模型（加分項）
│       ├── README.md
│       ├── data/
│       └── scripts/
├── shared/                            # 共用模組
│   ├── indicators/                    # 技術指標計算庫
│   ├── data_loader/                   # 數據加載器
│   └── llm_utils/                     # LLM 工具函數
└── docs/
    ├── learning-notes/                # 學習筆記
    └── references/                    # 參考資料
```

---

## 🚀 快速開始

```bash
# 克隆倉庫
git clone https://github.com/kleungkt/ai-engineer-fin-strategy.git
cd ai-engineer-fin-strategy

# 安裝依賴
pip install -r requirements.txt

# 開始第一個項目
cd projects/01-nl-stock-query
```

---

## 📊 進度追蹤

| 項目 | 狀態 | 核心技能 |
|-----|------|---------|
| 01 NL Stock Query | ✅ 完成 (223 tests) | Prompt Engineering, 結構化輸出, 回測, K線形態 |
| 02 RAG Financial KB | ✅ 完成 (110 tests) | RAG, 向量數據庫, 分塊策略, 重排序 |
| 03 AI Strategy Generator | ✅ 完成 (128 tests) | Agent, Tool Use, 代碼生成, 參數優化 |
| 04 Strategy Diagnostics | 🔲 未開始 | Prompt 進階, 金融專業表達 |
| 05 FinLLM Fine-tune | 🔲 未開始 | LoRA 微調, 數據準備 |

---

## 📝 學習資源

### LLM 開發
- [LangChain 文檔](https://python.langchain.com/)
- [LlamaIndex 文檔](https://docs.llamaindex.ai/)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/prompt-engineering)

### 量化金融
- [Backtrader 文檔](https://www.backtrader.com/docu/)
- [TA-Lib 技術指標](https://github.com/TA-Lib/ta-lib-python)
- [Investopedia 量化概念](https://www.investopedia.com/terms/q/quantitative-analysis.asp)

### 向量數據庫
- [Chroma 文檔](https://docs.trychroma.com/)
- [Milvus 文檔](https://milvus.io/docs)

---

## 📜 License

MIT
