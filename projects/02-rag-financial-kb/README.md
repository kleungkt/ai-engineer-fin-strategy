# 📚 RAG 金融知識庫問答系統

## 目標

建立一個基於 **RAG（Retrieval-Augmented Generation）** 技術的金融知識庫問答系統，能夠：

1. **文檔解析與索引**：自動解析金融法規、報告、研究論文等文檔，建立結構化知識庫
2. **語義搜索**：基於向量嵌入進行語義級別的文檔檢索，而非僅關鍵字匹配
3. **上下文增強生成**：將檢索到的相關文檔片段注入 LLM prompt，生成準確且有據可查的回答
4. **來源追溯**：每個回答都能標註引用來源，確保資訊可驗證
5. **多模態支持**：處理表格、圖表、PDF 等多種金融文檔格式

## 示例

### 基本問答
```
用戶: 什麼是 Basel III 的資本充足率要求？

系統: 根據 Basel III 框架，銀行需滿足以下資本充足率要求：
1. 核心一級資本（CET1）比率 ≥ 4.5%
2. 一級資本比率 ≥ 6%
3. 總資本比率 ≥ 8%
此外，還需計提 2.5% 的資本保存緩衝...

📄 來源: Basel III Framework, Section 3.1 (p.12)
📄 來源: 金管會銀行資本管理辦法第15條
```

### 多文檔交叉查詢
```
用戶: 比較台灣和美國對加密貨幣交易所的監管要求

系統: 台灣與美國在加密貨幣交易所監管方面的主要差異：

【台灣】
- 依《洗錢防制法》要求交易所進行洗錢防制聲明
- 金管會於2023年成為主管機關...

【美國】
- 需取得 Money Transmitter License（各州要求不同）
- SEC 和 CFTC 共同監管...

📄 來源: 台灣金管會加密貨幣監管框架 (2024)
📄 來源: SEC v. Ripple Labs, Inc. (2023)
```

## 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **文檔處理** | LangChain, Unstructured, PyMuPDF | 文檔載入、解析、分塊 |
| **向量嵌入** | OpenAI Embedding / BGE-M3 / Jina | 文本向量化 |
| **向量數據庫** | ChromaDB / Milvus / Weaviate | 向量存儲與檢索 |
| **LLM** | GPT-4 / Claude / Llama 3 | 答案生成 |
| **後端** | FastAPI, Python | API 服務 |
| **前端** | Streamlit / Gradio | 互動界面 |
| **評估** | RAGAS, DeepEval | RAG 質量評估 |

## 架構

```
┌─────────────────────────────────────────────────────────────┐
│                        用戶界面層                            │
│              (Streamlit / Gradio Web UI)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      API 服務層                              │
│                   (FastAPI Backend)                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐      │
│  │ 查詢路由  │  │ 對話管理器    │  │  來源追溯器       │      │
│  └────┬─────┘  └──────┬───────┘  └────────┬──────────┘      │
│       │               │                   │                 │
│       ▼               ▼                   ▼                 │
│  ┌──────────────────────────────────────────────┐           │
│  │              RAG 核心引擎                     │           │
│  │  ┌─────────┐ ┌──────────┐ ┌───────────────┐  │           │
│  │  │查詢改寫 │ │上下文壓縮│ │答案生成(帶引用)│  │           │
│  │  └────┬────┘ └────┬─────┘ └───────┬───────┘  │           │
│  └───────┼──────────┼───────────────┼───────────┘           │
│          │          │               │                       │
└──────────┼──────────┼───────────────┼───────────────────────┘
           │          │               │
           ▼          ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                      數據層                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  向量數據庫   │  │  文檔存儲     │  │  元數據索引   │       │
│  │  (ChromaDB)  │  │  (本地/S3)   │  │  (SQLite)    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
           ▲
           │
┌──────────┴──────────────────────────────────────────────────┐
│                   文檔處理管線                                │
│  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐   │
│  │文檔載入 │→│ 文本分塊  │→│ 向量嵌入 │→│ 索引存儲     │   │
│  │(PDF等) │  │(Chunking)│  │(Embedding)│ │(Upsert)     │   │
│  └────────┘  └──────────┘  └─────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
02-rag-financial-kb/
├── README.md                    # 本文件
├── requirements.txt             # Python 依賴
├── .env.example                 # 環境變數範例
├── config/
│   ├── settings.py              # 配置管理
│   └── prompts.py               # Prompt 模板
├── data/
│   ├── raw/                     # 原始文檔
│   │   ├── regulations/         # 法規文檔
│   │   ├── reports/             # 研究報告
│   │   └── papers/              # 學術論文
│   ├── processed/               # 處理後的文檔
│   └── sample_queries.json      # 範例查詢（用於測試）
├── src/
│   ├── __init__.py
│   ├── document_loader.py       # 文檔載入器
│   │   ├── PDFLoader            #   PDF 解析
│   │   ├── WebLoader            #   網頁爬取
│   │   └── ExcelLoader          #   Excel 表格
│   ├── text_splitter.py         # 文本分塊策略
│   │   ├── RecursiveSplitter    #   遞迴字元分割
│   │   ├── SemanticSplitter     #   語義分割
│   │   └── FinancialSplitter    #   金融專用分割（按段落/條款）
│   ├── embedding.py             # 嵌入模型封裝
│   ├── vector_store.py          # 向量數據庫操作
│   ├── retriever.py             # 檢索策略
│   │   ├── SimilarityRetriever  #   相似度檢索
│   │   ├── MMRRetriever         #   最大邊際相關性
│   │   └── HybridRetriever      #   混合檢索（向量+關鍵字）
│   ├── reranker.py              # 重排序模型
│   ├── rag_engine.py            # RAG 核心引擎
│   ├── source_tracker.py        # 來源追溯
│   └── evaluator.py             # RAG 評估模組
├── pipeline/
│   ├── ingest.py                # 文檔灌入管線
│   └── rebuild_index.py         # 重建索引
├── api/
│   ├── main.py                  # FastAPI 應用
│   ├── routes/
│   │   ├── query.py             # 查詢端點
│   │   └── admin.py             # 管理端點
│   └── models.py                # Pydantic 模型
├── app/
│   └── streamlit_app.py         # Streamlit 前端
├── notebooks/
│   ├── 01_document_processing.ipynb    # 文檔處理實驗
│   ├── 02_chunking_strategies.ipynb    # 分塊策略比較
│   ├── 03_retrieval_evaluation.ipynb   # 檢索評估
│   └── 04_rag_evaluation.ipynb        # RAG 整體評估
├── tests/
│   ├── test_loader.py
│   ├── test_retriever.py
│   └── test_rag_engine.py
└── scripts/
    ├── download_sample_data.py  # 下載範例數據
    └── batch_ingest.py          # 批量灌入腳本
```

## 學習重點

### 1. 文檔處理與分塊策略
- **為什麼分塊很重要**：塊太大 → 語義模糊；塊太小 → 丟失上下文
- **常見策略**：固定大小、遞迴分割、語義分割、按文檔結構分割
- **金融文檔特殊性**：條款編號、表格、跨頁引用的處理

### 2. 嵌入模型選擇
- **通用模型** vs **領域專用模型**的權衡
- **多語言支持**：中文金融文本的嵌入質量
- **評估方法**：用 MTEB benchmark 和領域特定測試集

### 3. 檢索策略
- **相似度搜索**：基本的 cosine similarity
- **MMR（最大邊際相關性）**：平衡相關性與多樣性
- **混合搜索**：結合向量搜索和 BM25 關鍵字搜索
- **重排序（Reranking）**：用 cross-encoder 提升檢索精度

### 4. Prompt Engineering for RAG
- **系統提示設計**：定義角色和回答風格
- **上下文注入策略**：如何組織檢索結果
- **引用指令**：確保 LLM 標註來源
- **幻覺防護**：處理「文檔中沒有答案」的情況

### 5. RAG 評估指標
- **檢索指標**：Precision@K, Recall@K, MRR, NDCG
- **生成指標**：Faithfulness, Answer Relevancy, Context Relevancy
- **端到端指標**：RAGAS 分數, 人工評估

## 開發步驟

### Step 1: 環境搭建與數據準備（Day 1-2）
```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入 API keys

# 下載範例金融文檔
python scripts/download_sample_data.py
```

### Step 2: 文檔處理管線（Day 3-4）
```python
# 實現文檔載入器
# src/document_loader.py
# 支持 PDF, Word, Excel, HTML 等格式

# 實現文本分塊
# src/text_splitter.py
# 嘗試不同分塊策略並比較效果

# 測試：處理一批金融文檔
python pipeline/ingest.py --input data/raw/ --output data/processed/
```

### Step 3: 向量嵌入與存儲（Day 5-6）
```python
# 實現嵌入模型封裝
# src/embedding.py - 支持 OpenAI / BGE-M3 / Jina

# 實現向量數據庫操作
# src/vector_store.py - ChromaDB 操作封裝

# 建立索引
python pipeline/rebuild_index.py
```

### Step 4: 檢索與重排序（Day 7-8）
```python
# 實現多種檢索策略
# src/retriever.py

# 加入重排序模型
# src/reranker.py

# 評估檢索效果
# notebooks/03_retrieval_evaluation.ipynb
```

### Step 5: RAG 引擎整合（Day 9-11）
```python
# 實現核心 RAG 邏輯
# src/rag_engine.py
# - 查詢改寫
# - 檢索 + 重排序
# - 上下文壓縮
# - 答案生成（帶引用）

# 實現來源追溯
# src/source_tracker.py
```

### Step 6: API 與前端（Day 12-13）
```python
# 建立 FastAPI 服務
# api/main.py

# 建立 Streamlit 前端
# app/streamlit_app.py

# 測試端到端流程
```

### Step 7: 評估與優化（Day 14-15）
```python
# 建立評估測試集
# data/sample_queries.json

# 運行 RAGAS 評估
# src/evaluator.py

# 優化策略：
# - 調整 chunk size
# - 嘗試不同 embedding model
# - 優化 prompt template
# - 調整 top_k 和 reranking
```

### Step 8: 進階功能（Day 16-18）
```
- [ ] 支持多輪對話（對話歷史管理）
- [ ] 實現 HyDE（假設文檔嵌入）提升檢索
- [ ] 加入知識圖譜輔助檢索
- [ ] 實現自動文檔更新管線
- [ ] 添加用戶反饋機制（RLHF 數據收集）
```

---

> 💡 **小提示**：先從簡單的 Naive RAG 開始，確保基本流程通順後，再逐步加入進階功能（查詢改寫、重排序、上下文壓縮等）。
