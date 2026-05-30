# 🗺️ 學習路線圖：AI 工程師（金融策略方向）

## 階段一：LLM 基礎（Week 1-2）

### Week 1: Prompt Engineering
- [ ] 學習 Few-shot prompting
- [ ] 學習 Chain-of-Thought (CoT)
- [ ] 學習 ReAct (Reasoning + Acting)
- [ ] 練習結構化輸出（JSON mode, function calling）
- [ ] **實戰**：寫一個 prompt，讓 LLM 從自然語言提取股票代碼和指標

### Week 2: LangChain 基礎
- [ ] LangChain 安裝與核心概念（Chain, Prompt, Output Parser）
- [ ] Tool Use：讓 LLM 調用 Python 函數
- [ ] Memory：對話上下文管理
- [ ] **實戰**：用 LangChain 構建一個簡單的 QA chain

---

## 階段二：RAG 與向量檢索（Week 3-4）

### Week 3: Embedding 與向量數據庫
- [ ] 理解 text embedding 概念
- [ ] Chroma 向量數據庫入門
- [ ] Chunking 策略（固定長度 vs 語義分割）
- [ ] **實戰**：把一份金融文檔嵌入 Chroma，測試相似度檢索

### Week 4: RAG Pipeline
- [ ] 構建完整 RAG pipeline（檢索 → 增強 → 生成）
- [ ] Reranking 策略
- [ ] 評估 RAG 質量（faithfulness, relevance）
- [ ] **實戰**：[Project 2: RAG 金融知識庫問答](projects/02-rag-financial-kb/)

---

## 階段三：量化金融基礎（Week 3-6，與 RAG 並行）

### 技術指標（Week 3-4）
- [ ] 均線（MA, EMA）計算與含義
- [ ] MACD（快線、慢線、柱狀圖）
- [ ] RSI（超買超賣判斷）
- [ ] 布林帶（Bollinger Bands）
- [ ] **實戰**：用 Pandas 計算以上指標，可視化

### 回測概念（Week 5-6）
- [ ] 理解回報率（Return）計算
- [ ] 理解夏普比率（Sharpe Ratio）
- [ ] 理解最大回撤（Max Drawdown）
- [ ] 理解勝率（Win Rate）
- [ ] Backtrader 框架入門
- [ ] **實戰**：用 Backtrader 回測一個簡單的均線交叉策略

---

## 階段四：Agent 與系統集成（Week 7-10）

### Week 7-8: Agent 架構
- [ ] ReAct Agent 模式
- [ ] 自定義 Tool 開發
- [ ] Multi-agent 協作
- [ ] **實戰**：[Project 1: 自然語言股票查詢器](projects/01-nl-stock-query/)

### Week 9-10: 核心項目 ⭐
- [ ] **實戰**：[Project 3: AI 策略生成器](projects/03-ai-strategy-generator/)
  - LLM 解析自然語言策略
  - 轉化為 Backtrader 代碼
  - 執行回測
  - 生成診斷報告

---

## 階段五：進階能力（Week 11-14）

### Week 11-12: 策略診斷
- [ ] **實戰**：[Project 4: 策略診斷助手](projects/04-strategy-diagnostics/)
- [ ] Prompt 模板設計
- [ ] 金融專業表達

### Week 13-14: 模型微調（加分項）
- [ ] **實戰**：[Project 5: FinLLM Fine-tune](projects/05-finllm-finetune/)
- [ ] 數據準備
- [ ] LoRA 微調
- [ ] 評估

---

## 📚 每日學習節奏

```
30 分鐘：技術指標學習（每天一個指標）
60 分鐘：項目開發
30 分鐘：閱讀 LLM/量化文檔
```

---

## ✅ 里程碑

| 里程碑 | 時間 | 產出 |
|-------|------|------|
| M1: LLM 基礎 | Week 2 | 能用 LangChain 構建 QA chain |
| M2: RAG 能力 | Week 4 | 能構建 RAG 金融知識庫 |
| M3: 量化基礎 | Week 6 | 能用 Backtrader 回測策略 |
| M4: 項目 1-3 完成 | Week 10 | 3 個可展示的 GitHub 項目 |
| M5: 全部完成 | Week 14 | 5 個項目 + 完整學習路徑 |
