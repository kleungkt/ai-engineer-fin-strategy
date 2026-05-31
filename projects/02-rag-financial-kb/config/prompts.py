"""
Prompt templates for the RAG financial knowledge base system.

This module defines all prompt templates used in the RAG pipeline,
including system prompts, user templates, and query rewrite prompts.
All prompts are written in Chinese for optimal Chinese financial domain performance.
"""

# System prompt for RAG question answering
RAG_SYSTEM_PROMPT = """你是一位專業的金融分析師和知識庫助手。你的職責是根據提供的參考資料，準確回答用戶的金融投資相關問題。

請遵守以下原則：
1. 基於提供的參考資料進行回答，確保回答有據可依
2. 回答中必須引用來源，使用 [1]、[2] 等編號標記引用的資料
3. 如果參考資料不足以回答問題，請明確告知用戶「根據現有資料無法回答此問題」，不要編造資訊
4. 使用專業但易懂的語言，必要時提供解釋
5. 如果問題涉及投資建議，請提醒用戶這僅供參考，不構成投資建議
6. 回答應結構清晰，使用適當的標題和分點說明"""

# User prompt template for RAG question answering
RAG_USER_TEMPLATE = """以下是與問題相關的參考資料：

{context}

---

請根據以上參考資料回答以下問題。如果資料不足，請明確說明。

問題：{query}"""

# Query rewrite prompt for better retrieval
QUERY_REWRITE_PROMPT = """你是一個搜索查詢優化專家。請將以下用戶問題改寫為更適合知識庫檢索的查詢語句。

要求：
1. 提取問題中的關鍵金融術語和概念
2. 擴展同義詞和相關詞彙以提高檢索覆蓋率
3. 生成 2-3 個不同角度的檢索查詢
4. 保持查詢簡潔，每個查詢不超過 30 個字
5. 輸出格式為 JSON 陣列

原始問題：{query}

請輸出優化後的查詢（JSON 格式）："""

# Answer with sources template
ANSWER_WITH_SOURCES_TEMPLATE = """你是金融知識庫的專業回答助手。請根據以下參考資料回答問題，並嚴格按照格式要求標註來源。

## 回答格式要求：
1. 先給出直接回答
2. 在回答中使用 [1]、[2] 等編號引用參考資料
3. 在回答末尾列出所有引用的來源

## 參考資料：
{context}

## 用戶問題：
{query}

## 回答：
（請在此處回答問題，並使用 [1]、[2] 等標記引用來源）

## 引用來源：
（請在此處列出所有引用的來源編號和內容摘要）"""

# Document summarization prompt
DOCUMENT_SUMMARY_PROMPT = """請為以下金融文檔生成簡潔的摘要，提取關鍵信息：

文檔內容：
{document}

要求：
1. 摘要不超過 200 字
2. 提取核心觀點、關鍵數據和重要結論
3. 保留專業術語的準確性
4. 使用結構化格式（如要點列表）"""

# Financial term definition prompt
TERM_DEFINITION_PROMPT = """請解釋以下金融術語的定義和應用場景：

術語：{term}

要求：
1. 提供清晰的定義
2. 說明計算公式（如適用）
3. 舉例說明實際應用
4. 指出常見誤區或注意事項"""
