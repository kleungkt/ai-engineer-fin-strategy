# Project 1: 自然語言股票查詢器

## 目標

讓用戶用自然語言描述查詢條件，系統自動解析並返回符合條件的股票。

## 示例

```
用戶: "幫我查最近 MACD 金叉的 A 股"
系統: [返回符合條件的股票列表]

用戶: "RSI 低於 30 的科技股有哪些？"
系統: [返回超賣的科技股]
```

## 技術棧

- **LLM**: OpenAI / Claude (結構化輸出)
- **數據**: Tushare / AKShare (A 股數據)
- **指標計算**: Pandas + TA-Lib
- **框架**: LangChain

## 架構

```
用戶輸入 → LLM 解析 (意圖 + 實體) → 指標計算 → 篩選 → 返回結果
```

## 目錄結構

```
01-nl-stock-query/
├── README.md
├── src/
│   ├── __init__.py
│   ├── parser.py          # LLM 解析用戶輸入
│   ├── indicators.py      # 技術指標計算
│   ├── screener.py        # 股票篩選器
│   └── main.py            # 主入口
├── tests/
│   └── test_parser.py
└── examples/
    └── sample_queries.md
```

## 學習重點

1. **Prompt Engineering**: 如何設計 prompt 讓 LLM 準確提取意圖和實體
2. **結構化輸出**: Function calling / JSON mode
3. **技術指標**: MA, MACD, RSI 的計算邏輯

## 開發步驟

### Step 1: 設計 Prompt
設計一個 prompt，能從自然語言提取：
- 查詢類型（指標篩選 / 形態識別 / 基本面查詢）
- 股票代碼/範圍
- 指標條件（指標名、閾值、比較方式）

### Step 2: 實現指標計算
用 Pandas 計算常見技術指標。

### Step 3: 構建篩選器
根據 LLM 解析結果，篩選符合條件的股票。

### Step 4: 集成測試
端到端測試：自然語言 → 結果
