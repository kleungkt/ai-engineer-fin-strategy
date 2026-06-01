# Unified E2E Pipeline — 端到端 AI 策略平台

從自然語言查詢到完整策略診斷的全鏈路整合。

## 架構

```
用戶輸入（自然語言）
         │
         ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  P1: NL 解析    │───▶│  P3: 策略生成 + 回測 │───▶│  P4: 策略診斷   │
│  意圖識別        │    │  代碼驗證 + Backtrader│    │  評分 + AI 分析 │
│  股票篩選        │    │                      │    │                 │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  完整報告輸出   │
                    └─────────────────┘
```

## 使用方式

### CLI

```bash
# 演示模式（無需 API key）
python pipeline/main.py analyze "帮我找最近 RSI 低於 30 的 A 股"

# 真實 API 模式
python pipeline/main.py analyze "MACD 金叉且 RSI < 30" --live

# JSON 輸出
python pipeline/main.py analyze "布林通道突破策略" --json
```

### API

```bash
# 啟動服務
cd pipeline && PYTHONPATH=src uvicorn api:app --reload

# 調用
curl -X POST http://localhost:8000/pipeline/analyze \
  -H "Content-Type: application/json" \
  -d '{"nl_request": "帮我找最近 RSI 低於 30 的 A 股", "demo_mode": true}'
```

### Python

```python
from pipeline.unified_pipeline import UnifiedPipeline

pipeline = UnifiedPipeline(demo_mode=True)
result = pipeline.run("帮我找 MACD 金叉的股票")
print(result.summary)
```

## 演示輸出

```
🎯 請求: 帮我找最近 RSI 低於 30 的 A 股
📌 意圖解析 → indicator_screener, 指標: RSI, MACD
📊 股票篩選 → 5 支符合条件的股票
🤖 策略生成 → 3 個策略 (每支股票一個)
🏥 策略診斷 → 評分 0-100, AI 分析報告
```

## 組件映射

| 模組 | 來源 | 功能 |
|------|------|------|
| `parse_query` | P1/src/parser.py | 自然語言意圖解析 |
| `fetch_stock_daily` | P1/src/data_fetcher.py | 股票數據獲取 |
| `StrategyAgent` | P3/src/strategy_agent.py | NL→策略代碼生成 |
| `run_backtest` | P3/src/backtester.py | Backtrader 回測執行 |
| `evaluate_backtest` | P4/src/evaluator.py | 多維度評分 |
| `generate_analysis` | P4/src/ai_analyst.py | GPT-4o AI 分析 |
| `format_report` | P4/src/formatter.py | 報告格式化 |

## 注意事項

- P4 使用相對導入 (`from .models`)，需將 `projects/04-strategy-diagnostics` 作為包加載
- 所有項目共享同一個 `_root` 路徑解析，確保跨目錄引用正確
- 演示模式 (`demo_mode=True`) 使用隨機模擬數據，無需 OpenAI API key