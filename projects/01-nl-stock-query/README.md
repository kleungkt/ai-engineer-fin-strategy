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
- **框架**: LangChain, FastAPI

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
│   ├── api.py             # FastAPI REST API 服務
│   ├── parser.py          # LLM 解析用戶輸入
│   ├── indicators.py      # 技術指標計算
│   ├── screener.py        # 股票篩選器
│   ├── backtester.py      # 回測引擎
│   ├── diagnostics.py     # 策略診斷報告
│   ├── patterns.py        # K 線型態偵測
│   ├── data_fetcher.py    # 真實數據拉取
│   └── main.py            # CLI 主入口
├── tests/
│   └── test_parser.py
└── examples/
    └── sample_queries.md
```

---

## API

The project ships with a **FastAPI** REST API (`src/api.py`) that exposes the full natural-language query → parse → screen → backtest pipeline over HTTP.

### Start the server

```bash
cd projects/01-nl-stock-query/src
pip install fastapi uvicorn
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger UI is available at **http://localhost:8000/docs**.

### Endpoints

#### `POST /query` — Natural Language Stock Screening

Parse a free-form question into structured indicator conditions and screen all stocks in the target market.

| Field    | Type   | Default | Description                                          |
|----------|--------|---------|------------------------------------------------------|
| `query`  | string | —       | Natural-language screening question (required)       |
| `market` | string | `A股`   | Market scope: A股, 美股, 港股, 加密貨幣              |
| `days`   | int    | `120`   | Historical trading days to evaluate (5–500)          |

```bash
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"query": "MACD golden cross and RSI below 30", "market": "A股", "days": 120}'
```

**Response**:

```json
{
  "query_intent": {
    "intent_type": "indicator_screener",
    "indicators": [
      {"name": "MACD", "comparison": "crossover", "value": null, "params": {}},
      {"name": "RSI", "comparison": "below", "value": 30.0, "params": {}}
    ],
    "stock_scope": "A股",
    "time_range": 30
  },
  "matched_stocks": [
    {"symbol": "AAPL", "matched": true, "indicators": {...}, "explanation": "..."},
    {"symbol": "MSFT", "matched": false, "indicators": {...}, "explanation": "..."}
  ],
  "total_scanned": 20
}
```

---

#### `POST /backtest` — Backtest from Natural Language

Describe a strategy in plain language and run a historical backtest on a specific symbol.

| Field          | Type   | Default    | Description                               |
|----------------|--------|------------|-------------------------------------------|
| `query`        | string | —          | Strategy description (required)           |
| `symbol`       | string | —          | Ticker symbol, e.g. `AAPL` (required)     |
| `days`         | int    | `250`      | Historical trading days (30–1000)         |
| `initial_cash` | float  | `100000`   | Starting portfolio cash                   |

```bash
curl -X POST http://localhost:8000/backtest \
     -H 'Content-Type: application/json' \
     -d '{"query": "MACD golden cross", "symbol": "AAPL", "days": 250}'
```

**Response**:

```json
{
  "strategy_type": "macd_crossover",
  "params": {"fast": 12, "slow": 26, "signal": 9},
  "backtest": {"total_return": 0.15, "trades": [...], "equity_curve": [...]},
  "diagnostic": {"sharpe_ratio": 1.2, "max_drawdown": -0.08, "win_rate": 0.6}
}
```

> **Note**: The backtest endpoint requires `backtester.py` and `diagnostics.py` to be implemented. Returns HTTP 501 if these modules are missing.

---

#### `GET /patterns/{symbol}` — K-Line Pattern Detection

Scan historical candlestick data for common patterns (doji, hammer, engulfing, morning/evening star, etc.).

| Parameter | Location | Type | Default | Description                     |
|-----------|----------|------|---------|---------------------------------|
| `symbol`  | path     | string | —     | Ticker symbol, e.g. `AAPL`     |
| `days`    | query    | int    | `60`  | Look-back window (5–500)       |

```bash
curl http://localhost:8000/patterns/AAPL?days=60
```

**Response**:

```json
{
  "symbol": "AAPL",
  "patterns": {
    "doji": false,
    "hammer": true,
    "engulfing": false,
    "morning_star": false,
    "evening_star": false
  },
  "summary": "Detected: hammer"
}
```

---

#### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2026-05-30T22:58:00+00:00"
}
```

---

### Error Handling

All endpoints return standard HTTP error codes:

| Code | Meaning                                  |
|------|------------------------------------------|
| 200  | Success                                  |
| 422  | Validation / parse failure               |
| 404  | Symbol not found or insufficient data    |
| 501  | Feature not implemented (missing module) |
| 502  | LLM service unreachable                  |
| 500  | Unexpected internal error                |

### CORS

CORS is enabled for all origins (`*`) by default for development. Tighten `allow_origins` in production.

---

## 學習重點

1. **Prompt Engineering**: 如何設計 prompt 讓 LLM 準確提取意圖和實體
2. **結構化輸出**: Function calling / JSON mode
3. **技術指標**: MA, MACD, RSI 的計算邏輯
4. **REST API**: FastAPI + Pydantic 模型設計

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
