# 🤖 AI 策略生成器（核心項目）

## 目標

打造一個 **AI 驅動的量化交易策略生成系統**，讓用戶能用自然語言描述交易想法，系統自動將其轉化為可執行的量化策略代碼：

1. **自然語言理解**：解析用戶的交易邏輯描述（如「當 RSI 低於 30 且成交量放大時買入」）
2. **策略代碼生成**：自動生成結構化的 Python 策略代碼
3. **參數優化建議**：根據歷史數據建議合理的參數範圍
4. **策略回測整合**：生成的代碼可直接在主流回測框架中運行
5. **多策略支持**：支持技術指標、統計套利、機器學習等多種策略類型
6. **策略版本管理**：追蹤策略的迭代歷史與修改差異

## 示例

### 基本策略生成
```
用戶: 幫我寫一個動量策略，當股票收盤價突破20日高點時買入，
     跌破10日低點時賣出，同時設置2%的止損

系統: 已生成策略 "Momentum Breakout v1"：

┌─────────────────────────────────────────────────────┐
│  策略概覽                                            │
│  類型: 動量突破                                      │
│  信號: 收盤價 > 20日高點 → 買入                       │
│        收盤價 < 10日低點 → 賣出                       │
│  風控: 2% 固定止損                                   │
│  建議回測區間: 2020-01-01 至 2024-12-31              │
└─────────────────────────────────────────────────────┘

```python
class MomentumBreakout(Strategy):
    params = {
        'entry_period': 20,
        'exit_period': 10,
        'stop_loss_pct': 0.02,
    }
    
    def generate_signals(self, data):
        high_20 = data['high'].rolling(self.params['entry_period']).max()
        low_10 = data['low'].rolling(self.params['exit_period']).min()
        
        data['signal'] = 0
        data.loc[data['close'] > high_20.shift(1), 'signal'] = 1
        data.loc[data['close'] < low_10.shift(1), 'signal'] = -1
        return data
```

### 複雜策略生成
```
用戶: 設計一個均值回歸策略，用布林通道和RSI結合，
     在布林通道下軌且RSI超賣時分批建倉，
     到上軌且RSI超買時分批平倉，
     同時用ATR做動態止損

系統: 已生成策略 "Mean Reversion Bollinger-RSI v1"：
- 使用 20 期布林通道（2倍標準差）
- RSI(14) 低於30 + 價格觸及下軌 → 建倉 50%
- RSI(14) 低於20 + 價格跌破下軌 → 加倉 50%
- RSI(14) 高於70 + 價格觸及上軌 → 平倉 50%
- RSI(14) 高於80 + 價格突破上軌 → 全部平倉
- ATR(14) × 2 作為動態止損距離
```

### 策略優化建議
```
用戶: 優化一下這個策略的參數

系統: 基於歷史數據分析，建議以下參數調整：
- entry_period: 20 → 15（提高信號敏感度）
- stop_loss_pct: 2% → 1.5%（降低單筆最大虧損）

⚠️ 注意：參數優化可能導致過擬合，建議：
1. 使用 Walk-Forward 驗證
2. 在樣本外數據上測試
3. 保持參數的經濟學意義
```

## 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **LLM** | GPT-4 / Claude / DeepSeek Coder | 自然語言理解與代碼生成 |
| **代碼框架** | Python AST, CodeGen | 代碼解析、生成、驗證 |
| **回測引擎** | Backtrader / VectorBT / Zipline | 策略回測 |
| **數據源** | yfinance, TWSE API, TEJ | 市場數據 |
| **技術指標** | TA-Lib, pandas_ta | 技術分析計算 |
| **後端** | FastAPI | API 服務 |
| **前端** | Streamlit + Plotly | 互動界面與可視化 |
| **存儲** | SQLite, GitPython | 策略存儲與版本管理 |

## 架構

```
┌─────────────────────────────────────────────────────────────┐
│                     用戶交互層                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Streamlit 互動界面                         │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐     │    │
│  │  │策略描述  │ │參數調整  │ │ 回測結果可視化    │     │    │
│  │  │輸入框    │ │滑桿/表單 │ │ (K線+信號+曲線)  │     │    │
│  │  └────┬─────┘ └────┬─────┘ └────────┬─────────┘     │    │
│  └───────┼────────────┼───────────────┼─────────────────┘    │
└──────────┼────────────┼───────────────┼──────────────────────┘
           │            │               │
           ▼            ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    API 服務層 (FastAPI)                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                策略生成引擎                          │    │
│  │                                                     │    │
│  │  ┌──────────────┐    ┌─────────────────────────┐    │    │
│  │  │ NLU 解析器   │    │ 代碼生成器               │    │    │
│  │  │ - 意圖識別   │───▶│ - 模板選擇               │    │    │
│  │  │ - 實體提取   │    │ - LLM 代碼生成           │    │    │
│  │  │ - 邏輯結構化│    │ - AST 驗證               │    │    │
│  │  └──────────────┘    │ - 安全檢查               │    │    │
│  │                      └───────────┬─────────────┘    │    │
│  │                                  │                  │    │
│  │                                  ▼                  │    │
│  │  ┌──────────────┐    ┌─────────────────────────┐    │    │
│  │  │ 參數優化器   │    │ 策略執行器               │    │    │
│  │  │ - 網格搜索   │◀──▶│ - 代碼執行沙箱           │    │    │
│  │  │ - 貝葉斯優化 │    │ - 回測引擎封裝           │    │    │
│  │  │ - Walk-Forward│   │ - 績效計算               │    │    │
│  │  └──────────────┘    └─────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      數據與存儲層                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │市場數據  │ │策略存儲  │ │回測結果  │ │ 版本歷史     │   │
│  │(SQLite)  │ │(JSON/DB) │ │(SQLite)  │ │ (Git-like)   │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
03-ai-strategy-generator/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依賴
├── .env.example                      # 環境變數範例
├── config/
│   ├── settings.py                   # 全局配置
│   ├── strategy_templates.py         # 策略模板庫
│   └── prompts/
│       ├── system_prompt.txt         # LLM 系統提示
│       ├── strategy_gen.txt          # 策略生成提示
│       ├── parameter_optimize.txt    # 參數優化提示
│       └── code_review.txt           # 代碼審查提示
├── src/
│   ├── __init__.py
│   ├── nlu/
│   │   ├── __init__.py
│   │   ├── intent_classifier.py      # 意圖分類
│   │   ├── entity_extractor.py       # 實體提取
│   │   └── logic_parser.py           # 交易邏輯解析
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── code_generator.py         # LLM 代碼生成
│   │   ├── template_engine.py        # 模板引擎
│   │   ├── ast_validator.py          # AST 語法驗證
│   │   └── safety_checker.py         # 代碼安全檢查
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── parameter_optimizer.py    # 參數優化
│   │   ├── walk_forward.py           # Walk-Forward 驗證
│   │   └── overfit_detector.py       # 過擬合檢測
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                 # 回測引擎封裝
│   │   ├── metrics.py                # 績效指標計算
│   │   ├── report_generator.py       # 回測報告生成
│   │   └── benchmark.py              # 基準比較
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_data.py            # 市場數據接口
│   │   ├── data_cleaner.py           # 數據清洗
│   │   └── feature_engineer.py       # 特徵工程
│   └── storage/
│       ├── __init__.py
│       ├── strategy_store.py         # 策略存儲
│       └── version_control.py        # 策略版本管理
├── strategies/
│   ├── momentum/                     # 動量策略模板
│   │   ├── breakout.py
│   │   └── dual_ma.py
│   ├── mean_reversion/               # 均值回歸模板
│   │   ├── bollinger.py
│   │   └── rsi_extreme.py
│   ├── trend_following/              # 趨勢跟蹤模板
│   │   ├── turtle.py
│   │   └── channel_breakout.py
│   └── ml_based/                     # 機器學習策略模板
│       ├── random_forest.py
│       └── lstm_predictor.py
├── api/
│   ├── main.py                       # FastAPI 應用
│   ├── routes/
│   │   ├── generate.py               # 策略生成端點
│   │   ├── backtest.py               # 回測端點
│   │   ├── optimize.py               # 優化端點
│   │   └── strategies.py             # 策略管理端點
│   └── models.py                     # Pydantic 模型
├── app/
│   ├── streamlit_app.py              # Streamlit 主界面
│   ├── pages/
│   │   ├── 1_strategy_builder.py     # 策略構建器
│   │   ├── 2_backtest_lab.py         # 回測實驗室
│   │   ├── 3_parameter_tuner.py      # 參數調整器
│   │   └── 4_strategy_gallery.py     # 策略展示廳
│   └── components/
│       ├── chart.py                  # 圖表組件
│       └── metrics_card.py           # 指標卡片
├── notebooks/
│   ├── 01_strategy_prototyping.ipynb # 策略原型開發
│   ├── 02_llm_code_gen.ipynb        # LLM 代碼生成實驗
│   ├── 03_parameter_optimization.ipynb # 參數優化實驗
│   └── 04_walk_forward.ipynb        # Walk-Forward 驗證
├── tests/
│   ├── test_nlu.py
│   ├── test_generator.py
│   ├── test_optimizer.py
│   └── test_backtest.py
└── scripts/
    ├── init_templates.py             # 初始化策略模板
    └── benchmark_strategies.py       # 策略基準測試
```

## 學習重點

### 1. LLM 代碼生成的最佳實踐
- **結構化輸出**：使用 JSON schema 約束 LLM 輸出格式
- **Few-shot 範例**：提供高質量的策略代碼範例
- **代碼驗證**：AST 解析 + 靜態分析確保生成代碼可執行
- **安全執行**：沙箱環境執行用戶代碼，防止惡意代碼

### 2. 自然語言到交易邏輯的映射
- **意圖分類**：識別策略類型（動量、均值回歸、趨勢跟蹤等）
- **實體識別**：提取指標名稱、參數值、閾值、時間框架
- **邏輯結構化**：將模糊描述轉化為精確的條件表達式

### 3. 回測框架設計
- **事件驅動 vs 向量化**：兩種回測引擎的優缺點
- **滑點與手續費**：真實交易成本的模擬
- **存活者偏差**：處理下市股票和成分股調整
- **績效指標**：Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor

### 4. 參數優化與過擬合防護
- **網格搜索**：暴力但全面的參數搜索
- **貝葉斯優化**：高效的智能參數搜索
- **Walk-Forward Analysis**：滾動窗口驗證防止過擬合
- **交叉驗證**：時間序列交叉驗證的特殊考量

### 5. 策略模板設計模式
- **策略基類**：統一的策略接口設計
- **信號生成器**：可組合的信號邏輯
- **倉位管理器**：靈活的倉位控制
- **風控模組**：止損止盈的模組化設計

## 開發步驟

### Step 1: 策略模板框架搭建（Day 1-3）
```python
# 定義策略基類
# src/backtest/engine.py

class Strategy(ABC):
    """策略基類"""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信號"""
        pass
    
    def calculate_position_size(self, signal, portfolio):
        """計算倉位大小"""
        pass
    
    def apply_risk_management(self, position, data):
        """風控處理"""
        pass

# 實現幾個經典策略模板
# strategies/momentum/dual_ma.py
# strategies/mean_reversion/bollinger.py
# strategies/trend_following/turtle.py
```

### Step 2: 回測引擎（Day 4-6）
```python
# 實現回測引擎
# src/backtest/engine.py

class BacktestEngine:
    def run(self, strategy, data, initial_capital=1000000):
        """執行回測"""
        signals = strategy.generate_signals(data)
        portfolio = self._simulate_trades(signals, initial_capital)
        return self._generate_report(portfolio)
    
    def _simulate_trades(self, signals, capital):
        """模擬交易執行"""
        pass
    
    def _generate_report(self, portfolio):
        """生成回測報告"""
        pass

# 實現績效指標計算
# src/backtest/metrics.py
# - Sharpe Ratio, Sortino Ratio, Calmar Ratio
# - Max Drawdown, Win Rate, Profit Factor
# - 年化報酬率, 波動率
```

### Step 3: 市場數據模組（Day 7-8）
```python
# 實現數據接口
# src/data/market_data.py

class MarketDataProvider:
    def get_daily_data(self, symbol, start, end):
        """獲取日線數據"""
        pass
    
    def get_intraday_data(self, symbol, interval, start, end):
        """獲取分鐘線數據"""
        pass
    
    def get_financial_data(self, symbol):
        """獲取財務數據"""
        pass

# 支持數據源：yfinance, TEJ, TWSE API
```

### Step 4: NLU 模組 - 理解用戶意圖（Day 9-11）
```python
# 實現意圖分類
# src/nlu/intent_classifier.py
# 識別：策略生成、參數調整、回測請求、策略優化等

# 實現實體提取
# src/nlu/entity_extractor.py
# 提取：指標(RSI, MA, MACD)、參數(週期、閾倉)、動作(買入、賣出)

# 實現邏輯解析
# src/nlu/logic_parser.py
# 將自然語言轉化為結構化的交易邏輯 JSON
```

### Step 5: LLM 代碼生成（Day 12-15）
```python
# 實現代碼生成器
# src/generator/code_generator.py

class StrategyCodeGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_template = load_prompt("strategy_gen.txt")
    
    def generate(self, logic: TradingLogic) -> str:
        """生成策略代碼"""
        prompt = self.prompt_template.format(logic=logic.to_json())
        code = self.llm.generate(prompt)
        
        # AST 驗證
        self.ast_validator.validate(code)
        
        # 安全檢查
        self.safety_checker.check(code)
        
        return code

# 編寫高質量的 prompt templates
# config/prompts/strategy_gen.txt
```

### Step 6: 參數優化模組（Day 16-18）
```python
# 實現參數優化器
# src/optimizer/parameter_optimizer.py

class ParameterOptimizer:
    def grid_search(self, strategy, data, param_grid):
        """網格搜索"""
        pass
    
    def bayesian_optimize(self, strategy, data, param_space, n_trials=100):
        """貝葉斯優化"""
        pass

# 實現 Walk-Forward 驗證
# src/optimizer/walk_forward.py

class WalkForwardValidator:
    def validate(self, strategy, data, n_splits=5):
        """滾動窗口驗證"""
        pass
```

### Step 7: 前端界面（Day 19-22）
```python
# 建立 Streamlit 應用
# app/streamlit_app.py

# 策略構建器頁面
# - 自然語言輸入框
# - 策略參數調整滑桿
# - 實時預覽生成的代碼

# 回測實驗室頁面
# - 股票選擇器
# - 日期範圍選擇
# - 回測結果圖表（K線 + 信號 + 資金曲線）

# 參數調整器頁面
# - 參數網格搜索
# - 優化進度顯示
# - 參數敏感度圖表
```

### Step 8: 測試與文檔（Day 23-25）
```bash
# 單元測試
pytest tests/ -v

# 整合測試
pytest tests/integration/ -v

# 策略模板測試
python scripts/benchmark_strategies.py

# 編寫使用文檔
```

### Step 9: 進階功能（Day 26-30）
```
- [ ] 支持更多策略類型（統計套利、事件驅動）
- [ ] 加入多資產組合策略
- [ ] 實時模擬交易（Paper Trading）
- [ ] 策略分享與社群功能
- [ ] 接入券商 API（富果、永豐）
- [ ] 策略風險分析（VaR, CVaR）
```

---

> 💡 **小提示**：策略生成的關鍵在於 Prompt Engineering。先手動寫出幾個高質量的策略代碼作為 few-shot 範例，這比任何優化技巧都有效。同時，確保回測引擎的可靠性——很多看似優秀的策略只是過擬合的結果。
