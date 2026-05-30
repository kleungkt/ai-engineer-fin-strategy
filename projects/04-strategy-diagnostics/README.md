# 🔍 策略診斷助手

## 目標

打造一個 **AI 驅動的量化策略診斷與分析系統**，幫助交易者深入理解策略的行為、識別潛在問題、並提供改進建議：

1. **策略健康檢查**：自動檢測策略的常見問題（過擬合、倖存者偏差、前視偏差等）
2. **績效歸因分析**：分解策略收益來源（Alpha、Beta、行業、因子貢獻）
3. **風險剖析**：識別策略在不同市場環境下的風險暴露
4. **行為分析**：分析策略的交易行為模式（持倉時間、交易頻率、虧損模式）
5. **診斷報告生成**：生成結構化的診斷報告，包含發現與建議
6. **交互式探索**：用戶可以對診斷結果提問，獲得深入解釋

## 示例

### 策略健康檢查
```
用戶: 請診斷我的動量策略

系統: 📊 策略診斷報告 - Momentum Breakout v2

═══════════════════════════════════════════════
  🟢 整體健康度: 72/100 (良好，但有改進空間)
═══════════════════════════════════════════════

┌─────────────────────────────────────────────┐
│ ⚠️ 發現 3 個潛在問題                        │
├─────────────────────────────────────────────┤
│                                             │
│ 🔴 嚴重: 疑似過擬合                         │
│    訓練集 Sharpe: 2.1 vs 測試集 Sharpe: 0.6 │
│    → 建議: 使用 Walk-Forward 驗證            │
│                                             │
│ 🟡 中等: 最大回撤過大                        │
│    Max Drawdown: -28.5% (目標 < -15%)       │
│    → 建議: 加入動態倉位管理或止損機制         │
│                                             │
│ 🟢 輕微: 交易頻率偏低                        │
│    年均交易次數: 8 次                        │
│    → 建議: 考慮縮短信號週期或增加標的池       │
│                                             │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 📈 績效概覽                                  │
├─────────────────────────────────────────────┤
│ 年化報酬率      │ 15.2%                    │
│ 年化波動率      │ 18.7%                    │
│ Sharpe Ratio    │ 0.81                     │
│ Sortino Ratio   │ 1.12                     │
│ Max Drawdown    │ -28.5%                   │
│ Calmar Ratio    │ 0.53                     │
│ 勝率            │ 42%                      │
│ 盈虧比          │ 2.3:1                    │
│ 平均持倉天數    │ 12 天                    │
└─────────────────────────────────────────────┘
```

### 績效歸因分析
```
用戶: 分析一下這個策略的超額收益來源

系統: 📊 Alpha 歸因分析

收益分解:
├── 市場 Beta 貢獻:    +8.3%  (佔比 54.6%)
├── 行業配置 Alpha:    +2.1%  (佔比 13.8%)
├── 個股選擇 Alpha:    +3.8%  (佔比 25.0%)
├── 動量因子貢獻:      +1.2%  (佔比 7.9%)
└── 殘差 (未解釋):     -0.2%  (佔比 -1.3%)

💡 洞察:
- 策略主要收益來自市場 Beta，純 Alpha 有限
- 行業配置有正貢獻，主要超配了半導體和AI類股
- 個股選擇能力不錯，但波動較大
- 建議: 考慮加入市場中性對沖，提升 Alpha 比重
```

### 交易行為分析
```
用戶: 我的策略有什麼交易行為上的問題？

系統: 🔍 交易行為分析

┌─────────────────────────────────────────────┐
│ 🧠 行為偏誤檢測                              │
├─────────────────────────────────────────────┤
│                                             │
│ ❌ 處置效應 (Disposition Effect)             │
│    盈利交易平均持倉: 5 天                    │
│    虧損交易平均持倉: 18 天                   │
│    → 你傾向於過早獲利了結，過晚止損          │
│                                             │
│ ❌ 過度交易 (Overtrading)                    │
│    月均交易次數: 45 次                       │
│    手續費佔收益: 32%                         │
│    → 交易成本侵蝕了大量收益                  │
│                                             │
│ ⚠️ 損失趨避 (Loss Aversion)                 │
│    連續虧損後交易量增加 150%                 │
│    → 可能在情緒影響下加大賭注               │
│                                             │
│ ✅ 止損紀律良好                              │
│    85% 的止損單得到執行                      │
│                                             │
└─────────────────────────────────────────────┘
```

## 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **數據分析** | pandas, numpy, scipy | 績效計算與統計分析 |
| **可視化** | Plotly, Matplotlib, Seaborn | 圖表與報告 |
| **因子分析** | statsmodels, sklearn | 回歸歸因分析 |
| **LLM** | GPT-4 / Claude | 報告生成與對話 |
| **後端** | FastAPI | API 服務 |
| **前端** | Streamlit | 互動界面 |
| **模板** | Jinja2 | 報告模板渲染 |

## 架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit 互動界面                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │策略上傳  │ │診斷面板  │ │圖表探索  │ │ AI 對話      │   │
│  │(代碼/結果)│ │(健康度)  │ │(交互式)  │ │ (問答式診斷) │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘   │
└───────┼────────────┼───────────┼──────────────┼────────────┘
        │            │           │              │
        ▼            ▼           ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API 服務層 (FastAPI)                       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 診斷引擎 (Diagnostic Engine)           │   │
│  │                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │健康檢查器   │  │績效分析器   │  │行為分析器   │  │   │
│  │  │             │  │             │  │             │  │   │
│  │  │- 過擬合檢測 │  │- 收益分解   │  │- 持倉分析   │  │   │
│  │  │- 偏差檢測   │  │- 風險歸因   │  │- 頻率分析   │  │   │
│  │  │- 穩定性測試 │  │- 因子暴露   │  │- 行為偏誤   │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │   │
│  │         │               │               │          │   │
│  │         ▼               ▼               ▼          │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │            診斷結果整合器                    │    │   │
│  │  │  - 問題優先級排序                           │    │   │
│  │  │  - 建議生成 (LLM)                           │    │   │
│  │  │  - 報告渲染                                 │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              AI 對話引擎 (Chat Engine)                │   │
│  │  - 診斷結果問答                                      │   │
│  │  - 改進建議對話                                      │   │
│  │  - 策略知識問答                                      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       數據層                                 │
│  ┌──────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │策略結果  │ │市場基準數據  │ │ 因子數據                  │ │
│  │(交易記錄)│ │(指數/ETF)    │ │ (Fama-French/自定義)     │ │
│  └──────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
04-strategy-diagnostics/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依賴
├── .env.example                       # 環境變數範例
├── config/
│   ├── settings.py                    # 配置管理
│   ├── thresholds.py                  # 診斷閾值設定
│   │   # 例: max_drawdown_warn = -0.15
│   │   # 例: sharpe_good = 1.0
│   └── prompts/
│       ├── diagnosis_report.txt       # 診斷報告生成提示
│       ├── behavior_analysis.txt      # 行為分析提示
│       └── improvement_suggestions.txt # 改進建議提示
├── src/
│   ├── __init__.py
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── base_checker.py            # 診斷器基類
│   │   ├── overfit_detector.py        # 過擬合檢測
│   │   │   # - 訓練/測試集績效差異
│   │   │   # - 參數敏感度分析
│   │   │   # - Monte Carlo 模擬
│   │   ├── bias_detector.py           # 偏差檢測
│   │   │   # - 存活者偏差
│   │   │   # - 前視偏差
│   │   │   # - 交易成本偏差
│   │   ├── stability_analyzer.py      # 穩定性分析
│   │   │   # - 滾動窗口績效
│   │   │   # - 市場環境適應性
│   │   │   # - 信號一致性
│   │   └── health_scorer.py           # 健康度評分
│   ├── performance/
│   │   ├── __init__.py
│   │   ├── returns_analyzer.py        # 收益分析
│   │   │   # - 年化/月化收益
│   │   │   # - 收益分布
│   │   │   # - 收益持續性
│   │   ├── risk_analyzer.py           # 風險分析
│   │   │   # - VaR / CVaR
│   │   │   # - 最大回撤分析
│   │   │   # - 波動率分析
│   │   ├── drawdown_analyzer.py       # 回撤分析
│   │   │   # - 回撤持續時間
│   │   │   # - 回撤恢復時間
│   │   │   # - 回撤頻率
│   │   └── benchmark_comparator.py    # 基準比較
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── alpha_attribution.py       # Alpha 歸因
│   │   ├── factor_exposure.py         # 因子暴露分析
│   │   │   # - Fama-French 三/五因子
│   │   │   # - 動量因子
│   │   │   # - 自定義因子
│   │   ├── sector_attribution.py      # 行業歸因
│   │   └── timing_analysis.py         # 擇時能力分析
│   ├── behavior/
│   │   ├── __init__.py
│   │   ├── trading_pattern.py         # 交易模式分析
│   │   │   # - 交易頻率
│   │   │   # - 持倉時間分布
│   │   │   # - 交易時間偏好
│   │   ├── bias_detector.py           # 行為偏誤檢測
│   │   │   # - 處置效應
│   │   │   # - 過度交易
│   │   │   # - 損失趨避
│   │   │   # - 確認偏誤
│   │   └── discipline_score.py        # 紀律評分
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── report_generator.py        # 報告生成器
│   │   ├── chart_factory.py           # 圖表工廠
│   │   │   # - 資金曲線圖
│   │   │   # - 回撤圖
│   │   │   # - 月度收益熱力圖
│   │   │   # - 因子暴露圖
│   │   └── llm_narrator.py            # LLM 報告敘述
│   └── chat/
│       ├── __init__.py
│       ├── chat_engine.py             # 對話引擎
│       └── context_manager.py         # 對話上下文管理
├── data/
│   ├── factor_data/                   # 因子數據
│   │   ├── fama_french_3factor.csv
│   │   └── fama_french_5factor.csv
│   ├── benchmark_data/                # 基準指數數據
│   │   ├── taiex.csv
│   │   └── sp500.csv
│   └── sample_strategies/             # 範例策略結果
│       ├── momentum_v1/
│       └── mean_reversion_v1/
├── api/
│   ├── main.py                        # FastAPI 應用
│   └── routes/
│       ├── diagnose.py                # 診斷端點
│       ├── report.py                  # 報告端點
│       └── chat.py                    # 對話端點
├── app/
│   ├── streamlit_app.py               # Streamlit 主界面
│   ├── pages/
│   │   ├── 1_health_check.py          # 健康檢查頁
│   │   ├── 2_performance.py           # 績效分析頁
│   │   ├── 3_attribution.py           # 歸因分析頁
│   │   ├── 4_behavior.py              # 行為分析頁
│   │   └── 5_chat.py                  # AI 對話頁
│   └── components/
│       ├── health_gauge.py            # 健康度儀表盤
│       ├── drawdown_chart.py          # 回撤圖表
│       └── attribution_sankey.py      # 歸因桑基圖
├── notebooks/
│   ├── 01_basic_diagnostics.ipynb     # 基礎診斷實驗
│   ├── 02_factor_analysis.ipynb       # 因子分析實驗
│   ├── 03_behavior_analysis.ipynb     # 行為分析實驗
│   └── 04_report_generation.ipynb     # 報告生成實驗
└── tests/
    ├── test_diagnostics.py
    ├── test_performance.py
    ├── test_attribution.py
    └── test_behavior.py
```

## 學習重點

### 1. 量化績效分析
- **風險調整收益**：Sharpe, Sortino, Calmar, Information Ratio
- **回撤分析**：最大回撤、回撤持續時間、恢復時間
- **收益分布**：偏態、峰態、尾部風險
- **績效持續性**：滾動窗口分析、績效穩定性

### 2. 因子歸因分析
- **Fama-French 模型**：市場、規模、價值因子的分離
- **Alpha 與 Beta**：區分市場收益與超額收益
- **行業歸因**：行業配置 vs 個股選擇的貢獻
- **擇時能力**：Treon's measure, Henriksson-Merton test

### 3. 行為金融學應用
- **處置效應**：過早賣出盈利、過晚賣出虧損
- **過度交易**：交易頻率與收益的關係
- **損失趨避**：風險偏好的非對稱性
- **確認偏誤**：忽略反面證據的傾向

### 4. 過擬合檢測方法
- **樣本外測試**：訓練集 vs 測試集績效比較
- **Walk-Forward Analysis**：滾動窗口驗證
- **Monte Carlo 模擬**：隨機參數的績效分布
- **Deflated Sharpe Ratio**：考慮多重比較的 Sharpe 調整

### 5. LLM 驅動的報告生成
- **結構化數據 → 自然語言**：將數字轉化為有意義的敘述
- **問題識別與建議生成**：基於規則 + LLM 的混合方法
- **交互式診斷**：用戶可以對診斷結果提問

## 開發步驟

### Step 1: 績效分析基礎（Day 1-3）
```python
# 實現收益分析器
# src/performance/returns_analyzer.py

class ReturnsAnalyzer:
    def calculate_metrics(self, returns: pd.Series) -> dict:
        """計算績效指標"""
        return {
            'annual_return': self._annual_return(returns),
            'annual_volatility': self._annual_volatility(returns),
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'max_drawdown': self._max_drawdown(returns),
            'calmar_ratio': self._calmar_ratio(returns),
            'win_rate': self._win_rate(returns),
            'profit_factor': self._profit_factor(returns),
        }

# 實現回撤分析器
# src/performance/drawdown_analyzer.py

class DrawdownAnalyzer:
    def analyze(self, equity_curve: pd.Series) -> dict:
        """詳細回撤分析"""
        return {
            'max_drawdown': ...,
            'max_drawdown_duration': ...,
            'recovery_time': ...,
            'drawdown_periods': [...],  # 所有回撤期間
        }
```

### Step 2: 過擬合與偏差檢測（Day 4-6）
```python
# 實現過擬合檢測器
# src/diagnostics/overfit_detector.py

class OverfitDetector:
    def detect(self, train_metrics, test_metrics) -> DiagnosisResult:
        """檢測過擬合"""
        issues = []
        
        # 1. 訓練/測試績效差異
        if train_metrics['sharpe'] - test_metrics['sharpe'] > 1.0:
            issues.append(Issue(
                severity='high',
                type='overfit',
                description='訓練集與測試集 Sharpe 差異過大'
            ))
        
        # 2. 參數敏感度
        sensitivity = self._parameter_sensitivity(strategy, data)
        if sensitivity > threshold:
            issues.append(...)
        
        return DiagnosisResult(issues=issues)

# 實現偏差檢測器
# src/diagnostics/bias_detector.py
# 檢測存活者偏差、前視偏差、交易成本偏差
```

### Step 3: 因子歸因分析（Day 7-9）
```python
# 實現 Alpha 歸因
# src/attribution/alpha_attribution.py

class AlphaAttribution:
    def decompose(self, portfolio_returns, factor_data) -> dict:
        """收益分解"""
        # OLS 回歸: R_p = alpha + beta_m * R_m + beta_s * SMB + ...
        model = sm.OLS(portfolio_returns, factor_data).fit()
        
        return {
            'alpha': model.params[0],
            'market_beta': model.params[1],
            'factor_contributions': {...},
            'r_squared': model.rsquared,
        }

# 實現因子暴露分析
# src/attribution/factor_exposure.py
# 支持 Fama-French 三/五因子模型
```

### Step 4: 行為分析（Day 10-12）
```python
# 實現交易行為分析
# src/behavior/trading_pattern.py

class TradingPatternAnalyzer:
    def analyze(self, trades: pd.DataFrame) -> dict:
        """分析交易模式"""
        return {
            'avg_holding_period': ...,
            'trade_frequency': ...,
            'win_loss_ratio': ...,
            'avg_win_vs_avg_loss': ...,
        }

# 實現行為偏誤檢測
# src/behavior/bias_detector.py

class BehavioralBiasDetector:
    def detect_disposition_effect(self, trades) -> float:
        """檢測處置效應"""
        pass
    
    def detect_overtrading(self, trades, returns) -> bool:
        """檢測過度交易"""
        pass
    
    def detect_loss_aversion(self, trades) -> dict:
        """檢測損失趨避"""
        pass
```

### Step 5: 健康度評分系統（Day 13-14）
```python
# 實現健康度評分
# src/diagnostics/health_scorer.py

class HealthScorer:
    """綜合健康度評分 (0-100)"""
    
    def score(self, diagnostics: AllDiagnostics) -> HealthReport:
        scores = {
            'overfit_risk': self._score_overfit(diagnostics),
            'risk_level': self._score_risk(diagnostics),
            'behavior': self._score_behavior(diagnostics),
            'stability': self._score_stability(diagnostics),
            'alpha_quality': self._score_alpha(diagnostics),
        }
        
        total = weighted_average(scores, weights)
        return HealthReport(score=total, details=scores)
```

### Step 6: 報告生成（Day 15-17）
```python
# 實現圖表工廠
# src/reporting/chart_factory.py

class ChartFactory:
    def equity_curve(self, data) -> plotly.Figure:
        """資金曲線圖"""
        pass
    
    def drawdown_chart(self, data) -> plotly.Figure:
        """回撤圖"""
        pass
    
    def monthly_heatmap(self, returns) -> plotly.Figure:
        """月度收益熱力圖"""
        pass
    
    def factor_radar(self, exposures) -> plotly.Figure:
        """因子暴露雷達圖"""
        pass

# 實現 LLM 報告敘述
# src/reporting/llm_narrator.py
# 將診斷數據轉化為自然語言報告
```

### Step 7: AI 對話功能（Day 18-19）
```python
# 實現對話引擎
# src/chat/chat_engine.py

class DiagnosticChatEngine:
    def ask(self, question: str, context: DiagnosticContext) -> str:
        """用戶對診斷結果提問"""
        # 將診斷結果作為上下文
        # 使用 RAG 檢索相關策略知識
        # 生成回答
        pass
```

### Step 8: 前端界面（Day 20-23）
```python
# 建立 Streamlit 應用
# app/streamlit_app.py

# 健康檢查頁面
# - 策略代碼/結果上傳
# - 健康度儀表盤
# - 問題列表與建議

# 績效分析頁面
# - 資金曲線
# - 回撤分析
# - 月度收益熱力圖

# 歸因分析頁面
# - 因子暴露雷達圖
# - 收益分解桑基圖
# - 行業配置分析

# 行為分析頁面
# - 交易模式統計
# - 行為偏誤檢測結果
# - 改進建議
```

### Step 9: 測試與優化（Day 24-25）
```bash
# 單元測試
pytest tests/ -v

# 用真實策略數據測試
python scripts/test_diagnostics.py

# 優化報告生成速度
# 優化圖表渲染性能
```

### Step 10: 進階功能（Day 26-30）
```
- [ ] 支持實時診斷（連接模擬交易系統）
- [ ] 策略比較功能（多策略對比診斷）
- [ ] 診斷歷史追蹤（策略改進的歷史記錄）
- [ ] 自動化監控（定期診斷 + 告警）
- [ ] 生成 PDF/HTML 報告
- [ ] 支持更多因子模型（Barra, Axioma）
```

---

> 💡 **小提示**：診斷系統的核心價值在於「能發現用戶自己沒注意到的問題」。先從最常見的陷阱（過擬合、處置效應、忽略交易成本）開始，這些問題在大多數策略中都存在。使用具體的數字和對比（如「你的策略在牛市表現優異，但在熊市回撤是基準的 2 倍」）比抽象描述更有說服力。
