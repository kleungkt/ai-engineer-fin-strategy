# 🧬 金融 LLM 微調

## 目標

通過 **領域專用微調（Fine-tuning）** 打造一個專精金融領域的大語言模型，使其在金融任務上超越通用模型：

1. **數據工程**：構建高質量的金融領域訓練數據集
2. **模型微調**：使用 LoRA/QLoRA 等參數高效方法微調開源 LLM
3. **評估基準**：建立金融領域的評估基準測試集
4. **部署優化**：模型量化、推理優化、服務部署
5. **持續學習**：建立數據飛輪，持續改進模型能力
6. **垂直應用**：微調後的模型在特定金融任務上達到專業水平

## 示例

### 微調前 vs 微調後

```
問題: 分析台積電 (2330.TW) 最近一季的財報亮點

【通用模型 GPT-4】
台積電是全球最大的半導體代工廠...
（泛泛而談，缺乏深度，可能引用過時數據）

【微調後的金融模型】
📊 台積電 2024Q3 財報分析

核心指標:
- 營收: 7,596.9 億新台幣 (YoY +39.0%, QoQ +12.8%)
- 毛利率: 57.8% (超出財測上限 55.5%)
- 營業利益率: 47.5%
- EPS: 12.54 元新台幣

亮點:
1. AI 需求持續驅動：HPC 營收佔比達 51%，首次超過手機成為最大平台
2. 先進製程領先：3nm 營收佔比 20%，5nm 佔 32%
3. CoWoS 先進封裝產能持續擴張，支撐 AI 晶片需求
4. 管理層上調全年營收指引至「略高於 mid-20%」成長

風險提示:
- 地緣政治風險（美國/日本擴廠成本壓力）
- 消費電子復甦力度不確定
- 先進封裝產能仍是瓶頸
```

### 金融文本分析
```
問題: 解讀這段央行聲明的鴿鷹立場

輸入: "委員會認為，在當前通膨逐步回落但仍高於目標的背景下，
      維持利率不變是適當的。委員會將密切關注數據發展，
      並準備在必要時調整政策立場。"

【微調模型回答】
🕊️ 整體偏鴿派，但保持靈活性

分析:
1. 「通膨逐步回落」→ 對通膨前景樂觀，偏鴿
2. 「但仍高於目標」→ 承認風險猶存，保持謹慎
3. 「維持利率不變」→ 暫停升息週期，偏鴿
4. 「密切關注數據」→ 數據依賴型指引，中性偏鴿
5. 「必要時調整」→ 保留政策選項，避免過度承諾

預期影響:
- 短期：債券收益率小幅下降，股市溫和上漲
- 中期：市場預期下次會議維持不變的概率 ~80%
- 關鍵觀察指標：下月 CPI、就業數據
```

### 策略邏輯生成
```
問題: 根據以下研究論文結論，設計一個可執行的交易策略

輸入: "研究發現，在美國聯準會 FOMC 會議前 24 小時，
      股市波動率顯著下降，而會議後 48 小時波動率急劇上升。
      這種模式在過去 20 年中持續存在。"

【微調模型回答】
策略名稱: FOMC Volatility Pattern

邏輯:
1. 在 FOMC 會議前 2 個交易日賣出跨式期權（Short Straddle）
2. 在 FOMC 聲明公佈前 1 小時平倉
3. 在聲明公佈後 30 分鐘買入跨式期權（Long Straddle）
4. 在會議後 2 個交易日平倉

風險管理:
- 最大虧損限制：帳戶的 2%
- 若 VIX 已高於 30，不做空波動率
- 僅在有明確利率決議的會議執行

預期表現:
- 歷史回測年化報酬：8-12%
- 勝率：~65%
- 最大回撤：-8%
```

## 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **基礎模型** | Llama 3 / Qwen 2.5 / Mistral | 開源基座模型 |
| **微調框架** | Hugging Face PEFT, Unsloth | LoRA/QLoRA 微調 |
| **訓練加速** | DeepSpeed, Flash Attention 2 | 分散式訓練加速 |
| **數據處理** | pandas, datasets, jsonlines | 數據清洗與格式化 |
| **評估** | lm-eval-harness, 自定義基準 | 模型評估 |
| **部署** | vLLM, TGI, Ollama | 推理服務 |
| **監控** | W&B, TensorBoard | 訓練監控 |
| **量化** | GPTQ, AWQ, GGUF | 模型壓縮 |

## 架構

```
┌─────────────────────────────────────────────────────────────┐
│                    數據工程管線                               │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ 數據收集  │→│ 數據清洗  │→│ 數據格式化│→│ 質量過濾  │   │
│  │          │  │          │  │          │  │          │   │
│  │- SEC 報告│  │- 去重    │  │- 指令格式│  │- 語義過濾│   │
│  │- 財報    │  │- 去噪    │  │- 對話格式│  │- 長度過濾│   │
│  │- 新聞    │  │- PII移除 │  │- CoT 格式│  │- 質量評分│   │
│  │- 論文    │  │          │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                              │             │
│                                              ▼             │
│                                    ┌──────────────────┐    │
│                                    │ 訓練數據集       │    │
│                                    │ (JSONL 格式)     │    │
│                                    └────────┬─────────┘    │
└─────────────────────────────────────────────┼──────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    模型微調管線                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              基座模型 (Llama 3 8B)                   │    │
│  └──────────────────────────┬──────────────────────────┘    │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          LoRA / QLoRA 微調                           │    │
│  │                                                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │    │
│  │  │ LoRA Config │  │ Training    │  │ Checkpoint  │ │    │
│  │  │ - rank: 64  │  │ - epochs: 3 │  │ - save every│ │    │
│  │  │ - alpha: 128│  │ - lr: 2e-4  │  │   500 steps │ │    │
│  │  │ - target:   │  │ - bs: 4     │  │ - best model│ │    │
│  │  │   q,k,v,o   │  │ - warmup    │  │   selection │ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │    │
│  └──────────────────────────┬──────────────────────────┘    │
│                             │                               │
│                             ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │          模型合併與導出                               │    │
│  │  - LoRA 權重合併到基座模型                           │    │
│  │  - 導出為 HuggingFace / GGUF / ONNX 格式            │    │
│  │  - 量化 (GPTQ / AWQ / GGUF Q4_K_M)                 │    │
│  └──────────────────────────┬──────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    評估與部署                                 │
│                                                             │
│  ┌───────────────────┐    ┌─────────────────────────────┐   │
│  │ 評估基準測試       │    │ 部署                        │   │
│  │                   │    │                             │   │
│  │- 金融知識問答     │    │ ┌────────┐  ┌────────────┐  │   │
│  │- 財報分析         │    │ │ vLLM   │  │ API Gateway│  │   │
│  │- 策略邏輯生成     │    │ │ 服務   │→│ (FastAPI)  │  │   │
│  │- 風險評估         │    │ └────────┘  └────────────┘  │   │
│  │- 情緒分析         │    │                             │   │
│  │- 合規檢查         │    │ ┌────────────────────────┐  │   │
│  │                   │    │ │ 監控 & 數據飛輪        │  │   │
│  │ vs                │    │ │ - 用戶反饋收集         │  │   │
│  │                   │    │ │ - A/B 測試             │  │   │
│  │- GPT-4 baseline   │    │ │ - 持續訓練數據準備     │  │   │
│  │- 未微調基座模型   │    │ └────────────────────────┘  │   │
│  └───────────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 目錄結構

```
05-finllm-finetune/
├── README.md                           # 本文件
├── requirements.txt                    # Python 依賴
├── .env.example                        # 環境變數範例
├── Makefile                            # 常用命令快捷方式
├── config/
│   ├── settings.py                     # 全局配置
│   ├── training/
│   │   ├── lora_config.yaml            # LoRA 配置
│   │   ├── training_args.yaml          # 訓練參數
│   │   └── deepspeed_config.json       # DeepSpeed 配置
│   └── evaluation/
│       └── benchmark_config.yaml       # 評估配置
├── data/
│   ├── raw/                            # 原始數據
│   │   ├── financial_reports/          # 財務報告
│   │   ├── sec_filings/                # SEC 文件
│   │   ├── news_articles/             # 金融新聞
│   │   ├── research_papers/           # 研究論文
│   │   └── regulatory_docs/           # 法規文件
│   ├── processed/                      # 處理後數據
│   │   ├── instruction_data.jsonl      # 指令微調數據
│   │   ├── conversation_data.jsonl     # 對話數據
│   │   └── cot_data.jsonl             # 思維鏈數據
│   ├── evaluation/                     # 評估數據
│   │   ├── financial_qa.json           # 金融問答測試集
│   │   ├── report_analysis.json        # 報告分析測試集
│   │   ├── strategy_generation.json    # 策略生成測試集
│   │   └── sentiment_analysis.json     # 情緒分析測試集
│   └── synthetic/                      # 合成數據
│       └── gpt4_generated.jsonl        # GPT-4 生成的訓練數據
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py                # 數據收集器
│   │   │   ├── SECParser               #   SEC 文件解析
│   │   │   ├── FinancialReportParser   #   財報解析
│   │   │   └── NewsCrawler             #   新聞爬取
│   │   ├── cleaner.py                  # 數據清洗
│   │   │   ├── deduplicator            #   去重
│   │   │   ├── noise_filter            #   噪音過濾
│   │   │   └── pii_remover             #   PII 移除
│   │   ├── formatter.py                # 數據格式化
│   │   │   ├── InstructionFormatter    #   指令格式
│   │   │   ├── ConversationFormatter   #   對話格式
│   │   │   └── CoTFormatter            #   思維鏈格式
│   │   ├── augmentor.py                # 數據增強
│   │   │   ├── back_translation        #   回譯
│   │   │   ├── paraphrase              #   改寫
│   │   │   └── gpt4_generation         #   GPT-4 生成
│   │   └── quality_scorer.py           # 數據質量評分
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # 訓練主邏輯
│   │   │   ├── SFTTrainer              #   監督微調
│   │   │   └── DPOTrainer              #   DPO 對齊
│   │   ├── lora_manager.py             # LoRA 配置管理
│   │   ├── callbacks.py                # 訓練回調
│   │   │   ├── LoggingCallback         #   日誌記錄
│   │   │   ├── EvalCallback            #   定期評估
│   │   │   └── CheckpointCallback      #   模型保存
│   │   └── optimizer.py                # 優化器配置
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py                # 評估主邏輯
│   │   ├── benchmarks/
│   │   │   ├── financial_qa.py         # 金融問答基準
│   │   │   ├── report_analysis.py      # 報告分析基準
│   │   │   ├── sentiment.py            # 情緒分析基準
│   │   │   └── strategy_gen.py         # 策略生成基準
│   │   ├── metrics.py                  # 評估指標
│   │   │   ├── BLEU/ROUGE              #   文本相似度
│   │   │   ├── accuracy                #   準確率
│   │   │   └── llm_judge               #   LLM 評判
│   │   └── comparison.py               # 模型比較
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── quantizer.py                # 模型量化
│   │   │   ├── GPTQQuantizer           #   GPTQ 量化
│   │   │   ├── AWQQuantizer            #   AWQ 量化
│   │   │   └── GGUFConverter           #   GGUF 轉換
│   │   ├── serving.py                  # 推理服務
│   │   │   ├── VLLMServer              #   vLLM 部署
│   │   │   └── TGIServer               #   TGI 部署
│   │   └── api.py                      # API 服務
│   └── pipeline/
│       ├── __init__.py
│       ├── data_pipeline.py            # 數據處理管線
│       └── training_pipeline.py        # 訓練管線
├── scripts/
│   ├── prepare_data.py                 # 數據準備腳本
│   ├── train.py                        # 訓練啟動腳本
│   ├── evaluate.py                     # 評估腳本
│   ├── export_model.py                 # 模型導出腳本
│   ├── quantize.py                     # 量化腳本
│   └── deploy.sh                       # 部署腳本
├── notebooks/
│   ├── 01_data_exploration.ipynb       # 數據探索
│   ├── 02_data_quality.ipynb           # 數據質量分析
│   ├── 03_training_experiments.ipynb   # 訓練實驗
│   ├── 04_evaluation.ipynb             # 模型評估
│   └── 05_error_analysis.ipynb         # 錯誤分析
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_deployment.py
├── docker/
│   ├── Dockerfile.training             # 訓練環境
│   ├── Dockerfile.serving              # 推理服務
│   └── docker-compose.yaml             # 編排配置
└── outputs/
    ├── models/                         # 訓練好的模型
    ├── logs/                           # 訓練日誌
    └── evaluation_results/             # 評估結果
```

## 學習重點

### 1. 數據工程（最重要！）
- **數據質量 > 數據數量**：1000 條高質量數據 > 10000 條低質量數據
- **數據來源多樣性**：財報、新聞、論文、法規、對話
- **指令格式設計**：如何設計有效的指令-回應對
- **數據清洗**：去重、去噪、PII 移除、質量評分
- **數據增強**：回譯、改寫、LLM 生成

### 2. 參數高效微調（PEFT）
- **LoRA 原理**：低秩分解的思想與數學基礎
- **超參數調優**：rank, alpha, target_modules 的選擇
- **QLoRA**：4-bit 量化 + LoRA 的結合
- **訓練技巧**：學習率、warmup、gradient accumulation

### 3. 模型評估
- **自動評估**：BLEU, ROUGE, 準確率等指標
- **LLM 評判**：用 GPT-4 評估模型輸出質量
- **領域基準**：建立金融領域的標準測試集
- **錯誤分析**：識別模型的系統性弱點

### 4. 模型優化與部署
- **量化技術**：GPTQ, AWQ, GGUF 的比較
- **推理加速**：vLLM, Flash Attention, Continuous Batching
- **服務架構**：API 設計、負載均衡、監控

### 5. 數據飛輪
- **用戶反饋收集**：thumbs up/down, 修正記錄
- **主動學習**：識別模型不確定的樣本
- **持續訓練**：增量微調策略
- **A/B 測試**：新舊模型的效果比較

## 開發步驟

### Step 1: 環境搭建與基礎模型選擇（Day 1-2）
```bash
# 建立訓練環境
# 需要: NVIDIA GPU (建議 A100 40GB 或以上)

# 安裝依賴
pip install -r requirements.txt

# 下載基座模型
python scripts/download_base_model.py --model meta-llama/Meta-Llama-3-8B

# 測試基礎模型
python scripts/test_base_model.py --prompt "什麼是本益比？"
```

### Step 2: 數據收集與處理（Day 3-8）
```bash
# 數據收集
python scripts/prepare_data.py collect --sources sec,reports,news

# 數據清洗
python scripts/prepare_data.py clean --input data/raw --output data/cleaned

# 數據格式化（轉為指令微調格式）
python scripts/prepare_data.py format \
    --input data/cleaned \
    --output data/processed/instruction_data.jsonl \
    --format instruction

# 數據質量評估
python scripts/prepare_data.py quality-check \
    --input data/processed/instruction_data.jsonl

# 數據增強（用 GPT-4 生成更多訓練數據）
python scripts/prepare_data.py augment \
    --input data/processed/instruction_data.jsonl \
    --output data/synthetic/gpt4_generated.jsonl \
    --num-samples 5000
```

**數據格式範例**:
```json
{
  "instruction": "分析以下財務指標並給出投資建議",
  "input": "公司: 台積電\n本益比: 25.3\n營收成長率: 15%\n負債比率: 35%\nROE: 28%",
  "output": "📊 台積電財務分析\n\n1. 本益比 25.3 處於合理區間...\n2. 營收成長 15% 表現穩健...\n3. 負債比率 35% 財務結構健康...\n4. ROE 28% 資本效率優異...\n\n投資建議: 維持「買入」評級..."
}
```

### Step 3: LoRA 微調實驗（Day 9-14）
```python
# 配置 LoRA
# config/training/lora_config.yaml
lora_config:
  r: 64                    # LoRA rank
  lora_alpha: 128          # scaling factor
  target_modules:          # 要微調的模組
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  lora_dropout: 0.05
  bias: none
  task_type: CAUSAL_LM

# 開始訓練
python scripts/train.py \
    --config config/training/training_args.yaml \
    --data data/processed/instruction_data.jsonl \
    --output outputs/models/finllm-lora-v1

# 監控訓練
# 使用 W&B 或 TensorBoard 查看訓練曲線
```

**訓練參數建議**:
```yaml
training_args:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # effective batch size = 32
  learning_rate: 2e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_length: 4096
  bf16: true
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
```

### Step 4: 模型評估（Day 15-17）
```bash
# 運行評估基準
python scripts/evaluate.py \
    --model outputs/models/finllm-lora-v1 \
    --benchmark all \
    --output outputs/evaluation_results/v1.json

# 比較微調前後
python scripts/evaluate.py \
    --compare \
    --model-a meta-llama/Meta-Llama-3-8B \
    --model-b outputs/models/finllm-lora-v1 \
    --benchmark financial_qa

# LLM 評判
python scripts/evaluate.py \
    --model outputs/models/finllm-lora-v1 \
    --evaluator gpt-4 \
    --samples 100
```

### Step 5: 模型合併與量化（Day 18-19）
```bash
# 合併 LoRA 權重到基座模型
python scripts/export_model.py \
    --base-model meta-llama/Meta-Llama-3-8B \
    --lora-path outputs/models/finllm-lora-v1 \
    --output outputs/models/finllm-merged

# 量化為 GGUF 格式（用於 Ollama）
python scripts/quantize.py \
    --model outputs/models/finllm-merged \
    --format gguf \
    --quantization Q4_K_M \
    --output outputs/models/finllm-Q4_K_M.gguf

# 量化為 GPTQ 格式（用於 vLLM）
python scripts/quantize.py \
    --model outputs/models/finllm-merged \
    --format gptq \
    --bits 4 \
    --output outputs/models/finllm-gptq-4bit
```

### Step 6: 部署（Day 20-22）
```bash
# 使用 vLLM 部署
python -m vllm.entrypoints.openai.api_server \
    --model outputs/models/finllm-merged \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000

# 或使用 Ollama 部署（更簡單）
ollama create finllm -f Modelfile
ollama serve finllm

# 或使用 Docker 部署
docker-compose -f docker/docker-compose.yaml up -d

# 測試 API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finllm",
    "messages": [{"role": "user", "content": "什麼是本益比？"}]
  }'
```

### Step 7: 持續優化（Day 23-30）
```bash
# 收集用戶反饋
# 通過 API 記錄用戶對模型回答的評分

# 準備 DPO 數據（偏好數據）
# 將用戶偏好的回答和不偏好的回答配對

# DPO 對齊訓練
python scripts/train.py \
    --method dpo \
    --config config/training/dpo_config.yaml \
    --data data/processed/dpo_data.jsonl \
    --output outputs/models/finllm-dpo-v1

# A/B 測試
python scripts/ab_test.py \
    --model-a outputs/models/finllm-lora-v1 \
    --model-b outputs/models/finllm-dpo-v1 \
    --test-data data/evaluation/ab_test_set.json
```

### Step 8: 進階優化方向
```
- [ ] 多任務微調（同時訓練問答、分析、生成等任務）
- [ ] 長上下文支持（擴展到 32K tokens）
- [ ] RAG 增強（微調模型 + 外部知識檢索）
- [ ] 多語言支持（中文 + 英文金融文本）
- [ ] 工具使用能力（調用計算器、數據查詢等）
- [ ] 安全護欄（防止生成有害投資建議）
```

---

> 💡 **小提示**：
> 1. **數據為王**：微調的效果 80% 取決於數據質量。花足夠多的時間在數據清洗和格式設計上。
> 2. **從小開始**：先用小數據集（1000 條）快速實驗，確認方向正確後再擴大規模。
> 3. **持續評估**：每個階段都要對比評估，避免「訓練了但沒進步」的情況。
> 4. **注意過擬合**：金融數據噪聲大，LoRA rank 不要設太高，訓練 epoch 不要太多。
> 5. **合規意識**：微調後的模型不應給出具體投資建議，需加上免責聲明。
