"""Financial data preparation for LLM fine-tuning."""

import json
import random
from pathlib import Path

from pydantic import BaseModel, field_validator


class FinDataSample(BaseModel):
    """A single financial training data sample."""

    instruction: str
    input: str
    output: str
    category: str  # sentiment, qa, summary, analysis

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {"sentiment", "qa", "summary", "analysis"}
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}, got '{v}'")
        return v


def load_raw_data(path: str) -> list[FinDataSample]:
    """Load financial training data from JSON or JSONL file.

    Args:
        path: Path to a .json (array) or .jsonl file.

    Returns:
        List of FinDataSample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    samples: list[FinDataSample] = []

    if p.suffix == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(FinDataSample(**json.loads(line)))
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            samples.append(FinDataSample(**item))
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .json or .jsonl")

    return samples


# ---------------------------------------------------------------------------
# Synthetic data templates
# ---------------------------------------------------------------------------

_SENTIMENT_NEWS = [
    ("Apple reported record quarterly earnings beating analyst estimates by 15%", "bullish"),
    ("Tesla stock plunges 12% after disappointing delivery numbers", "bearish"),
    ("Federal Reserve holds interest rates steady amid mixed economic signals", "neutral"),
    ("Amazon announces massive layoffs of 18,000 employees to cut costs", "bearish"),
    ("Nvidia revenue surges 200% driven by AI chip demand", "bullish"),
    ("Oil prices stabilize as OPEC agrees to maintain production levels", "neutral"),
    ("Microsoft cloud business grows 29% year-over-year", "bullish"),
    ("Banking sector faces uncertainty as SVB collapses", "bearish"),
    ("Gold prices remain flat as dollar strengthens slightly", "neutral"),
    ("Google parent Alphabet announces $70B share buyback program", "bullish"),
    ("Retail sales decline for third consecutive month", "bearish"),
    ("Manufacturing PMI holds steady at 50.2 indicating flat growth", "neutral"),
    ("Bitcoin rallies past $50,000 as institutional adoption grows", "bullish"),
    ("Chinese real estate giant Evergrande files for bankruptcy", "bearish"),
    ("European Central Bank maintains current monetary policy stance", "neutral"),
]

_QA_PAIRS = [
    ("What is a P/E ratio?", "The Price-to-Earnings (P/E) ratio is a valuation metric that compares a company's stock price to its earnings per share (EPS). It indicates how much investors are willing to pay per dollar of earnings. A high P/E may suggest the stock is overvalued or that investors expect high growth rates."),
    ("Explain compound interest.", "Compound interest is interest calculated on both the initial principal and the accumulated interest from previous periods. Unlike simple interest, compound interest grows exponentially over time. The formula is A = P(1 + r/n)^(nt), where P is principal, r is annual rate, n is compounding frequency, and t is time in years."),
    ("What is a stop-loss order?", "A stop-loss order is a risk management tool that automatically sells a security when it reaches a specified price. It limits an investor's loss on a position. For example, setting a stop-loss at $90 on a stock purchased at $100 means the position will be sold if the price drops to $90, capping the loss at 10%."),
    ("What are options in finance?", "Options are financial derivatives that give the holder the right, but not the obligation, to buy (call option) or sell (put option) an underlying asset at a predetermined price (strike price) before a specific date (expiration). They are used for hedging, speculation, and income generation."),
    ("Define market capitalization.", "Market capitalization (market cap) is the total market value of a company's outstanding shares of stock. It is calculated by multiplying the current stock price by the total number of outstanding shares. Companies are classified as large-cap (>$10B), mid-cap ($2-10B), or small-cap (<$2B)."),
    ("What is dollar-cost averaging?", "Dollar-cost averaging (DCA) is an investment strategy where a fixed amount is invested at regular intervals regardless of market conditions. This approach reduces the impact of volatility by buying more shares when prices are low and fewer when prices are high, resulting in a lower average cost per share over time."),
    ("Explain the bid-ask spread.", "The bid-ask spread is the difference between the highest price a buyer is willing to pay (bid) and the lowest price a seller is willing to accept (ask). A narrow spread indicates high liquidity and active trading, while a wide spread suggests lower liquidity. Market makers profit from the spread."),
    ("What is a bond yield?", "A bond yield is the return an investor earns on a bond. Current yield is the annual coupon payment divided by the bond price. Yield to maturity (YTM) accounts for all future coupon payments and the face value received at maturity, providing a more comprehensive measure of expected return."),
]

_SUMMARIES = [
    (
        "Q3 2024 Earnings: Revenue $45.2B (+12% YoY), Net Income $11.3B (+18% YoY), EPS $2.45 (est $2.30). Cloud segment revenue grew 28%. Operating margin expanded to 25.1%. Company raised full-year guidance. Strong performance in AI and cloud services drove the beat.",
        "The company delivered strong Q3 2024 results with revenue of $45.2B (up 12% YoY) and net income of $45.2B, beating EPS estimates at $2.45 vs $2.30 expected. Cloud and AI segments were standout performers, with cloud revenue up 28%. Operating margins improved to 25.1%, and management raised full-year guidance reflecting confidence in continued momentum."
    ),
    (
        "Annual Report 2024: Total assets $280B, total debt $95B, cash $42B. Revenue $120B, free cash flow $18B. R&D spending $22B (+15%). New product launches in 3 markets. Employee count grew 8% to 150,000. Dividend increased 10%. Share repurchase program of $10B authorized.",
        "In 2024, the company reported solid financials with $280B in total assets and $120B in revenue generating $18B in free cash flow. Strategic investments included $22B in R&D (up 15%) and expansion into 3 new markets. The balance sheet remains healthy with $42B cash against $95B debt. Shareholder returns were enhanced through a 10% dividend increase and a new $10B buyback program."
    ),
    (
        "Quarterly update: Revenue missed estimates by 3%. Gross margin declined 200bps due to input cost inflation. International segment grew 8% while domestic declined 2%. Management announced cost-cutting initiative targeting $500M in savings. Full-year guidance lowered to $8.50-8.75 EPS from $9.00-9.25.",
        "The company's quarterly results disappointed with revenue missing estimates by 3% and gross margins contracting 200 basis points from input cost pressures. Geographic performance was mixed with international growth of 8% offset by a 2% domestic decline. Management responded with a $500M cost reduction program and lowered full-year EPS guidance from $9.00-9.25 to $8.50-8.75."
    ),
    (
        "FY2024 Results: Revenue $8.7B (+22%), adjusted EBITDA $2.1B (+30%), subscriber base reached 250M (+15M QoQ). ARPU increased to $12.50. Content spend $3.2B. Launched ad-supported tier reaching 40M users. International revenue now 55% of total.",
        "FY2024 was a strong year with revenue growing 22% to $8.7B and adjusted EBITDA up 30% to $2.1B. The subscriber base hit 250M with 15M net additions in Q4 alone, while ARPU rose to $12.50. The ad-supported tier, launched this year, already has 40M users. International markets now contribute 55% of total revenue, underscoring successful global expansion."
    ),
]

_ANALYSES = [
    (
        "Analyze a momentum trading strategy that uses 50-day and 200-day moving average crossovers.",
        "A moving average crossover strategy uses the 50-day MA crossing above the 200-day MA (golden cross) as a buy signal and crossing below (death cross) as a sell signal. This trend-following approach works well in trending markets but generates false signals in sideways/choppy conditions. Historical backtests on the S&P 500 show annualized returns of ~8-10% with lower drawdowns than buy-and-hold. Key considerations: (1) Signal lag means entries/exits are late, (2) Works best on liquid large-cap assets, (3) Adding volume confirmation and RSI filters can reduce whipsaws, (4) Transaction costs and taxes from frequent trading reduce net returns, (5) Best combined with fundamental filters for stock selection."
    ),
    (
        "Evaluate a covered call strategy for income generation.",
        "A covered call strategy involves holding a long stock position while selling call options against it, typically at strikes 5-10% out-of-the-money with 30-45 day expirations. This generates premium income of 1-3% monthly but caps upside potential. It works best in flat-to-moderately-bullish markets. Key metrics: the strategy typically yields 8-12% annually versus 10-12% for pure stock ownership, but with 20-30% lower volatility. Risks include: (1) Limited upside if stock rallies significantly, (2) Full downside exposure in market crashes, (3) Early assignment risk near ex-dividend dates, (4) Tax implications of covered calls vs. holding stock. Best suited for mature, lower-volatility stocks in taxable accounts."
    ),
    (
        "Analyze the effectiveness of mean reversion strategies in forex markets.",
        "Mean reversion strategies in forex aim to profit from prices returning to their historical average after extreme deviations. Using Bollinger Bands (2-sigma) and RSI oversold/overbought signals, these strategies can generate Sharpe ratios of 0.8-1.2. They work best on major currency pairs (EUR/USD, GBP/USD) during range-bound periods. Key findings: (1) Success rate is ~60-65% for well-calibrated signals, (2) Risk-reward is typically 1:1.5 to 1:2, (3) Central bank interventions and macro events can break mean reversion assumptions, (4) Position sizing should account for volatility (ATR-based), (5) Stop-losses at 2-3x ATR are essential to manage tail risk. Best combined with carry trade signals for directional bias."
    ),
]


def generate_sample_dataset(n: int = 100) -> list[FinDataSample]:
    """Generate synthetic financial training data.

    Creates a balanced dataset across sentiment, QA, summary, and analysis
    categories.  When *n* exceeds the available templates, entries are
    reused with minor variation.

    Args:
        n: Total number of samples to generate.

    Returns:
        List of FinDataSample objects.
    """
    samples: list[FinDataSample] = []
    per_category = max(1, n // 4)
    remainder = n - per_category * 4

    categories_and_generators: list[tuple[str, list, str]] = []

    # Sentiment
    for news, label in _SENTIMENT_NEWS:
        samples.append(FinDataSample(
            instruction="Analyze the sentiment of the following financial news headline.",
            input=news,
            output=f"The sentiment is {label}. {news} — this news is {label} for the stock/market.",
            category="sentiment",
        ))
        if len(samples) >= per_category:
            break
    # Fill remaining sentiment slots if needed
    while sum(1 for s in samples if s.category == "sentiment") < per_category:
        news, label = random.choice(_SENTIMENT_NEWS)
        samples.append(FinDataSample(
            instruction="Analyze the sentiment of the following financial news headline.",
            input=news,
            output=f"The sentiment is {label}.",
            category="sentiment",
        ))

    # QA
    for question, answer in _QA_PAIRS:
        samples.append(FinDataSample(
            instruction="Answer the following financial question accurately and concisely.",
            input=question,
            output=answer,
            category="qa",
        ))
        if sum(1 for s in samples if s.category == "qa") >= per_category:
            break
    while sum(1 for s in samples if s.category == "qa") < per_category:
        question, answer = random.choice(_QA_PAIRS)
        samples.append(FinDataSample(
            instruction="Answer the following financial question accurately and concisely.",
            input=question,
            output=answer,
            category="qa",
        ))

    # Summary
    for raw, summary in _SUMMARIES:
        samples.append(FinDataSample(
            instruction="Summarize the following financial report concisely.",
            input=raw,
            output=summary,
            category="summary",
        ))
        if sum(1 for s in samples if s.category == "summary") >= per_category:
            break
    while sum(1 for s in samples if s.category == "summary") < per_category:
        raw, summary = random.choice(_SUMMARIES)
        samples.append(FinDataSample(
            instruction="Summarize the following financial report concisely.",
            input=raw,
            output=summary,
            category="summary",
        ))

    # Analysis
    for prompt, analysis in _ANALYSES:
        samples.append(FinDataSample(
            instruction="Provide a detailed financial analysis.",
            input=prompt,
            output=analysis,
            category="analysis",
        ))
        if sum(1 for s in samples if s.category == "analysis") >= per_category:
            break
    while sum(1 for s in samples if s.category == "analysis") < per_category:
        prompt, analysis = random.choice(_ANALYSES)
        samples.append(FinDataSample(
            instruction="Provide a detailed financial analysis.",
            input=prompt,
            output=analysis,
            category="analysis",
        ))

    # Handle remainder by adding from random categories
    while len(samples) < n:
        category = random.choice(["sentiment", "qa", "summary", "analysis"])
        if category == "sentiment":
            news, label = random.choice(_SENTIMENT_NEWS)
            samples.append(FinDataSample(
                instruction="Analyze the sentiment of the following financial news headline.",
                input=news,
                output=f"The sentiment is {label}.",
                category="sentiment",
            ))
        elif category == "qa":
            question, answer = random.choice(_QA_PAIRS)
            samples.append(FinDataSample(
                instruction="Answer the following financial question accurately and concisely.",
                input=question,
                output=answer,
                category="qa",
            ))
        elif category == "summary":
            raw, summary = random.choice(_SUMMARIES)
            samples.append(FinDataSample(
                instruction="Summarize the following financial report concisely.",
                input=raw,
                output=summary,
                category="summary",
            ))
        else:
            prompt, analysis = random.choice(_ANALYSES)
            samples.append(FinDataSample(
                instruction="Provide a detailed financial analysis.",
                input=prompt,
                output=analysis,
                category="analysis",
            ))

    return samples[:n]


def format_for_training(
    samples: list[FinDataSample], format: str = "alpaca"
) -> list[dict]:
    """Format samples for a specific training framework.

    Args:
        samples: List of FinDataSample objects.
        format: Target format — ``'alpaca'`` or ``'chatml'``.

    Returns:
        List of dictionaries in the requested format.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    if format not in ("alpaca", "chatml"):
        raise ValueError(f"Unsupported format '{format}'. Use 'alpaca' or 'chatml'")

    formatted: list[dict] = []

    for s in samples:
        if format == "alpaca":
            formatted.append({
                "instruction": s.instruction,
                "input": s.input,
                "output": s.output,
            })
        else:  # chatml
            formatted.append({
                "messages": [
                    {"role": "system", "content": s.instruction},
                    {"role": "user", "content": s.input},
                    {"role": "assistant", "content": s.output},
                ]
            })

    return formatted


def split_dataset(
    samples: list, test_ratio: float = 0.1
) -> tuple[list, list]:
    """Split a dataset into train and test sets.

    Args:
        samples: List of samples (FinDataSample or dict).
        test_ratio: Fraction of data reserved for testing (0-1).

    Returns:
        (train_samples, test_samples) tuple.

    Raises:
        ValueError: If test_ratio is not between 0 and 1.
    """
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    shuffled = list(samples)
    random.shuffle(shuffled)

    split_idx = max(1, int(len(shuffled) * (1 - test_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_dataset(samples: list, path: str) -> None:
    """Save samples to a JSONL file.

    Args:
        samples: List of FinDataSample or dict objects.
        path: Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        for sample in samples:
            if isinstance(sample, FinDataSample):
                f.write(json.dumps(sample.model_dump(), ensure_ascii=False) + "\n")
            elif isinstance(sample, dict):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
