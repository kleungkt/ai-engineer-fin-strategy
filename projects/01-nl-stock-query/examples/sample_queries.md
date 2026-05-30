# Sample Queries & Expected Outputs

Below are example natural-language queries you can type into the screener and what you should expect.

---

## 1. RSI Oversold

```
Query > RSI below 30
```

**Parsed intent:** name=RSI, comparison=below, value=30

**Expected:** Stocks whose 14-period RSI is currently under 30 (oversold territory).
In the demo data this typically matches a handful of the 20 stocks depending on the random seed.

```
✅  MATCHED STOCKS:
  ★  INTC
       RSI(14) below 30: RSI(14)=24.37 < 30 → ✓
```

---

## 2. MACD Golden Cross

```
Query > MACD golden cross
```

**Parsed intent:** name=MACD, comparison=crossover

**Expected:** Stocks where the MACD line just crossed above the signal line (bullish momentum shift). This is relatively rare in random data, so 0–3 matches are normal.

```
✅  MATCHED STOCKS:
  ★  NVDA
       MACD crossover None: MACD=0.8432, Signal=0.7901 (crossover ✓)
```

---

## 3. Price Above Moving Average

```
Query > SMA 20 above
```

**Parsed intent:** name=SMA/MA, period=20, comparison=above

**Expected:** Stocks whose current close is above their 20-day simple moving average (uptrend).

```
✅  MATCHED STOCKS:
  ★  AAPL
       MA above None: Close=152.10 > SMA(20)=148.32 → ✓
  ★  MSFT
       MA above None: Close=318.90 > SMA(20)=310.45 → ✓
```

---

## 4. High-Price Stocks

```
Query > price above 100
```

**Parsed intent:** name=PRICE, comparison=above, value=100

**Expected:** Stocks trading above $100. Number of matches depends on sample data.

---

## 5. RSI Overbought

```
Query > RSI above 70
```

**Parsed intent:** name=RSI, comparison=above, value=70

**Expected:** Stocks in overbought territory. Useful for identifying potential pullbacks.

---

## 6. MACD Death Cross

```
Query > MACD death cross
```

**Parsed intent:** name=MACD, comparison=crossunder

**Expected:** Stocks where the MACD line just crossed below the signal line (bearish).

---

## 7. Price Below Round Number

```
Query > price below 50
```

**Parsed intent:** name=PRICE, comparison=below, value=50

**Expected:** All stocks with close < $50.

---

## 8. Bollinger Band Breakout

```
Query > Bollinger above
```

**Parsed intent:** name=BOLL, comparison=above

**Expected:** Stocks whose close is above the upper Bollinger Band (potential breakout).

---

## 9. KDJ Golden Cross

```
Query > KDJ golden cross
```

**Parsed intent:** name=KDJ, comparison=crossover

**Expected:** Stocks where the K line just crossed above the D line.

---

## Tips

- Phrasing is flexible: "RSI under 30", "RSI less than 30", "RSI < 30" all work the same way.
- MACD conditions: "MACD golden cross", "MACD crossover", "MACD bullish" → crossover.
- MA conditions: "SMA 20 above", "close above MA 50" → close vs moving average.
- Try combining with `--demo` flag to auto-run several queries: `python main.py --demo`
- The parser uses OpenAI function calling, so you need `OPENAI_API_KEY` set in your environment.
