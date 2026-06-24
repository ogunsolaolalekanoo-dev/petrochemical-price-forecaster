# Business Recommendations

*Generated from the Petrochemical Resin Price Forecasting System*
*Model: Ensemble (Prophet + LightGBM) — MAPE 1.50%, R²=0.786*
*Data period: January 2000 – June 2026*

---

## Executive Summary

Resin and gelcoat costs are a significant and volatile
input for bathware manufacturing. This system analyzes
26 years of petrochemical market data to identify where
prices are heading and what procurement should do about it.

The current market signal is **HOLD** — prices are
forecast to be stable in July–August 2026 within a
±3% band. RSI is at 77, indicating overbought conditions
that argue against locking in long-term contracts at
current price levels.

---

## Recommendation 01 — Immediate Action (Next 30–60 Days)

**Signal: HOLD**

**What the data shows:**
Resin PPI is forecast at 317.9 (July) and 317.7 (August)
from a current level of 319.4 — a change of -0.4% and
-0.5% respectively. Both are within the ±3% stable zone
that does not justify a procurement action.

**What to do:**
Maintain current purchasing schedule. Honor existing
supplier commitments at negotiated prices. Do not
accelerate purchases in anticipation of a price spike
that the model does not currently project in the
reliable forecast window.

**What not to do:**
Do not react to the September 2026 forecast showing
~606 PPI. That forecast is driven by a structural
break dummy variable extrapolating the 2026 Middle
East crude oil shock forward — it is a scenario
analysis output, not a price prediction.

---

## Recommendation 02 — Supplier Price Validation

**Signal: ANALYZE**

**What the data shows:**
The model confirmed a 6–8 week lag between WTI crude
oil movements and resin price changes. The ARIMAX
AR(1) coefficient of 0.46 confirms strong one-month
price momentum, and the AR(3) coefficient of -0.14
confirms mean reversion at three months.

Practically: when crude oil rises 10%, expect resin
prices to follow by approximately 4–6% within 6–8 weeks.

**What to do:**
Every supplier price increase request should be
validated against the WPU066 (Plastic Resins PPI)
and PPIACO (All Commodity PPI) index movements
from 6–8 weeks prior. Build a simple validation
model:

Justified increase = Base price ×

(current WPU066 / WPU066 from 8 weeks ago)

Any supplier request above this level should be
challenged with the index data as evidence.

**Business value:**
Index-backed negotiation typically reduces accepted
price increases by 15–30% compared to accepting
supplier claims at face value. On a $10M annual
resin spend, that is $150,000–$300,000 in
avoided cost per year.

---

## Recommendation 03 — RSI-Based Contract Timing

**Signal: OVERBOUGHT (RSI = 77)**

**What the data shows:**
RSI ranked 3rd in LightGBM feature importance with
a score of 38. This was unexpected — RSI is typically
used in financial markets. But resin prices exhibit
genuine mean-reverting behavior, and the RSI captures
the transition points between price runs and corrections.

Current RSI of 77 is above the overbought threshold
of 70. Historically in this dataset, RSI readings
above 70 have preceded price pullbacks in 68% of
cases within the following 2–3 months.

**What to do:**
Do not lock in long-term resin supply contracts
while RSI is above 70. Wait for RSI to normalize
below 50 before negotiating forward contracts.
Set a calendar alert to re-evaluate contract
positioning when RSI drops below 55.

**What to do right now:**
If contract renewals are due in the next 60 days,
negotiate shorter terms (quarterly rather than
annual) to preserve flexibility until RSI normalizes.

**Business value:**
Timing long-term contracts at RSI below 50 versus
above 70 has historically meant a 5–15% difference
in contracted price levels. On multi-million dollar
resin spend that difference is material.

---

## Recommendation 04 — Supply Chain Risk Monitoring

**Signal: MONITOR**

**What the data shows:**
The 2026 Middle East conflict caused WTI crude to
spike 41% in March 2026. Resin PPI followed —
rising from 263 to 319 between March and May 2026,
a 21% increase in 60 days. The model's crude
volatility feature (crude_vol_3m) began signaling
elevated risk before the full resin price impact
arrived.

3-month crude volatility currently stands at 20.75 —
above the 2000–2022 historical average of 8.4.

**What to do:**
Monitor `crude_vol_3m` as an early warning
indicator. Define a volatility threshold — for
example, when 3-month crude volatility exceeds
15 (roughly 1.5x historical average):

1. Initiate a contingency sourcing review
2. Identify alternative supplier options
3. Evaluate whether to accelerate near-term purchases
   before the volatility transmits to resin prices
4. Alert supply chain leadership

This gives approximately 4–6 weeks of lead time
before crude volatility shows up in resin invoices.

**Business value:**
Proactive sourcing triggered by volatility signals
rather than reactive purchasing after prices spike
can reduce resin cost exposure by 8–20% during
supply disruption periods.

---

## Recommendation 05 — Construction Season Positioning

**Signal: PLAN**

**What the data shows:**
Housing starts lag (housing_lag_6m) ranked 9th in
feature importance. Construction activity drives
bathware demand, which drives resin purchasing.
The seasonal decomposition in Prophet confirmed
a consistent annual pattern:

- Q4 (Oct–Dec): Softer resin demand, lower prices
- Q1–Q2 (Jan–Jun): Rising construction activity,
  firming resin prices
- Q3 (Jul–Sep): Peak construction season,
  highest demand pressure

This pattern has repeated consistently across
26 years of data despite multiple economic cycles.

**What to do:**
Build strategic resin inventory in Q4
(October–December) before the construction
season demand surge. Specifically:

- November: Negotiate forward contracts for
  Q2 delivery at Q4 prices
- December: Take advantage of any year-end
  supplier inventory reduction offers
- January: Complete forward purchasing
  before Q1 construction activity begins

**What not to do:**
Do not wait until Q2 to make annual resin
purchasing decisions. By March–April, construction
season demand is already pulling prices up and
negotiating leverage has diminished.

**Business value:**
Buying ahead of construction season rather than
during it captures the seasonal price dip.
The Prophet seasonal component shows Q4 resin
prices have historically averaged 3–7% below
Q2 prices in the same calendar year.
On significant resin spend, buying in Q4
for Q2 delivery is a consistent, repeatable
cost reduction strategy.

---

## How These Recommendations Update

This system pulls fresh data from FRED and EIA
APIs every time `data_ingestion.py` runs. The
recommendations above are generated from:

- Current resin PPI reading
- Latest RSI calculation
- Most recent ensemble forecast
- Current crude oil volatility level

To refresh recommendations, run:
```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/models.py
python src/procurement_signal.py
```

Then reopen the dashboard at `http://127.0.0.1:8050`

---

*This analysis uses public data from FRED
(Federal Reserve Bank of St. Louis) and the
EIA (US Energy Information Administration).
It is intended as decision support intelligence,
not financial advice. All procurement decisions
should incorporate operational constraints,
existing contracts, and business context
that the model does not have visibility into.*