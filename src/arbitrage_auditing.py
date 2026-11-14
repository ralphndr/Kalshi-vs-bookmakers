#!/usr/bin/env python
# coding: utf-8

# La Liga: audit of 2 fee-adjusted NO-bundle candidates
# 
# Read-only diagnostic of the 2 historical apparent arbitrage obs. Uses the already-downloaded La Liga candle file and the full-analysis hourly output. Hourly candles aren't synch order-book snapshots.* Primary calc uses the notebook's cent-rounded 7% taker fee, while the alternative centicent rounded calculation is a sensitivity check rather than proof of the historical rule.

# In[1]:


# import libraries
from pathlib import Path
import pandas as pd
import numpy as np
PROJECT = (Path.cwd().parent if Path.cwd().name == "src" else Path.cwd())  # project folder (run from it or from src)
RESULT = PROJECT / "results/full/laliga/hourly_arbitrage.csv"
CANDLES = PROJECT / "data/laliga/laliga_full_candlesticks.csv"
assert RESULT.is_file(), f"Missing {RESULT}"
assert CANDLES.is_file(), f"Missing {CANDLES}"
h = pd.read_csv(RESULT, parse_dates=["timestamp"])
c = pd.read_csv(CANDLES, parse_dates=["timestamp"])
print("Hourly result columns:", h.columns.tolist())
print("Raw candle columns:", c.columns.tolist())


# Identify the candidates + recover each contract's bid and ask
# 
# The reported timestamp identifies an hourly candle, not the precise time each quote was executable.

# In[2]:


hits = h.loc[h[["profit_buy", "profit_sell"]].max(axis=1) > 0].copy()
print("Candidate hours:", len(hits), "| Distinct matches:", hits.event_ticker.nunique())
display(hits[["event_ticker", "timestamp", "sum_bid", "sum_ask", "fee_sell", "profit_sell", "profit_sell_nofee"]])
assert len(hits) == 2, "Unexpected candidate count—investigate rather than assume exactly two."
raw = c.merge(hits[["event_ticker", "timestamp"]], on=["event_ticker", "timestamp"], how="inner")
raw["no_ask_implied"] = 1 - raw["yes_bid"]
raw["no_bid_implied"] = 1 - raw["yes_ask"]
raw["fee_7pct_cent"] = np.ceil((0.07 * raw.no_ask_implied * (1-raw.no_ask_implied)) * 100 - 1e-10) / 100
raw["fee_7pct_centicent"] = np.ceil((0.07 * raw.no_ask_implied * (1-raw.no_ask_implied)) * 10000 - 1e-10) / 10000
raw = raw.sort_values(["event_ticker", "timestamp", "ticker"])
cols = ["event_ticker","timestamp","ticker","yes_bid","yes_ask","no_ask_implied", "fee_7pct_cent", "fee_7pct_centicent", "volume"]
display(raw[cols])
assert raw.groupby(["event_ticker","timestamp"]).ticker.nunique().eq(3).all(), "Not all 3 legs present"


# Reconcile the bundle's model profit
# 
# With exactly one YES winner, buying all three NO claims pays $2. We compare raw summed prices with original hourly model's `profit_sell_nofee`, which already discounts the settlement payout. All profits are per one-contract-per-outcome bundle. The 7% centicent model below is basically only a rounding sensitivity.

# In[3]:


g=raw.groupby(["event_ticker","timestamp"],as_index=False).agg(
  no_cost=("no_ask_implied","sum"),
  fee_7pct_cent=("fee_7pct_cent","sum"),
  fee_7pct_centicent=("fee_7pct_centicent","sum"),
  sum_yes_bid=("yes_bid","sum"),
  legs=("ticker","nunique"),
)
out=g.merge(hits[["event_ticker","timestamp","profit_sell_nofee","profit_sell","fee_sell"]], on=["event_ticker","timestamp"])
out["gross_undiscounted"] = 2-out.no_cost
out["profit_cent_undiscounted"] = 2-out.no_cost-out.fee_7pct_cent
out["profit_cent_discounted"] = out.profit_sell_nofee-out.fee_7pct_cent
out["profit_centicent_discounted"] = out.profit_sell_nofee-out.fee_7pct_centicent
out["fee_matches_original"] = np.isclose(out.fee_7pct_cent, out.fee_sell)
display(out.round(5))
print("Cent-rounded profits reconcile:", np.allclose(out.profit_cent_discounted, out.profit_sell, atol=0.00011))
DEST=PROJECT/"results/final_checks"
DEST.mkdir(parents=True,exist_ok=True)
raw[cols].to_csv(DEST/"laliga_candidate_individual_legs.csv",index=False)
out.to_csv(DEST/"laliga_candidate_fee_reconciliation.csv",index=False)
print("Saved individual leg and fee audit CSVs to", DEST.resolve())


# key takes for report
# 
# - A match-hour != a confirmed executable trade. -> need synchronized historical level-2 order-book data, timestamps and displayed size for all 3 legs
# - The candle volume is trade activity over the period, not available depth at the reconstructed NO price.
# - Feb 2026 published general fee model uses a coefficient of 0.07 and rounds to cents. Series-specific historical applicability is still unverified; July 2026 fee schedule cannot be retroactively applied w/o evidence.
# - A positive modeled profit != a confirmed historical arbitrage or evidence of a scalable strategy.
# 
