#!/usr/bin/env python
# coding: utf-8

# Soccer prediction markets: data collection
# 
# Collects historical Kalshi match-winner quotes for the assigned leagues. Run cells top to bottom. Each successful contract is checkpointed right away, so a rerun resumes w/o redownloading it.
# 
# Work allocation: me -> La Liga, Bundesliga, Eredivisie, Liga MX. Partner -> Serie A, Brazil Série A, Ligue 1, Champions League. EPL = common benchmark, already collected.
# 
# Output per league in `data/<tag>/`: contract metadata (`.pkl`), raw candle checkpoint (`.pkl`) and standardized hourly prices (`.csv`). Nothing downloads until the run cell is executed.

# In[1]:


from pathlib import Path
import os, pickle, time
import numpy as np
import pandas as pd
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
OUTPUT_ROOT = (Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()) / "data"
OUTPUT_ROOT.mkdir(exist_ok=True)
LEAGUES = {
    "epl": "KXEPLGAME",
    "laliga": "KXLALIGAGAME",
    "bundesliga": "KXBUNDESLIGAGAME",
    "seriea": "KXSERIEAGAME",
    "ligue1": "KXLIGUE1GAME",
    "ucl": "KXUCLGAME",
    "eredivisie": "KXEREDIVISIEGAME",
    "ligamx": "KXLIGAMXGAME",
    "brasileiro": "KXBRASILEIROGAME",
}

# 1. Pick the leagues
# Change `SELECTED` to 1 league for a trial, or the whole 4-league assignment. Leave `epl` alone unless re-downloading it.

# In[2]:


# change as needed
SELECTED = ["seriea", "brasileiro", "ligue1", "ucl"]
# my assignment: ["laliga", "bundesliga", "eredivisie", "ligamx"]
# partner's: ["seriea", "brasileiro", "ligue1", "ucl"]

PAUSE_SECONDS = 1.5
WINDOW_DAYS = 8
MAX_CONTRACTS_PER_LEAGUE = None  # 6 to test first, None for the full download

assert all(tag in LEAGUES for tag in SELECTED)
print("Downloading:", {tag: LEAGUES[tag] for tag in SELECTED})
print("Output:", OUTPUT_ROOT.resolve())

# 2. API access + reusable helpers
# Handles pagination, current vs archived quote schemas, rate limits and interruption-safe checkpointing.

# In[3]:


def api_get(session, path, params, attempts=9):
    # GET w/ retries on rate limits / server errors / timeouts
    for i in range(attempts):
        try:
            r = session.get(BASE + path, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                try:
                    delay = float(retry_after)
                except (ValueError, TypeError):
                    delay = min(2 ** (i + 1), 90)
                print(f"  HTTP {r.status_code}; retrying in {delay:g}s")
                time.sleep(max(delay, 1))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(min(2 ** (i + 1), 90))
    raise RuntimeError(f"API failed after {attempts} tries: {path}")


def list_markets(session, series, endpoint, pause=1.5):
    # all markets of a series, following the pagination cursor
    result, cursor = [], None
    while True:
        params = {"series_ticker": series, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = api_get(session, "/" + endpoint, params)
        result.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            return result
        time.sleep(pause)


def candle_value(c, key):
    # close price, new schema (_dollars) or old one
    obj = c.get(key) or {}
    v = obj.get("close_dollars")
    if v is None:
        v = obj.get("close")
    return pd.to_numeric(v, errors="coerce")


def candle_scalar(c, key):
    # volume / open interest, new schema (_fp) or old one
    v = c.get(key + "_fp")
    if v is None:
        v = c.get(key)
    return pd.to_numeric(v, errors="coerce")


def atomic_save_pickle(obj, path):
    # write to a temp file then swap -> no half-written cache if interrupted
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        pickle.dump(obj, f)
    os.replace(temp, path)

# In[4]:


def download_league(tag, max_contracts=6):
    series = LEAGUES[tag]
    folder = OUTPUT_ROOT / tag
    folder.mkdir(parents=True, exist_ok=True)
    market_path = folder / f"kalshi_{tag}_markets.pkl"
    cache_path = folder / f"{tag}_full_candlesticks.pkl"
    csv_path = folder / f"{tag}_full_candlesticks.csv"

    session = requests.Session()
    current = list_markets(session, series, "markets", PAUSE_SECONDS)
    historical = list_markets(session, series, "historical/markets", PAUSE_SECONDS)
    archived = {m["ticker"] for m in historical}
    if not current and not historical:
        print(f"{tag}: no contracts returned")
        return {"league": tag, "resolved_matches": 0, "cached": 0, "pending": 0, "errors": 0}

    markets = pd.DataFrame(current + historical).drop_duplicates("ticker")
    markets.to_pickle(market_path)
    # keep only matches w/ 3 settled contracts: 1 yes, 2 no
    finalized = markets[markets.status.eq("finalized")].copy()
    counts = finalized.groupby("event_ticker").agg(
        contracts=("ticker", "size"),
        yes=("result", lambda x: x.eq("yes").sum()),
        no=("result", lambda x: x.eq("no").sum()))
    valid_events = counts.index[(counts.contracts == 3) & (counts.yes == 1) & (counts.no == 2)]
    finalized = finalized[finalized.event_ticker.isin(valid_events)].copy()
    cache = {}
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    pending = finalized[~finalized.ticker.isin(cache)].sort_values("ticker")
    if max_contracts is not None:
        pending = pending.head(max_contracts)
    errors = []
    print(f"{tag}: {len(valid_events)} complete matches, {len(finalized)} contracts, {len(cache)} already cached")
    for i, m in enumerate(pending.to_dict("records"), 1):
        # hourly candles over the WINDOW_DAYS b4 expiration, archived markets have their own endpoint
        expected = pd.to_datetime(m["expected_expiration_time"], utc=True)
        endpoint = (f"/historical/markets/{m['ticker']}/candlesticks" if m["ticker"] in archived
                    else f"/series/{series}/markets/{m['ticker']}/candlesticks")
        params = {"start_ts": int((expected - pd.Timedelta(days=WINDOW_DAYS)).timestamp()),
                  "end_ts": int(expected.timestamp()), "period_interval": 60}
        try:
            data = api_get(session, endpoint, params)
            cache[m["ticker"]] = data.get("candlesticks", [])
            atomic_save_pickle(cache, cache_path)
        except Exception as e:
            errors.append((m["ticker"], str(e)))
            print("FAILED:", m["ticker"], e)
        if i % 25 == 0 or i == len(pending):
            print(f"  {i}/{len(pending)} requested; {len(set(cache) & set(finalized.ticker))}/{len(finalized)} cached")
        time.sleep(PAUSE_SECONDS)

    # flatten the cache into 1 row per contract-hour
    records = []
    for m in finalized.to_dict("records"):
        for c in cache.get(m["ticker"], []):
            records.append({
                "event_ticker": m["event_ticker"], "ticker": m["ticker"], "result": m["result"],
                "scheduled_event_time": m["expected_expiration_time"],
                "timestamp": pd.to_datetime(c["end_period_ts"], unit="s", utc=True),
                "yes_bid": candle_value(c, "yes_bid"), "yes_ask": candle_value(c, "yes_ask"),
                "trade_price": candle_value(c, "price"),
                "volume": candle_scalar(c, "volume"), "open_interest": candle_scalar(c, "open_interest")})
    columns = ["event_ticker", "ticker", "result", "scheduled_event_time", "timestamp",
               "yes_bid", "yes_ask", "trade_price", "volume", "open_interest"]
    pd.DataFrame(records, columns=columns).to_csv(csv_path, index=False)
    remaining = len(set(finalized.ticker) - set(cache))
    if errors:
        (folder / "download_errors.txt").write_text("\n".join(f"{t}: {e}" for t,e in errors))
    print(f"Saved {len(records):,} candles to {csv_path} | remaining {remaining}")
    return {"league": tag, "resolved_matches": len(valid_events), "cached": len(set(cache) & set(finalized.ticker)),
            "pending": remaining, "errors": len(errors), "candles": len(records)}

# 3. Run the selected downloads
# Test run = 6 contracts/league. Once it works, set `MAX_CONTRACTS_PER_LEAGUE = None` in section 1, rerun that cell, then rerun below. Completed tickers are skipped.

# In[5]:


download_summaries = []
for tag in SELECTED:
    result = download_league(tag, max_contracts=MAX_CONTRACTS_PER_LEAGUE)
    download_summaries.append(result)
display(pd.DataFrame(download_summaries))

# 4. Validate saved CSVs
# Flags missing bid/ask prices, crossed books and incomplete downloads. An empty candle response is cached but not automatically treated as usable data.

# In[6]:


checks = []
for tag in SELECTED:
    folder = OUTPUT_ROOT / tag
    csv_path = folder / f"{tag}_full_candlesticks.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    checks.append({
        "league": tag, "hourly_rows": len(df),
        "quoted_contracts": df.ticker.nunique() if len(df) else 0,
        "quoted_matches": df.event_ticker.nunique() if len(df) else 0,
        "missing_bidask_rows": int(df[["yes_bid", "yes_ask"]].isna().any(axis=1).sum()) if len(df) else 0,
        "invalid_bidask_rows": int(((df.yes_bid > df.yes_ask) | (df.yes_bid < 0) | (df.yes_ask > 1)).sum()) if len(df) else 0
    })
display(pd.DataFrame(checks))
