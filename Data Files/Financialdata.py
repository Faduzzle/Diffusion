import os
import time
from pathlib import Path

import requests
import pandas as pd
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY        = os.getenv("ALPHA_VANTAGE_API_KEY", "HS8QXJX3FRR0FNK8")
SAVE_DIR       = Path("Data Files/Finance")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IV_TICKERS     = ["VIX", "VXN", "GVZ"]  # no '^'
TICKERS        = {
    "macro_rates":   ["TNX", "IRX", "TLT", "IEF", "SHY"],
    "broad_indices": ["GSPC", "IXIC", "DJI", "SPY", "QQQ"],
    "volatility":    IV_TICKERS,
    "sector_etfs":   ["XLF", "XLK", "XLE", "XLV", "XLI"],
    "megacap_tech":  ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
}

INTERVAL       = "60min"    # 1min,5min,15min,30min,60min
OUTPUTSIZE     = "full"     # "compact" or "full"
ROLLING_WINDOW = 20
CALL_DELAY     = 12.0       # seconds between calls (5 calls/minute limit)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def fetch_intraday_av(symbol: str):
    url    = "https://www.alphavantage.co/query"
    params = {
        "function":   "TIME_SERIES_INTRADAY",
        "symbol":     symbol,
        "interval":   INTERVAL,
        "outputsize": OUTPUTSIZE,
        "apikey":     API_KEY
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    resp = r.json()

    # detect invalid-symbol error
    if "Error Message" in resp:
        raise ValueError(f"Invalid API call for {symbol}")

    ts_key = f"Time Series ({INTERVAL})"
    raw    = resp.get(ts_key)
    if not raw:
        raise ValueError(f"No time series data for {symbol}")

    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={"4. close": "Value"})[["Value"]]
    df = df.reset_index().rename(columns={"index": "Date"})
    # convert from string to float
    df["Value"] = df["Value"].astype(float)
    return df

def save_version(df, out_dir: Path, name: str, suffix: str):
    out = out_dir / f"{name}{suffix}.csv"
    df.to_csv(out, index=False)

def process_ticker_av(ticker: str, out_dir: Path):
    name = ticker.lstrip("^")
    try:
        df = fetch_intraday_av(name)
    except ValueError as ve:
        print(f"⚠️ Skipping {ticker}: {ve}")
        return
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return

    time.sleep(CALL_DELAY)

    # === transforms ===
    df_log = df.copy()
    df_log["Value"] = np.log(df_log["Value"] + 1e-8)

    df_logret = df_log.copy()
    df_logret["Value"] = df_logret["Value"].diff().dropna()

    df_vol = df_logret.copy()
    df_vol["Value"] = df_vol["Value"].rolling(window=ROLLING_WINDOW).std().dropna()

    # === save ===
    save_version(df_log,    out_dir, name, "")
    save_version(df_logret, out_dir, name, "_logret")
    save_version(df_vol,    out_dir, name, "_vol")

    # Raw IV copy if requested
    if ticker in IV_TICKERS:
        save_version(df, out_dir, name, "_iv")

    print(f"✅ Done with {ticker}")

def main():
    for category, symbols in TICKERS.items():
        cat_dir = SAVE_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n⬇️ Processing category: {category}")
        for sym in symbols:
            process_ticker_av(sym, cat_dir)

if __name__ == "__main__":
    main()
