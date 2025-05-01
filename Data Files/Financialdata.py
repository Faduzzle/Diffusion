import os
import yfinance as yf
import pandas as pd
import numpy as np

SAVE_DIR = "Data Files/Finance"
os.makedirs(SAVE_DIR, exist_ok=True)

IV_TICKERS = ["^VIX", "^VXN", "GVZ"]
TICKERS = {
    "macro_rates": ["^TNX", "^IRX", "TLT", "IEF", "SHY"],
    "broad_indices": ["^GSPC", "^IXIC", "^DJI", "SPY", "QQQ"],
    "volatility": IV_TICKERS,
    "sector_etfs": ["XLF", "XLK", "XLE", "XLV", "XLI"],
    "megacap_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
}

all_tickers = [ticker for group in TICKERS.values() for ticker in group]

start_date = "2023-01-01"
end_date = "2024-01-01"
interval = "1h"
rolling_window = 20

for ticker in all_tickers:
    try:
        print(f"📥 Downloading {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

        if df.empty:
            print(f"⚠️ No data for {ticker}")
            continue

        df = df[["Adj Close"]].copy()
        df.reset_index(inplace=True)
        df.rename(columns={"Adj Close": "Value"}, inplace=True)

        safe_name = ticker.replace("^", "")

        # === 1. Log-Transformed Price ===
        df_log = df.copy()
        df_log["Value"] = np.log(df_log["Value"] + 1e-8)
        df_log.to_csv(os.path.join(SAVE_DIR, f"{safe_name}.csv"), index=False)

        # === 2. Log Returns ===
        df_logret = df_log.copy()
        df_logret["Value"] = df_logret["Value"].diff()
        df_logret.dropna(inplace=True)
        df_logret.to_csv(os.path.join(SAVE_DIR, f"{safe_name}_logret.csv"), index=False)

        # === 3. Historical Volatility (rolling std of log returns) ===
        df_vol = df_logret.copy()
        df_vol["Value"] = df_vol["Value"].rolling(window=rolling_window).std()
        df_vol.dropna(inplace=True)
        df_vol.to_csv(os.path.join(SAVE_DIR, f"{safe_name}_vol.csv"), index=False)

        # === 4. IV Index Raw Copy ===
        if ticker in IV_TICKERS:
            df_log.to_csv(os.path.join(SAVE_DIR, f"{safe_name}_iv.csv"), index=False)

        print(f"✅ Done with {ticker}")

    except Exception as e:
        print(f"❌ Failed {ticker}: {e}")
