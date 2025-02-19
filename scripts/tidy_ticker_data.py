import yfinance as yf
import pandas as pd

# XWMS  for ticker for Xtrackers MSCI World Momentum ESG UCITS ETF

# source: https://www.tidy-finance.org/python/introduction-to-tidy-finance.html

# The adjusted prices: are corrected for anything that might affect the stock price after the market closes, e.g., stock splits and dividends. 
# These actions affect the quoted prices but have no direct impact on the 
# investors who hold the stock. Therefore, we often rely on adjusted prices when it comes to analyzing the returns
# an investor would have earned by holding the stock continuously.

# Stock market: start="2015-01-01",   end="2024-12-31"
# Trends   1/1/2016	....  12/31/2024




tickers = pd.read_csv("../data_raw/seo_tickers_gapsdotcom.csv")["ticker"].to_list()
tickers = [ticker[1:] for ticker in tickers]
tickers.extend(["^GSPC", "^IXIC", "AAPL"])



prices_daily = (yf.download(
    tickers=tickers, 
    start="2015-01-01", 
    end="2024-12-31", 
    progress=False,
    auto_adjust=False,
    multi_level_index=False
  ))

prices_daily = (prices_daily
  .stack()
  .reset_index(level=1, drop=False)
  .reset_index()
  .rename(columns={
    "Date": "date",
    "Ticker": "ticker",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted",
    "Volume": "volume"}
  )
)


prices_daily.to_csv("../data_proc/prices_daily.csv", index=False)

prices_daily.head().round(3)

############  for single ticker ##################

