

###

tickers = pd.read_csv("../data_raw/seo_companies.csv")["ticker"].to_list()
tickers = [ticker[1:] for ticker in tickers]
tickers.extend(["^GSPC", "^IXIC", "AAPL"])




import yfinance as yf
import pandas as pd

# def get_market_cap_df(tickers):
#     data = {"Ticker": [], "Market_Cap": []}
#     
#     for ticker in tickers:
#         stock = yf.Ticker(ticker)
#         market_cap = stock.info.get("marketCap", "N/A")
#         data["Ticker"].append(ticker)
#         data["Market_Cap"].append(market_cap)
#         data["Market_Cap"] = pd.to_numeric(data["Market_Cap"], errors="coerce")
#         
#     data=data.rename(columns={
#     "Market_Cap": "market_cap",
#     "Ticker": "ticker"})
#     
#     return pd.DataFrame(data)


def get_market_cap_df(tickers):
    data = {"Ticker": [], "Market_Cap": []}
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap", "N/A")
        data["Ticker"].append(ticker)
        data["Market_Cap"].append(market_cap)  # Don't modify in the loop

    # Create DataFrame first
    df = pd.DataFrame(data)

    # Convert Market_Cap column to numeric after DataFrame creation
    df["Market_Cap"] = pd.to_numeric(df["Market_Cap"], errors="coerce")

    # Rename columns
    df = df.rename(columns={"Market_Cap": "market_cap", "Ticker": "ticker"})
    
    return df




# Example usage
market_cap_df = get_market_cap_df(tickers)

#market_cap_df["Market_Cap"] = pd.to_numeric(market_cap_df["Market_Cap"], errors="coerce")


market_cap_df.to_csv("market_cap_df_all.csv", index= False)


print(market_cap_df)


###

