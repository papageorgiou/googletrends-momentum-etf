import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
# Comparison indices (for plotting only; these tickers will be excluded from selection)
comparison_indices = ['^GSPC', '^IXIC', 'AAPL']

# Weighting method options: 'market_cap', 'equal', or 'slope'
weighting_method = 'market_cap'  # change as desired

# Selection method options: 'search_interest' or 'price_trend'
#   - "search_interest" uses historical monthly search interest data,
#   - "price_trend" uses historical daily adjusted prices.
selection_method = 'search_interest'  # change as desired

# Evaluation period (in months): used both as the lookback window for selection and the investment period length.
evaluation_period_months = 6

# Start and end dates for the backtest
start_date_str = "2015-01-01"
end_date_str = "2024-12-31"

# -------------------------------
# Load and Standardize the Datasets
# -------------------------------
df_market = pd.read_csv("../data_raw/market_cap_df_all.csv")
df_prices = pd.read_csv("../data_raw/prices_daily.csv")
df_search = pd.read_csv("../data_raw/monthly_search_interest_data_tickers.csv")

# Rename and standardize columns in search data
df_search.rename(columns={"date": "year_month", "value": "monthly_search_interest"}, inplace=True)
df_prices["date"] = pd.to_datetime(df_prices["date"])
df_search["year_month"] = pd.to_datetime(df_search["year_month"])
df_search["ticker"] = df_search["ticker_name"].str.replace(r"^\$", "", regex=True)

# -------------------------------
# Helper Functions
# -------------------------------
def compute_slope(values):
    """Compute the slope using a linear fit over the given values."""
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope

def get_periods(start_date, end_date, period_months):
    """
    Return a list of (period_start, period_end) tuples representing consecutive periods,
    each of length `period_months` months.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    periods = []
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + pd.DateOffset(months=period_months) - pd.Timedelta(days=1)
        if current_end > end_date:
            current_end = end_date
        periods.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
    return periods

def extend_index(df_index, period_start, period_end, value):
    """
    Extend the momentum index DataFrame over all business days in the period,
    setting each day's value to the given `value`.
    """
    dates = pd.date_range(period_start, period_end, freq="B")
    return pd.concat([df_index, pd.DataFrame({"date": dates, "momentum_index": value})], ignore_index=True)

def cap_weights(weights, cap=0.3):
    """
    Given a dictionary of weights, cap each weight at the specified cap (default 30%),
    and redistribute any excess proportionally among the remaining stocks.
    """
    iteration = 0
    while True:
        new_weights = {}
        total_capped = 0
        non_capped = {}
        for ticker, w in weights.items():
            if w > cap:
                new_weights[ticker] = cap
                total_capped += cap
            else:
                non_capped[ticker] = w
        total_non_capped = sum(non_capped.values())
        remaining = 1 - total_capped
        if total_non_capped == 0:
            break
        factor = remaining / total_non_capped
        updated = False
        for ticker, w in non_capped.items():
            new_w = w * factor
            new_weights[ticker] = new_w
            if new_w > cap:
                updated = True
        iteration += 1
        if not updated or iteration > 10:
            break
        weights = new_weights
    return new_weights

# -------------------------------
# Set Backtest Parameters and Prepare Data
# -------------------------------
start_date = pd.Timestamp(start_date_str)
end_date = pd.Timestamp(end_date_str)

# Exclude the comparison indices from candidate tickers
all_tickers = df_market["ticker"].unique()
candidate_tickers = [ticker for ticker in all_tickers if ticker not in comparison_indices]

# Create the investment periods (each of length evaluation_period_months)
periods = get_periods(start_date, end_date, evaluation_period_months)

# Initialize the momentum index DataFrame and other log containers
df_index = pd.DataFrame(columns=["date", "momentum_index"])
selected_tickers_log = []
previous_index_value = 100  # starting value for the index

# For attribution and weight logging
contrib_dict = {}
semester_weights = []

# -------------------------------
# Backtesting Loop
# -------------------------------
for period_start, period_end in periods:
    # Define the historical period (the lookback window) used for ticker selection.
    # It runs from (period_start - evaluation_period_months) up to the day before period_start.
    historical_start = period_start - pd.DateOffset(months=evaluation_period_months)
    if historical_start < start_date:
        historical_start = start_date
    historical_end = period_start - pd.Timedelta(days=1)
    
    # --- Ticker Selection Based on Historical Data ---
    if selection_method == 'search_interest':
        df_search_hist = df_search[(df_search["year_month"] >= historical_start) &
                                   (df_search["year_month"] <= historical_end)]
        # If no historical search data is available (e.g. very first period), default to all candidates.
        if df_search_hist.empty:
            selected_tickers = candidate_tickers.copy()
        else:
            selected_tickers = [
                ticker for ticker in candidate_tickers
                if compute_slope(
                    df_search_hist[df_search_hist["ticker"] == ticker]["monthly_search_interest"].values
                ) > 0
            ]
    elif selection_method == 'price_trend':
        df_prices_hist = df_prices[(df_prices["date"] >= historical_start) &
                                   (df_prices["date"] <= historical_end)]
        if df_prices_hist.empty:
            selected_tickers = candidate_tickers.copy()
        else:
            selected_tickers = [
                ticker for ticker in candidate_tickers
                if compute_slope(
                    df_prices_hist[df_prices_hist["ticker"] == ticker]["adjusted"].values
                ) > 0
            ]
    else:
        raise ValueError("Unknown selection_method specified. Use 'search_interest' or 'price_trend'.")
    
    # --- Filter to Tickers with Price Data in the Investment Period ---
    df_prices_period = df_prices[(df_prices["date"] >= period_start) &
                                 (df_prices["date"] <= period_end)]
    available_tickers = df_prices_period["ticker"].unique()
    selected_tickers = [t for t in selected_tickers if t in available_tickers]
    selected_tickers_log.append((period_start, period_end, selected_tickers))
    
    # If no tickers are selected, extend the index with the previous value.
    if not selected_tickers:
        df_index = extend_index(df_index, period_start, period_end, previous_index_value)
        continue
    
    # --- Compute Weights Based on the Selected Weighting Method ---
    if weighting_method == 'market_cap':
        df_selected_market = df_market[df_market["ticker"].isin(selected_tickers)].dropna(subset=["market_cap"]).copy()
        total_cap = df_selected_market["market_cap"].sum()
        df_selected_market["weight"] = df_selected_market["market_cap"] / total_cap
        weight_dict = df_selected_market.set_index("ticker")["weight"].to_dict()
    elif weighting_method == 'equal':
        weight_dict = {ticker: 1/len(selected_tickers) for ticker in selected_tickers}
    elif weighting_method == 'slope':
        slopes = {}
        # Use the same historical data as used in selection
        if selection_method == 'price_trend':
            for ticker in selected_tickers:
                slope_val = compute_slope(
                    df_prices_hist[df_prices_hist["ticker"] == ticker]["adjusted"].values
                )
                slopes[ticker] = slope_val
        else:  # search_interest
            for ticker in selected_tickers:
                slope_val = compute_slope(
                    df_search_hist[df_search_hist["ticker"] == ticker]["monthly_search_interest"].values
                )
                slopes[ticker] = slope_val
        total_slope = sum(slopes.values())
        if total_slope == 0:
            weight_dict = {ticker: 1/len(selected_tickers) for ticker in selected_tickers}
        else:
            weight_dict = {ticker: slope_val / total_slope for ticker, slope_val in slopes.items()}
        # Impose a cap of 30% per stock and redistribute any excess.
        weight_dict = cap_weights(weight_dict, cap=0.3)
    else:
        # Default to market cap weighting if an unknown method is specified.
        df_selected_market = df_market[df_market["ticker"].isin(selected_tickers)].dropna(subset=["market_cap"]).copy()
        total_cap = df_selected_market["market_cap"].sum()
        df_selected_market["weight"] = df_selected_market["market_cap"] / total_cap
        weight_dict = df_selected_market.set_index("ticker")["weight"].to_dict()
    
    # Record the weights for this period
    for ticker, weight in weight_dict.items():
        semester_weights.append({
            'Period Start': period_start,
            'Period End': period_end,
            'Stock': ticker,
            'Weight (%)': weight * 100
        })
    
    # --- Calculate Portfolio Returns During the Investment Period ---
    df_prices_pivot = df_prices_period.pivot(index="date", columns="ticker", values="adjusted").sort_index()
    first_day = df_prices_pivot.index.min()
    
    # Remove any tickers that are missing a price on the first day.
    tickers_missing = df_prices_pivot.loc[first_day].isna()
    tickers_missing = tickers_missing[tickers_missing].index.tolist()
    for t in tickers_missing:
        weight_dict.pop(t, None)
    df_prices_pivot.drop(columns=tickers_missing, inplace=True)
    
    if df_prices_pivot.empty or not weight_dict:
        df_index = extend_index(df_index, period_start, period_end, previous_index_value)
        continue
    
    # Compute daily returns and apply the computed weights.
    df_returns = df_prices_pivot.pct_change().fillna(0)
    for ticker, weight in weight_dict.items():
        if ticker in df_returns.columns:
            df_returns[ticker] *= weight
    df_returns["portfolio_return"] = df_returns[list(weight_dict.keys())].sum(axis=1)
    
    # --- Attribution: Accumulate the Contributions for Each Ticker ---
    for ticker in weight_dict.keys():
        ticker_contrib = df_returns[ticker].sum()
        contrib_dict[ticker] = contrib_dict.get(ticker, 0) + ticker_contrib

    # Compute the cumulative return for the period and update the index.
    index_series = (1 + df_returns["portfolio_return"]).cumprod() * previous_index_value
    previous_index_value = index_series.iloc[-1]
    df_temp = index_series.reset_index()
    df_temp.columns = ["date", "momentum_index"]
    df_index = pd.concat([df_index, df_temp], ignore_index=True)

# -------------------------------
# Process and Plot Comparison Indices Data
# -------------------------------
comparison_data = {}
for comp in comparison_indices:
    df_comp = df_prices[
        (df_prices["date"] >= start_date) &
        (df_prices["date"] <= end_date) &
        (df_prices["ticker"] == comp)
    ][["date", "adjusted"]].copy()
    if not df_comp.empty:
        df_comp.sort_values("date", inplace=True)
        # Normalize so that the series starts at 100.
        df_comp["normalized"] = df_comp["adjusted"] / df_comp["adjusted"].iloc[0] * 100
        comparison_data[comp] = df_comp

# Convert index dates to plain Python dates for display.
df_index["date"] = pd.to_datetime(df_index["date"]).dt.date

plt.figure(figsize=(10, 6))
plt.plot(df_index["date"], df_index["momentum_index"], label="Momentum ETF Index", linewidth=2)
for comp, df_comp in comparison_data.items():
    plt.plot(df_comp["date"], df_comp["normalized"], label=comp,
             linestyle="dashed", linewidth=2, alpha=0.75)
plt.xlabel("Date")
plt.ylabel("Index Value (Normalized to 100)")
plt.title("Momentum ETF Index vs. Comparison Indices ({} to {})".format(start_date_str, end_date_str))
plt.ticklabel_format(style='plain', axis='y')  # Disable scientific notation on the y-axis
plt.legend()
plt.grid(True)
plt.savefig("momentum_vs_comparisons_pricetrends_correctted.png")
plt.show()

df_index.to_csv("df_index.csv", index=False)
#df_comp.to_csv("df_comp.csv", index=False)
comparison_data['^GSPC'].to_csv("comparison_data_snp.csv", index = False)
comparison_data['^IXIC'].to_csv("comparison_data_nasdaq.csv", index = False)

#'^GSPC', '^IXIC', 'AAPL'




# -------------------------------
# Plot the Evolution of a $100K Investment
# -------------------------------
investment_multiplier = 100000 / 100  # This equals 1000
df_index["investment_value"] = df_index["momentum_index"] * investment_multiplier

plt.figure(figsize=(10, 6))
plt.plot(df_index["date"], df_index["investment_value"],
         label="Momentum ETF ($100K Investment)", linewidth=2)
for comp, df_comp in comparison_data.items():
    df_comp["investment_value"] = df_comp["normalized"] * investment_multiplier
    plt.plot(df_comp["date"], df_comp["investment_value"],
             label="{} ($100K Investment)".format(comp),
             linestyle="dashed", linewidth=2, alpha=0.75)
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.title("Evolution of $100K Investment: Momentum ETF vs. Comparison Indices")
plt.ticklabel_format(style='plain', axis='y')
plt.legend()
plt.grid(True)
plt.savefig("investment_evolution_comparisons.png")
plt.show()

# -------------------------------
# Create a Tidy DataFrame for Stock Attribution (Overall Contribution)
# -------------------------------
attribution_df = pd.DataFrame({
    'Stock': list(contrib_dict.keys()),
    'Total Contribution (%)': [v * 100 for v in contrib_dict.values()]
})
attribution_df.sort_values('Total Contribution (%)', ascending=False, inplace=True)

#print("Overall Stock Contribution to ETF Growth:")
#print(attribution_df)
#attribution_df.to_csv("stock_attribution.csv", index=False)


total_contribution = sum(contrib_dict.values())  # Get total contribution sum
total_attribution_df = pd.DataFrame({
    'Stock': list(contrib_dict.keys()),
    'Total Contribution (%)': [(v / total_contribution) * 100 for v in contrib_dict.values()]
})

print("Overall Stock Contribution to ETF Growth:")
print(total_attribution_df)
total_attribution_df.to_csv("total_stock_attribution.csv", index=False)


# -------------------------------
# Create a Tidy DataFrame for Period Weights
# -------------------------------
semester_weights_df = pd.DataFrame(semester_weights)
semester_weights_df.sort_values(by=["Period Start", "Weight (%)"], ascending=[True, False], inplace=True)
print("\nStock Weights for Each Period:")
print(semester_weights_df)
semester_weights_df.to_csv("semester_stock_weights.csv", index=False)

# -------------------------------
# Explanation of Weighting and Selection Methods:
# -------------------------------
# Weighting Methods:
# 1. Market Cap Weighting: Stocks are weighted according to their market capitalization.
#
# 2. Equal Weighting: Each selected stock receives the same allocation.
#
# 3. Slope Weighting: Stocks are weighted proportionally to the magnitude of their positive
#    trend (price or search interest). A cap of 30% per stock is imposed and any excess
#    is redistributed among the other stocks.
#
# Selection Methods:
# 1. search_interest: Uses historical monthly search interest data from the preceding
#    evaluation period (X months) to compute a slope for each ticker. Only tickers with a
#    positive search interest slope are selected.
#
# 2. price_trend: Uses historical daily adjusted price data from the preceding evaluation
#    period (X months) to compute a slope for each ticker. Only tickers with a positive
#    price trend (i.e., positive slope) are selected.
