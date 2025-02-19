import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# =============================================================================
# Data Loading (cached)
# =============================================================================
@st.cache_data
def load_data():
    df_market = pd.read_csv("../data_raw/market_cap_df_all.csv")
    df_prices = pd.read_csv("../data_proc/prices_daily.csv")
    df_search = pd.read_csv("../data_raw/monthly_search_interest_data_tickers.csv")
    # Standardize search data column names
    df_search.rename(columns={"date": "year_month", "value": "monthly_search_interest"}, inplace=True)
    # Convert dates
    df_prices["date"] = pd.to_datetime(df_prices["date"])
    df_search["year_month"] = pd.to_datetime(df_search["year_month"])
    # Clean ticker names in search data (remove a leading '$')
    df_search["ticker"] = df_search["ticker_name"].str.replace(r"^\$", "", regex=True)
    return df_market, df_prices, df_search

@st.cache_data
def load_company_names():
    # Load the updated company file; columns: "ticker" and "company"
    df_companies = pd.read_csv("data_raw/seo_companies.csv")
    df_companies["ticker"] = df_companies["ticker"].str.replace(r"^\$", "", regex=True).str.strip().str.upper()
    company_map = dict(zip(df_companies["ticker"], df_companies["company"]))
    return company_map

df_market, df_prices, df_search = load_data()
company_names = load_company_names()

def get_full_label(ticker):
    """Return a label string 'Company (TICKER)' given a ticker."""
    company_name = company_names.get(ticker)
    if company_name:
        return f"{company_name} ({ticker})"
    return ticker

# =============================================================================
# Helper Functions
# =============================================================================
def compute_slope(values):
    """Compute the slope using a linear fit over the given values."""
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return slope

def get_periods(start_date, end_date, period_months):
    """
    Return a list of (period_start, period_end) tuples representing periods of a given number of months.
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
    Given a period and a constant index value, returns a DataFrame with all business days in that period set to that value.
    """
    dates = pd.date_range(period_start, period_end, freq="B")
    return pd.concat([df_index, pd.DataFrame({"date": dates, "momentum_index": value})], ignore_index=True)

def cap_weights(weights, cap=0.3):
    """
    Given a dictionary of weights, cap each weight at the specified cap (default 0.3)
    and redistribute the excess proportionally among the other stocks.
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

def run_backtest(start_date, end_date, evaluation_period_months, weighting_method, comparison_indices):
    """
    Run the backtest using the parameters and return a dictionary of results.
    
    For each investment period, the strategy selects stocks based on historical monthly search interest data 
    from the _prior_ evaluation period. That is, for each period [period_start, period_end], it uses data from:
    
        historical_start = period_start - evaluation_period_months 
        historical_end   = period_start - 1 day
    
    to compute the search interest slope. Only tickers with a positive slope are selected.
    """
    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)
    periods = get_periods(start_date_ts, end_date_ts, evaluation_period_months)
    
    # Exclude the comparison tickers from portfolio selection.
    all_tickers = df_market["ticker"].unique()
    candidate_tickers = [t for t in all_tickers if t not in comparison_indices]
    
    df_index = pd.DataFrame(columns=["date", "momentum_index"])
    selected_tickers_log = []
    previous_index_value = 100  # starting value for the index
    contrib_dict = {}
    period_weights = []
    
    for period_start, period_end in periods:
        # Define historical window for selection:
        historical_start = period_start - pd.DateOffset(months=evaluation_period_months)
        if historical_start < start_date_ts:
            historical_start = start_date_ts
        historical_end = period_start - pd.Timedelta(days=1)
        
        # Use historical search interest data to select tickers.
        df_search_hist = df_search[(df_search["year_month"] >= historical_start) & 
                                   (df_search["year_month"] <= historical_end)]
        if df_search_hist.empty:
            selected_tickers = candidate_tickers.copy()
        else:
            selected_tickers = [
                ticker for ticker in candidate_tickers
                if compute_slope(df_search_hist[df_search_hist["ticker"] == ticker]["monthly_search_interest"].values) > 0
            ]
        
        # Filter to tickers with price data during the investment period.
        df_prices_period = df_prices[(df_prices["date"] >= period_start) & (df_prices["date"] <= period_end)]
        available_tickers = df_prices_period["ticker"].unique()
        selected_tickers = [t for t in selected_tickers if t in available_tickers]
        selected_tickers_log.append((period_start, period_end, selected_tickers))
        
        if not selected_tickers:
            df_index = extend_index(df_index, period_start, period_end, previous_index_value)
            continue
        
        # Compute weights.
        if weighting_method == 'market_cap':
            df_selected_market = df_market[df_market["ticker"].isin(selected_tickers)].dropna(subset=["market_cap"]).copy()
            total_cap = df_selected_market["market_cap"].sum()
            df_selected_market["weight"] = df_selected_market["market_cap"] / total_cap
            weight_dict = df_selected_market.set_index("ticker")["weight"].to_dict()
        elif weighting_method == 'equal':
            weight_dict = {ticker: 1/len(selected_tickers) for ticker in selected_tickers}
        elif weighting_method == 'slope':
            slopes = {}
            for ticker in selected_tickers:
                slope_val = compute_slope(df_search_hist[df_search_hist["ticker"] == ticker]["monthly_search_interest"].values)
                slopes[ticker] = slope_val
            total_slope = sum(slopes.values())
            if total_slope == 0:
                weight_dict = {ticker: 1/len(selected_tickers) for ticker in selected_tickers}
            else:
                weight_dict = {ticker: slopes[ticker]/total_slope for ticker in selected_tickers}
            weight_dict = cap_weights(weight_dict, cap=0.3)
        else:
            # Default to market cap weighting if unknown.
            df_selected_market = df_market[df_market["ticker"].isin(selected_tickers)].dropna(subset=["market_cap"]).copy()
            total_cap = df_selected_market["market_cap"].sum()
            df_selected_market["weight"] = df_selected_market["market_cap"] / total_cap
            weight_dict = df_selected_market.set_index("ticker")["weight"].to_dict()
        
        for ticker, weight in weight_dict.items():
            period_weights.append({
                'Period Start': period_start,
                'Period End': period_end,
                'Stock': ticker,
                'Weight (%)': weight * 100
            })
        
        # Calculate portfolio returns during the investment period.
        df_prices_pivot = df_prices_period.pivot(index="date", columns="ticker", values="adjusted").sort_index()
        first_day = df_prices_pivot.index.min()
        first_day_na = df_prices_pivot.loc[first_day].isna()
        tickers_missing = first_day_na[first_day_na].index.tolist()
        for t in tickers_missing:
            weight_dict.pop(t, None)
        df_prices_pivot.drop(columns=tickers_missing, inplace=True)
        
        if df_prices_pivot.empty or not weight_dict:
            df_index = extend_index(df_index, period_start, period_end, previous_index_value)
            continue
        
        df_returns = df_prices_pivot.pct_change().fillna(0)
        for ticker, weight in weight_dict.items():
            if ticker in df_returns.columns:
                df_returns[ticker] *= weight
        df_returns["portfolio_return"] = df_returns[list(weight_dict.keys())].sum(axis=1)
        
        for ticker in weight_dict.keys():
            ticker_contrib = df_returns[ticker].sum()
            contrib_dict[ticker] = contrib_dict.get(ticker, 0) + ticker_contrib
        
        index_series = (1 + df_returns["portfolio_return"]).cumprod() * previous_index_value
        previous_index_value = index_series.iloc[-1]
        df_temp = index_series.reset_index()
        df_temp.columns = ["date", "momentum_index"]
        df_index = pd.concat([df_index, df_temp], ignore_index=True)
    
    # Process comparison indices data for plotting.
    comparison_data = {}
    for comp in comparison_indices:
        df_comp = df_prices[
            (df_prices["date"] >= start_date_ts) &
            (df_prices["date"] <= end_date_ts) &
            (df_prices["ticker"] == comp)
        ][["date", "adjusted"]].copy()
        if not df_comp.empty:
            df_comp.sort_values("date", inplace=True)
            df_comp["normalized"] = df_comp["adjusted"] / df_comp["adjusted"].iloc[0] * 100
            comparison_data[comp] = df_comp
            
    df_index["date"] = pd.to_datetime(df_index["date"]).dt.date
    
    return {
        "df_index": df_index,
        "contrib_dict": contrib_dict,
        "period_weights": pd.DataFrame(period_weights),
        "selected_tickers_log": selected_tickers_log,
        "comparison_data": comparison_data,
        "periods": periods
    }

# =============================================================================
# Page Navigation
# =============================================================================
page = st.sidebar.radio("Navigation", ["Backtest", "Stock Analysis", "Period Analysis"])

# Global (sidebar) parameters for the backtest and period analysis pages:
global_start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
global_end_date = st.sidebar.date_input("End Date", datetime.date(2024, 12, 31))
evaluation_period_months = st.sidebar.selectbox("Evaluation Period (months)", [3, 6, 12], index=1)
weighting_method = st.sidebar.selectbox("Weighting Method", ["market_cap", "equal", "slope"])
comparison_indices_input = st.sidebar.text_input("Comparison Indices (comma-separated)", "^GSPC,^IXIC,AAPL")
comparison_indices = [x.strip() for x in comparison_indices_input.split(",") if x.strip()]

# =============================================================================
# Page 1: Backtest
# =============================================================================
if page == "Backtest":
    st.title("Momentum ETF Backtest")
    st.write("Running backtest with the specified parameters...")
    
    backtest_results = run_backtest(global_start_date, global_end_date, evaluation_period_months, weighting_method, comparison_indices)
    df_index = backtest_results["df_index"]
    comparison_data = backtest_results["comparison_data"]
    period_weights_df = backtest_results["period_weights"]
    contrib_dict = backtest_results["contrib_dict"]
    
    # Plot 1: Momentum ETF Index vs. Comparison Indices
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(df_index["date"], df_index["momentum_index"], label="Momentum ETF Index", linewidth=2)
    for comp, df_comp in comparison_data.items():
        label = get_full_label(comp)
        ax1.plot(df_comp["date"], df_comp["normalized"], label=label,
                 linestyle="dashed", linewidth=2, alpha=0.75)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Index Value (Normalized to 100)")
    ax1.set_title("Momentum ETF Index vs Comparison Indices")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    
    # Plot 2: Evolution of a $100K Investment
    investment_multiplier = 100000 / 100  # Normalize to $100K starting from index 100.
    df_index["investment_value"] = df_index["momentum_index"] * investment_multiplier
    
    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.plot(df_index["date"], df_index["investment_value"],
             label="Momentum ETF ($100K Investment)", linewidth=2)
    for comp, df_comp in comparison_data.items():
        df_comp["investment_value"] = df_comp["normalized"] * investment_multiplier
        label = get_full_label(comp)
        ax2.plot(df_comp["date"], df_comp["investment_value"],
                 label=f"{label} ($100K Investment)",
                 linestyle="dashed", linewidth=2, alpha=0.75)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.set_title("Evolution of $100K Investment")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    
    st.subheader("Overall Stock Contribution to ETF Growth")
    
    total_contribution = sum(contrib_dict.values())  # Get total contribution sum
    attribution_df = pd.DataFrame({
        'Stock': list(contrib_dict.keys()),
        'Total Contribution (%)': [(v / total_contribution) * 100 for v in contrib_dict.values()]
    })    
    
    
    attribution_df.sort_values('Total Contribution (%)', ascending=False, inplace=True)
    attribution_df["Stock"] = attribution_df["Stock"].map(get_full_label)
    st.dataframe(attribution_df)
    
    st.subheader("Stock Weights for Each Period")
    period_weights_df.sort_values(by=["Period Start", "Weight (%)"], ascending=[True, False], inplace=True)
    display_pw = period_weights_df.copy()
    display_pw["Stock"] = display_pw["Stock"].map(get_full_label)
    st.dataframe(display_pw)

# =============================================================================
# Page 2: Stock Analysis
# =============================================================================
elif page == "Stock Analysis":
    st.title("Stock / Index Analysis")
    
    ticker_options = sorted(df_prices["ticker"].unique())
    options_dict = {ticker: f"{company_names.get(ticker, ticker)} ({ticker})" for ticker in ticker_options}
    selected_stock = st.selectbox("Select Stock/Index", list(options_dict.keys()),
                                  format_func=lambda x: options_dict[x])
    
    analysis_start_date = st.date_input("Analysis Start Date", datetime.date(2015, 1, 1), key="analysis_start")
    analysis_end_date = st.date_input("Analysis End Date", datetime.date(2024, 12, 31), key="analysis_end")
    
    df_stock = df_prices[(df_prices["ticker"] == selected_stock) &
                         (df_prices["date"] >= pd.Timestamp(analysis_start_date)) &
                         (df_prices["date"] <= pd.Timestamp(analysis_end_date))].copy()
    if df_stock.empty:
        st.write("No price data available for this stock in the selected period.")
    else:
        df_stock.sort_values("date", inplace=True)
        df_stock["normalized"] = df_stock["adjusted"] / df_stock["adjusted"].iloc[0] * 100
        st.subheader("Stock Performance")
        fig_stock, ax_stock = plt.subplots(figsize=(10,4))
        ax_stock.plot(df_stock["date"], df_stock["normalized"],
                      label=get_full_label(selected_stock), linewidth=2)
        ax_stock.set_xlabel("Date")
        ax_stock.set_ylabel("Normalized Price (100 = Start)")
        ax_stock.set_title(f"{get_full_label(selected_stock)} Price Performance")
        ax_stock.legend()
        ax_stock.grid(True)
        st.pyplot(fig_stock)
    
    df_search_stock = df_search[(df_search["ticker"] == selected_stock) &
                                (df_search["year_month"] >= pd.Timestamp(analysis_start_date)) &
                                (df_search["year_month"] <= pd.Timestamp(analysis_end_date))].copy()
    if df_search_stock.empty:
        st.write("No search interest data available for this stock in the selected period.")
    else:
        df_search_stock.sort_values("year_month", inplace=True)
        st.subheader("Search Interest")
        rolling_months = st.number_input("Rolling Average Window (months)", min_value=1, max_value=12, value=3, step=1)
        df_search_stock["rolling_interest"] = df_search_stock["monthly_search_interest"].rolling(window=rolling_months).mean()
        
        fig_search, ax_search = plt.subplots(figsize=(10,4))
        ax_search.plot(df_search_stock["year_month"], df_search_stock["monthly_search_interest"],
                       label="Raw Search Interest", alpha=0.5)
        ax_search.plot(df_search_stock["year_month"], df_search_stock["rolling_interest"],
                       label=f"{rolling_months}-Month Rolling Average", linewidth=2)
        ax_search.set_xlabel("Date")
        ax_search.set_ylabel("Search Interest")
        ax_search.set_title(f"Search Interest for {get_full_label(selected_stock)}")
        ax_search.legend()
        ax_search.grid(True)
        st.pyplot(fig_search)

# =============================================================================
# Page 3: Period Analysis
# =============================================================================
elif page == "Period Analysis":
    st.title("Period Analysis")
    
    backtest_results = run_backtest(global_start_date, global_end_date, evaluation_period_months, weighting_method, comparison_indices)
    df_index = backtest_results["df_index"]
    period_weights_df = backtest_results["period_weights"]
    comparison_data = backtest_results["comparison_data"]
    periods = backtest_results["periods"]
    
    period_options = [f"{p[0].date()} to {p[1].date()}" for p in periods]
    selected_period_str = st.selectbox("Select a Period", period_options)
    selected_index = period_options.index(selected_period_str)
    selected_period = periods[selected_index]
    period_start, period_end = selected_period
    
    st.write(f"Selected period: {period_start.date()} to {period_end.date()}")
    
    df_index_period = df_index[(pd.to_datetime(df_index["date"]) >= period_start) &
                               (pd.to_datetime(df_index["date"]) <= period_end)]
    fig_period, ax_period = plt.subplots(figsize=(10,6))
    ax_period.plot(df_index_period["date"], df_index_period["momentum_index"],
                   label="Momentum ETF Index", linewidth=2)
    for comp, df_comp in comparison_data.items():
        df_comp_period = df_comp[(df_comp["date"] >= period_start) & (df_comp["date"] <= period_end)]
        if not df_comp_period.empty:
            label = get_full_label(comp)
            ax_period.plot(df_comp_period["date"], df_comp_period["normalized"],
                           label=label, linestyle="dashed", linewidth=2, alpha=0.75)
    ax_period.set_xlabel("Date")
    ax_period.set_ylabel("Index Value (Normalized to 100)")
    ax_period.set_title("Momentum ETF Index vs Comparison Indices (Selected Period)")
    ax_period.legend()
    ax_period.grid(True)
    st.pyplot(fig_period)
    
    period_weights_selected = period_weights_df[
        (pd.to_datetime(period_weights_df["Period Start"]) == period_start) &
        (pd.to_datetime(period_weights_df["Period End"]) == period_end)
    ]
    st.subheader("Stocks Selected and Their % Participation")
    display_pw = period_weights_selected.copy()
    display_pw["Stock"] = display_pw["Stock"].map(get_full_label)
    st.dataframe(display_pw)
    
    st.subheader("Individual Stock Performance in the Selected Period")
    for idx, row in period_weights_selected.iterrows():
        ticker = row["Stock"]
        display_label = get_full_label(ticker)
        st.write(f"### {display_label}")
        df_stock_period = df_prices[(df_prices["ticker"] == ticker) &
                                    (df_prices["date"] >= period_start) &
                                    (df_prices["date"] <= period_end)].copy()
        if df_stock_period.empty:
            st.write("No data available.")
        else:
            df_stock_period.sort_values("date", inplace=True)
            df_stock_period["normalized"] = df_stock_period["adjusted"] / df_stock_period["adjusted"].iloc[0] * 100
            fig_stock, ax_stock = plt.subplots(figsize=(10,4))
            ax_stock.plot(df_stock_period["date"], df_stock_period["normalized"],
                          label=display_label, linewidth=2)
            ax_stock.set_xlabel("Date")
            ax_stock.set_ylabel("Normalized Price")
            ax_stock.set_title(f"{display_label} Performance in Selected Period")
            ax_stock.legend()
            ax_stock.grid(True)
            st.pyplot(fig_stock)
