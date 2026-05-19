import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import os
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="Earnings Position Checker", page_icon="📈")

# --- FINANCIAL LOGIC (Preserved from original) ---

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    return result.iloc[-1] if return_last_only else result.dropna()

def build_term_structure(days, ivs):
    days, ivs = np.array(days), np.array(ivs)
    sort_idx = days.argsort()
    days, ivs = days[sort_idx], ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]: return ivs[0]
        elif dte > days[-1]: return ivs[-1]
        else: return float(spline(dte))
    return term_spline

def get_marketdata_token():
    try:
        token = st.secrets.get("MARKETDATA_TOKEN")
    except Exception:
        token = None
    return token or os.getenv("MARKETDATA_TOKEN")

def marketdata_get(path, params=None):
    token = get_marketdata_token()
    if not token:
        raise ValueError("Missing MARKETDATA_TOKEN.")

    response = requests.get(
        f"https://api.marketdata.app/v1/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params or {},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("s") == "error":
        raise ValueError(data.get("errmsg", "MarketData.app returned an error."))
    if data.get("s") == "no_data":
        raise ValueError("MarketData.app returned no data.")
    return data

def first_value(data, key):
    values = data.get(key, [])
    if isinstance(values, list) and values:
        return values[0]
    return None

def option_rows(chain):
    row_count = len(chain.get("optionSymbol", []))
    rows = []
    for i in range(row_count):
        rows.append({key: values[i] for key, values in chain.items() if isinstance(values, list) and len(values) > i})
    return rows

def marketdata_price_history(ticker_symbol):
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=110)
    data = marketdata_get(
        f"stocks/candles/D/{ticker_symbol}/",
        {"from": start_date.isoformat(), "to": end_date.isoformat()},
    )

    import pandas as pd
    price_history = pd.DataFrame({
        "Open": data.get("o", []),
        "High": data.get("h", []),
        "Low": data.get("l", []),
        "Close": data.get("c", []),
        "Volume": data.get("v", []),
    })
    return price_history.dropna()

def compute_recommendation_marketdata(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()
    dte_targets = [7, 14, 30, 45]
    atm_iv = {}
    straddle = None
    underlying_price = None

    for dte_target in dte_targets:
        chain = marketdata_get(
            f"options/chain/{ticker_symbol}/",
            {"dte": dte_target, "strikeLimit": 2, "weekly": "true", "monthly": "true"},
        )
        rows = option_rows(chain)
        if not rows:
            continue

        underlying_price = underlying_price or rows[0].get("underlyingPrice")
        expiration_ts = rows[0].get("expiration")
        if expiration_ts is None:
            continue

        expiration_date = datetime.fromtimestamp(expiration_ts).date()
        dte = max((expiration_date - datetime.today().date()).days, 0)
        calls = [row for row in rows if row.get("side") == "call"]
        puts = [row for row in rows if row.get("side") == "put"]
        if not calls or not puts or underlying_price is None:
            continue

        call = min(calls, key=lambda row: abs(row.get("strike", 0) - underlying_price))
        put = min(puts, key=lambda row: abs(row.get("strike", 0) - underlying_price))

        if call.get("iv") is not None and put.get("iv") is not None:
            atm_iv[dte] = (float(call["iv"]) + float(put["iv"])) / 2.0

        if straddle is None:
            call_mid = call.get("mid")
            put_mid = put.get("mid")
            if call_mid is None:
                call_mid = (float(call.get("bid", 0)) + float(call.get("ask", 0))) / 2.0
            if put_mid is None:
                put_mid = (float(put.get("bid", 0)) + float(put.get("ask", 0))) / 2.0
            straddle = float(call_mid) + float(put_mid)

    if underlying_price is None:
        quote = marketdata_get(f"stocks/quotes/{ticker_symbol}/")
        underlying_price = first_value(quote, "last") or first_value(quote, "mid")

    if not atm_iv or max(atm_iv.keys()) < 45:
        return {"error": "Not enough option data from MarketData.app (need dates 45+ days out)."}
    if underlying_price is None:
        return {"error": "Could not retrieve current stock price."}

    dtes = list(atm_iv.keys())
    iv_vals = list(atm_iv.values())
    term_spline = build_term_structure(dtes, iv_vals)
    ts_slope = (term_spline(45) - term_spline(min(dtes))) / (45 - min(dtes))

    price_history = marketdata_price_history(ticker_symbol)
    if price_history.empty:
        return {"error": "Could not retrieve price history."}

    iv30_rv30 = term_spline(30) / yang_zhang(price_history)
    avg_volume = price_history['Volume'].rolling(30).mean().iloc[-1]
    expected_move = f"{round(straddle / underlying_price * 100, 2)}%" if straddle else "N/A"

    return {
        'raw_vol': avg_volume,
        'raw_iv_rv': iv30_rv30,
        'raw_slope': ts_slope,
        'avg_volume': avg_volume >= 1500000,
        'iv30_rv30': iv30_rv30 >= 1.25,
        'ts_slope_0_45': ts_slope <= -0.00406,
        'expected_move': expected_move,
        'error': None
    }

@st.cache_data(ttl=900, show_spinner=False)
def compute_recommendation(ticker_symbol):
    if get_marketdata_token():
        return compute_recommendation_marketdata(ticker_symbol)

    ticker_symbol = ticker_symbol.strip().upper()
    stock = yf.Ticker(ticker_symbol)
    
    if not stock.options:
        return {"error": f"No options found for '{ticker_symbol}'."}
    
    try:
        exp_dates = filter_dates(list(stock.options))
    except Exception:
        return {"error": "Not enough option data (need dates 45+ days out)."}
    
    # Get current price
    hist = stock.history(period='1d')
    if hist.empty:
        return {"error": "Could not retrieve current stock price."}
    underlying_price = hist['Close'].iloc[-1]

    atm_iv = {}
    straddle = None
    
    for i, exp_date in enumerate(exp_dates):
        chain = stock.option_chain(exp_date)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty: continue

        call_idx = (calls['strike'] - underlying_price).abs().idxmin()
        put_idx = (puts['strike'] - underlying_price).abs().idxmin()
        
        iv_val = (calls.loc[call_idx, 'impliedVolatility'] + puts.loc[put_idx, 'impliedVolatility']) / 2.0
        atm_iv[exp_date] = iv_val

        if i == 0:
            call_mid = (calls.loc[call_idx, 'bid'] + calls.loc[call_idx, 'ask']) / 2.0
            put_mid = (puts.loc[put_idx, 'bid'] + puts.loc[put_idx, 'ask']) / 2.0
            straddle = call_mid + put_mid

    if not atm_iv:
        return {"error": "Could not determine ATM IV."}

    # Term Structure
    today = datetime.today().date()
    dtes = [(datetime.strptime(d, "%Y-%m-%d").date() - today).days for d in atm_iv.keys()]
    iv_vals = list(atm_iv.values())
    term_spline = build_term_structure(dtes, iv_vals)
    
    ts_slope = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
    
    # Volatility and Volume
    price_history = stock.history(period='3mo')
    iv30_rv30 = term_spline(30) / yang_zhang(price_history)
    avg_volume = price_history['Volume'].rolling(30).mean().iloc[-1]
    expected_move = f"{round(straddle / underlying_price * 100, 2)}%" if straddle else "N/A"

    return {
        'raw_vol': avg_volume,            # Raw volume number
        'raw_iv_rv': iv30_rv30,           # Raw ratio (e.g. 1.45)
        'raw_slope': ts_slope,            # Raw slope (e.g. -0.005)
        'avg_volume': avg_volume >= 1500000,
        'iv30_rv30': iv30_rv30 >= 1.25,
        'ts_slope_0_45': ts_slope <= -0.00406,
        'expected_move': expected_move,
        'error': None
    }
# --- STREAMLIT UI ---

st.title("📈 Earnings Position Checker")

st.markdown("""
<p style='font-size: 13px; color: gray; margin-bottom: 20px;'>
<strong>DISCLAIMER:</strong> This software is for educational purposes only and not financial advisors. Consult a professional before investing.
</p>
""", unsafe_allow_html=True)

# --- NEW: MONEY MANAGEMENT SECTION ---
st.subheader("🛡️ Money Management & Sizing")
st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #00ffcc; margin-bottom: 20px;">
    <strong>Kelly Criterion Guidance:</strong><br>
    <ul>
        <li><strong>Short Straddle:</strong> Trade size should be <strong>2%</strong> of total capital (30% Kelly).</li>
        <li><strong>Long Calendar:</strong> Trade size should be <strong>6%</strong> of total capital (10% Kelly).</li>
    </ul>
    <em>Example: For a $10,000 account, a Long Calendar allocation/risk would be <strong>$600</strong>.</em>
</div>
""", unsafe_allow_html=True)

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "").upper()

st.markdown("""
    <style>
    /* Make the table font larger */
    .stTable {
        font-size: 20px !important;
    }
    /* Optional: Make the metric labels a bit bigger too */
    [data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

if st.button("Run Analysis"):
    if ticker:
        with st.spinner(f'Fetching data for {ticker}...'):
            try:
                result = compute_recommendation(ticker)
                
                if result.get('error'):
                    st.error(result['error'])
                else:
                    # Recommendation Header
                    avg_v, iv_rv, slope = result['avg_volume'], result['iv30_rv30'], result['ts_slope_0_45']
                    
                    if avg_v and iv_rv and slope:
                        st.success("### ✅ Recommendation: RECOMMENDED")
                    elif slope and (avg_v or iv_rv):
                        st.warning("### ⚠️ Recommendation: CONSIDER")
                    else:
                        st.error("### ❌ Recommendation: AVOID")
                    
                    # Analysis Table - Using st.table for better font control
                    import pandas as pd
                    summary_df = pd.DataFrame({
                        "Metric": ["Daily Volume", "IV30 / RV30 Ratio", "TS Slope (0-45)"],
                        "Actual Value": [
                            f"{result['raw_vol']:,.0f}", 
                            f"{result['raw_iv_rv']:.2f}", 
                            f"{result['raw_slope']:.5f}"
                        ],
                        "Target Threshold": [">= 1,500,000", ">= 1.25", "<= -0.00406"],
                        "Status": ["✅ PASS" if avg_v else "❌ FAIL", 
                                   "✅ PASS" if iv_rv else "❌ FAIL", 
                                   "✅ PASS" if slope else "❌ FAIL"]
                    })
                    
                    # Displaying with st.table (no index)
                    st.write("### Analysis Summary")
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
                    
                    # Large Styled Expected Move
                    st.markdown(f"""
                        <div style="text-align: center; padding: 25px; border: 2px solid #4CAF50; border-radius: 15px; background-color: rgba(0, 255, 204, 0.05); margin-top: 20px;">
                            <span style="font-size: 1.4em; color: #888; text-transform: uppercase; letter-spacing: 1px;">Expected Move (Market Price)</span><br>
                            <span style="font-size: 3.5em; font-weight: 900; color: #00ffcc;">{result['expected_move']}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                error_message = str(e)
                if "Too Many Requests" in error_message or "Rate limited" in error_message:
                    st.error(
                        "Yahoo Finance is rate-limiting requests from the Streamlit Cloud server. "
                        "Please wait a few minutes and try again, or try a different ticker."
                    )
                else:
                    st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a stock ticker.")
