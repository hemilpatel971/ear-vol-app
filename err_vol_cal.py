import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Earnings Position Checker", page_icon="📈")

# --- FINANCIAL LOGIC (Preserved from original) ---

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    sorted_dates = [date for date in sorted_dates if date > today]

    if not sorted_dates:
        raise ValueError("No future option expiration dates found.")

    first_date = sorted_dates[0]
    for date in sorted_dates:
        if date >= cutoff_date:
            selected_dates = [first_date, date]
            return [d.strftime("%Y-%m-%d") for d in dict.fromkeys(selected_dates)]

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

def retry_yfinance_call(callable_obj, attempts=2, delay=2):
    last_error = None
    for attempt in range(attempts):
        try:
            return callable_obj()
        except Exception as error:
            last_error = error
            if "Too Many Requests" not in str(error) and "Rate limited" not in str(error):
                raise
            if attempt < attempts - 1:
                time.sleep(delay)
    raise last_error

@st.cache_data(ttl=3600, show_spinner=False)
def compute_recommendation(ticker_symbol):
    ticker_symbol = ticker_symbol.strip().upper()
    stock = yf.Ticker(ticker_symbol)
    
    options = retry_yfinance_call(lambda: list(stock.options))
    if not options:
        return {"error": f"No options found for '{ticker_symbol}'."}
    
    try:
        exp_dates = filter_dates(options)
    except Exception:
        return {"error": "Not enough option data (need dates 45+ days out)."}
    
    price_history = retry_yfinance_call(lambda: stock.history(period='3mo'))
    if price_history.empty:
        return {"error": "Could not retrieve current stock price."}
    underlying_price = price_history['Close'].iloc[-1]

    atm_iv = {}
    straddle = None
    
    for i, exp_date in enumerate(exp_dates):
        chain = retry_yfinance_call(lambda exp_date=exp_date: stock.option_chain(exp_date))
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
