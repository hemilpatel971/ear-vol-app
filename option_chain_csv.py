import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm


CONTRACT_MULTIPLIER = 100


st.set_page_config(page_title="Option Chain CSV Exporter", page_icon="📄", layout="wide")


def get_current_price(stock):
    history = stock.history(period="5d")
    if history.empty:
        raise ValueError("Could not fetch the current stock price.")
    return float(history["Close"].dropna().iloc[-1])


@st.cache_data(ttl=300, show_spinner=False)
def get_expiration_dates(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    return list(stock.options)


@st.cache_data(ttl=300, show_spinner=False)
def get_option_chain(ticker_symbol, expiration_date):
    stock = yf.Ticker(ticker_symbol)
    current_price = get_current_price(stock)
    chain = stock.option_chain(expiration_date)
    return current_price, chain.calls.copy(), chain.puts.copy()


def calculate_greeks(row, underlying_price, risk_free_rate):
    iv = row.get("impliedVolatility", np.nan)
    strike = row.get("strike", np.nan)
    dte = row.get("dte", 0)
    option_type = row.get("option_type")

    if pd.isna(iv) or pd.isna(strike) or iv <= 0 or strike <= 0 or dte <= 0:
        return pd.Series({"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan})

    t = dte / 365.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(underlying_price / strike) + (risk_free_rate + 0.5 * iv ** 2) * t) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(underlying_price * norm.pdf(d1) * iv) / (2 * sqrt_t)
            - risk_free_rate * strike * math.exp(-risk_free_rate * t) * norm.cdf(d2)
        ) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(underlying_price * norm.pdf(d1) * iv) / (2 * sqrt_t)
            + risk_free_rate * strike * math.exp(-risk_free_rate * t) * norm.cdf(-d2)
        ) / 365.0

    gamma = norm.pdf(d1) / (underlying_price * iv * sqrt_t)
    vega = underlying_price * norm.pdf(d1) * sqrt_t / 100.0

    return pd.Series({"delta": delta, "gamma": gamma, "theta": theta, "vega": vega})


def prepare_chain(calls, puts, ticker_symbol, expiration_date, underlying_price, risk_free_rate):
    calls = calls.copy()
    puts = puts.copy()
    calls["option_type"] = "call"
    puts["option_type"] = "put"
    options = pd.concat([calls, puts], ignore_index=True)

    expiration = datetime.strptime(expiration_date, "%Y-%m-%d").date()
    today = datetime.today().date()
    dte = max((expiration - today).days, 0)

    options["ticker"] = ticker_symbol
    options["expiration_date"] = expiration_date
    options["dte"] = dte
    options["underlying_price"] = underlying_price
    options["mid_price"] = (options["bid"].fillna(0) + options["ask"].fillna(0)) / 2.0
    options["bid_ask_spread"] = options["ask"] - options["bid"]
    options["spread_pct_of_mid"] = np.where(
        options["mid_price"] > 0,
        options["bid_ask_spread"] / options["mid_price"],
        np.nan,
    )

    options["moneyness"] = options["strike"] / underlying_price
    options["distance_from_spot"] = options["strike"] - underlying_price
    options["distance_from_spot_pct"] = options["distance_from_spot"] / underlying_price
    options["is_itm"] = np.where(
        ((options["option_type"] == "call") & (options["strike"] < underlying_price))
        | ((options["option_type"] == "put") & (options["strike"] > underlying_price)),
        True,
        False,
    )
    options["is_otm"] = ~options["is_itm"]

    options["intrinsic_value"] = np.where(
        options["option_type"] == "call",
        np.maximum(underlying_price - options["strike"], 0),
        np.maximum(options["strike"] - underlying_price, 0),
    )
    options["extrinsic_value"] = options["mid_price"] - options["intrinsic_value"]

    options["oi_x_strike"] = options["openInterest"].fillna(0) * options["strike"]
    options["oi_notional_100x"] = options["oi_x_strike"] * CONTRACT_MULTIPLIER
    options["volume_x_mid"] = options["volume"].fillna(0) * options["mid_price"]
    options["volume_notional_100x"] = options["volume_x_mid"] * CONTRACT_MULTIPLIER

    greeks = options.apply(calculate_greeks, axis=1, underlying_price=underlying_price, risk_free_rate=risk_free_rate)
    options = pd.concat([options, greeks], axis=1)

    options["delta_x_oi"] = options["delta"] * options["openInterest"].fillna(0)
    options["delta_notional"] = options["delta_x_oi"] * underlying_price * CONTRACT_MULTIPLIER
    options["gamma_x_oi"] = options["gamma"] * options["openInterest"].fillna(0)
    options["gamma_notional"] = options["gamma_x_oi"] * underlying_price * CONTRACT_MULTIPLIER
    options["vega_x_oi"] = options["vega"] * options["openInterest"].fillna(0)
    options["theta_x_oi"] = options["theta"] * options["openInterest"].fillna(0)

    preferred_order = [
        "ticker",
        "expiration_date",
        "dte",
        "option_type",
        "contractSymbol",
        "underlying_price",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "mid_price",
        "bid_ask_spread",
        "spread_pct_of_mid",
        "volume",
        "openInterest",
        "impliedVolatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "delta_x_oi",
        "delta_notional",
        "gamma_x_oi",
        "gamma_notional",
        "vega_x_oi",
        "theta_x_oi",
        "oi_x_strike",
        "oi_notional_100x",
        "volume_x_mid",
        "volume_notional_100x",
        "moneyness",
        "distance_from_spot",
        "distance_from_spot_pct",
        "is_itm",
        "is_otm",
        "intrinsic_value",
        "extrinsic_value",
        "lastTradeDate",
        "inTheMoney",
        "contractSize",
        "currency",
    ]
    remaining_cols = [col for col in options.columns if col not in preferred_order]
    return options[[col for col in preferred_order if col in options.columns] + remaining_cols]


def build_summary(options):
    calls = options[options["option_type"] == "call"]
    puts = options[options["option_type"] == "put"]

    call_oi = calls["openInterest"].fillna(0).sum()
    put_oi = puts["openInterest"].fillna(0).sum()
    call_volume = calls["volume"].fillna(0).sum()
    put_volume = puts["volume"].fillna(0).sum()
    itm_oi = options.loc[options["is_itm"], "openInterest"].fillna(0).sum()
    otm_oi = options.loc[options["is_otm"], "openInterest"].fillna(0).sum()

    max_oi_row = options.loc[options["openInterest"].fillna(0).idxmax()]
    max_volume_row = options.loc[options["volume"].fillna(0).idxmax()]

    return {
        "Put/Call OI Ratio": put_oi / call_oi if call_oi else np.nan,
        "Put/Call Volume Ratio": put_volume / call_volume if call_volume else np.nan,
        "ITM/OTM OI Ratio": itm_oi / otm_oi if otm_oi else np.nan,
        "Total Call OI": call_oi,
        "Total Put OI": put_oi,
        "Total Call Volume": call_volume,
        "Total Put Volume": put_volume,
        "Max OI Strike": max_oi_row["strike"],
        "Max OI Type": max_oi_row["option_type"],
        "Max Volume Strike": max_volume_row["strike"],
        "Max Volume Type": max_volume_row["option_type"],
        "Net Delta Notional": options["delta_notional"].sum(skipna=True),
        "Total Gamma Notional": options["gamma_notional"].sum(skipna=True),
        "Total Vega x OI": options["vega_x_oi"].sum(skipna=True),
        "Total Theta x OI": options["theta_x_oi"].sum(skipna=True),
    }


st.title("Option Chain CSV Exporter")
st.caption("Fetch one yfinance option chain, enrich it with open-interest, moneyness, and Greek exposure columns, then download it as CSV.")

with st.sidebar:
    ticker_symbol = st.text_input("Ticker", value="AAPL").strip().upper()
    risk_free_rate_pct = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.25)
    risk_free_rate = risk_free_rate_pct / 100.0

if ticker_symbol:
    try:
        expiration_dates = get_expiration_dates(ticker_symbol)
    except Exception as error:
        st.error(f"Could not fetch expiration dates for {ticker_symbol}: {error}")
        st.stop()

    if not expiration_dates:
        st.error(f"No option expirations found for {ticker_symbol}.")
        st.stop()

    expiration_date = st.selectbox("Expiration date", expiration_dates)

    if st.button("Fetch Option Chain", type="primary"):
        try:
            with st.spinner(f"Fetching {ticker_symbol} option chain for {expiration_date}..."):
                underlying_price, calls, puts = get_option_chain(ticker_symbol, expiration_date)
                enriched_chain = prepare_chain(calls, puts, ticker_symbol, expiration_date, underlying_price, risk_free_rate)
                summary = build_summary(enriched_chain)

            st.success(f"Fetched {len(enriched_chain):,} contracts. Current underlying price: ${underlying_price:,.2f}")

            metric_cols = st.columns(4)
            metric_cols[0].metric("Put/Call OI", f"{summary['Put/Call OI Ratio']:.2f}")
            metric_cols[1].metric("Put/Call Volume", f"{summary['Put/Call Volume Ratio']:.2f}")
            metric_cols[2].metric("ITM/OTM OI", f"{summary['ITM/OTM OI Ratio']:.2f}")
            metric_cols[3].metric("Net Delta Notional", f"${summary['Net Delta Notional']:,.0f}")

            st.subheader("Summary")
            st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

            st.subheader("Enriched Option Chain")
            st.dataframe(enriched_chain, use_container_width=True, hide_index=True)

            csv_data = enriched_chain.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"{ticker_symbol}_{expiration_date}_option_chain.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.error(f"Could not fetch or process the option chain: {error}")
