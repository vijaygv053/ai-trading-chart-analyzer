import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="India NSE MCX Analyzer", page_icon="📈", layout="wide")

st.title("🇮🇳 Indian NSE & MCX Trading Analyzer")
st.caption("Free data version | No API key | No broker login")

st.info("NSE data uses Yahoo Finance (unofficial; delays and availability vary). MCX analysis uses an uploaded OHLC CSV. This is not guaranteed live data.")

st_autorefresh(interval=300000, key="refresh")

st.sidebar.caption("Dashboard refresh: 5 minutes")
market = st.sidebar.radio("Market", ["NSE", "MCX"])


@st.cache_data(ttl=240)
def get_data(ticker, interval, period):
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [str(c).lower() for c in data.columns]

    required = ["open", "high", "low", "close"]
    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    if "volume" not in data.columns:
        data["volume"] = 0

    return data[required + ["volume"]].dropna()


def load_mcx_csv(upload):
    raw = pd.read_csv(upload)
    raw.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in raw.columns
    ]

    date_col = next(
        (c for c in ["datetime", "date_time", "timestamp", "date"]
         if c in raw.columns),
        None
    )

    if date_col is None:
        raise ValueError("CSV needs a Date or DateTime column.")

    dates = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)

    result = pd.DataFrame(index=dates)

    for col in ["open", "high", "low", "close"]:
        if col not in raw.columns:
            raise ValueError("Missing required column: " + col)
        result[col] = pd.to_numeric(raw[col], errors="coerce")

    if "volume" in raw.columns:
        result["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    else:
        result["volume"] = 0

    return result.dropna(
        subset=["open", "high", "low", "close"]
    ).sort_index()


def indicators(df):
    df = df.copy()

    for period in [9, 21, 50, 200]:
        df[f"EMA{period}"] = df["close"].ewm(
            span=period, adjust=False
        ).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    macd = (
        df["close"].ewm(span=12, adjust=False).mean()
        - df["close"].ewm(span=26, adjust=False).mean()
    )

    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd.ewm(span=9, adjust=False).mean()

    previous = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous).abs(),
        (df["low"] - previous).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    return df


def pattern(df):
    if len(df) < 2:
        return "Not enough candles"

    a = df.iloc[-2]
    b = df.iloc[-1]

    body = abs(b["close"] - b["open"])
    candle_range = max(b["high"] - b["low"], 0.000001)

    upper = b["high"] - max(b["open"], b["close"])
    lower = min(b["open"], b["close"]) - b["low"]

    if body / candle_range < 0.10:
        return "Doji"

    if b["close"] > b["open"] and lower > body * 2:
        return "Hammer"

    if b["close"] < b["open"] and upper > body * 2:
        return "Shooting Star"

    if (
        a["close"] < a["open"]
        and b["close"] > b["open"]
        and b["close"] > a["open"]
        and b["open"] < a["close"]
    ):
        return "Bullish Engulfing"

    if (
        a["close"] > a["open"]
        and b["close"] < b["open"]
        and b["close"] < a["open"]
        and b["open"] > a["close"]
    ):
        return "Bearish Engulfing"

    return "No major pattern"


# ---------------- MARKET DATA ----------------

if market == "NSE":

    selected = st.sidebar.selectbox(
        "Choose NSE instrument",
        [
            "NIFTY 50",
            "BANKNIFTY",
            "RELIANCE",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "TCS",
            "SBIN",
            "ITC",
            "Other"
        ]
    )

    if selected == "Other":
        symbol = st.sidebar.text_input(
            "Enter NSE symbol",
            "TATAMOTORS"
        ).strip().upper()
    else:
        symbol = selected

    if symbol == "NIFTY 50":
        ticker = "^NSEI"
    elif symbol == "BANKNIFTY":
        ticker = "^NSEBANK"
    elif symbol.endswith(".NS") or symbol.startswith("^"):
        ticker = symbol
    else:
        ticker = symbol.replace(" ", "") + ".NS"

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["5 minutes", "15 minutes", "30 minutes", "1 hour", "Daily"]
    )

    settings = {
        "5 minutes": ("5m", "5d"),
        "15 minutes": ("15m", "1mo"),
        "30 minutes": ("30m", "1mo"),
        "1 hour": ("60m", "3mo"),
        "Daily": ("1d", "2y")
    }

    interval, period = settings[timeframe]

    with st.spinner("Loading NSE data..."):
        try:
            df = get_data(ticker, interval, period)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

    source = "Yahoo Finance - unofficial data"

else:

    st.subheader("MCX Commodity Analyzer")

    commodity = st.sidebar.selectbox(
        "Commodity",
        [
            "GOLD",
            "GOLDM",
            "SILVER",
            "SILVERM",
            "CRUDEOIL",
            "NATURALGAS",
            "COPPER",
            "Other"
        ]
    )

    uploaded = st.file_uploader(
        "Upload MCX OHLC CSV",
        type=["csv"]
    )

    if uploaded is None:
        st.warning("Upload an MCX CSV file to analyse commodity prices.")
        st.code(
            "DateTime,Open,High,Low,Close,Volume\n"
            "2026-11-06 09:00,12300,12350,12280,12320,1500"
        )
        st.stop()

    try:
        df = load_mcx_csv(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    symbol = commodity
    source = "Uploaded MCX CSV - verify data source and timestamp"


if df.empty or len(df) < 20:
    st.error("Insufficient market data. Try another symbol or timeframe.")
    st.stop()


# ---------------- ANALYSIS ----------------

df = indicators(df)

last = df.iloc[-1]
price = float(last["close"])
atr = float(last["ATR"]) if pd.notna(last["ATR"]) else 0

recent = df.tail(60)

support = float(recent["low"].min())
resistance = float(recent["high"].max())

bull = 0
bear = 0

if last["EMA9"] > last["EMA21"]:
    bull += 1
else:
    bear += 1

if last["close"] > last["EMA50"]:
    bull += 1
else:
    bear += 1

if last["MACD"] > last["MACD_SIGNAL"]:
    bull += 1
else:
    bear += 1

if last["RSI"] > 55:
    bull += 1
elif last["RSI"] < 45:
    bear += 1

if bull >= 3:
    trend = "BULLISH"
elif bear >= 3:
    trend = "BEARISH"
else:
    trend = "NEUTRAL / WAIT"


# ---------------- DISPLAY ----------------

st.divider()
st.subheader(f"{market}: {symbol}")

st.caption(f"Data source: {source}")
st.caption(f"Last candle: {df.index[-1]}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Last Price", f"{price:,.2f}")
c2.metric("RSI", f"{last['RSI']:.2f}")
c3.metric("Bullish score", f"{bull}/4")
c4.metric("Bearish score", f"{bear}/4")

st.subheader("Market Direction")
st.write(f"**{trend}**")

st.write(f"**Candlestick Pattern:** {pattern(df)}")

s1, s2 = st.columns(2)

s1.metric("Support", f"{support:,.2f}")
s2.metric("Resistance", f"{resistance:,.2f}")


# ---------------- PRICE CHART ----------------

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["open"],
    high=df["high"],
    low=df["low"],
    close=df["close"],
    name="Candles"
))

for name in ["EMA9", "EMA21", "EMA50", "EMA200"]:
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[name],
        name=name
    ))

fig.add_hline(
    y=support,
    line_dash="dash",
    annotation_text="Support"
)

fig.add_hline(
    y=resistance,
    line_dash="dash",
    annotation_text="Resistance"
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)


# ---------------- MACD CHART ----------------

st.subheader("MACD Analysis")

macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD"],
    name="MACD"
))

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD_SIGNAL"],
    name="Signal"
))

macd_fig.update_layout(
    height=300,
    template="plotly_dark"
)

st.plotly_chart(macd_fig, use_container_width=True)


# ---------------- TRADE SCENARIOS ----------------

st.subheader("Possible Trade Scenarios")

if atr > 0:

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Bullish scenario**")
        st.write(f"Reference: {price:.2f}")
        st.write(f"Stop Loss: {price - 1.5 * atr:.2f}")
        st.write(f"Target 1: {price + 2 * atr:.2f}")
        st.write(f"Target 2: {price + 3 * atr:.2f}")

    with col2:
        st.write("**Bearish scenario**")
        st.write(f"Reference: {price:.2f}")
        st.write(f"Stop Loss: {price + 1.5 * atr:.2f}")
        st.write(f"Target 1: {price - 2 * atr:.2f}")
        st.write(f"Target 2: {price - 3 * atr:.2f}")

else:
    st.warning("ATR unavailable. More candles are needed.")


# ---------------- RECENT DATA ----------------

st.subheader("Recent Market Data")

st.dataframe(
    df.tail(15).sort_index(ascending=False),
    use_container_width=True
)

st.warning(
    "This is a rule-based technical-analysis tool, not a validated "
    "prediction system. Targets and signals are illustrative only. "
    "Check live prices, contract expiry, liquidity and risk before trading."
)
