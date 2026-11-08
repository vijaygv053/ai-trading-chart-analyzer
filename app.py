import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="India NSE & MCX Analyzer", page_icon="📈", layout="wide")
st.title("🇮🇳 Indian NSE & MCX Multi-Timeframe Analyzer")
st.caption("Technical ratings from 1 minute to 1 month | Free-first data | Analysis only")
st.info("NSE data uses Yahoo Finance (unofficial; availability and delay vary). MCX analysis uses your uploaded OHLC CSV. Refreshing does not guarantee a new exchange quote.")
st_autorefresh(interval=300000, key="refresh_5_minutes")
st.sidebar.caption("Dashboard refresh: every 5 minutes")

TIMEFRAMES = [
    "1 minute", "2 minutes", "3 minutes", "5 minutes",
    "10 minutes", "15 minutes", "30 minutes", "45 minutes",
    "1 hour", "2 hours", "4 hours",
    "Daily", "Weekly", "Monthly"
]

RESAMPLE_RULES = {
    "1 minute": None,
    "2 minutes": "2min",
    "3 minutes": "3min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "45 minutes": "45min",
    "1 hour": "1h",
    "2 hours": "2h",
    "4 hours": "4h",
    "Daily": "1D",
    "Weekly": "1W",
    "Monthly": "1ME"
}

@st.cache_data(ttl=240, show_spinner=False)
def fetch_yahoo(ticker, interval, period):
    d = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if d is None or d.empty:
        return pd.DataFrame()

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d.columns = [
        str(c).lower().replace(" ", "_")
        for c in d.columns
    ]

    required = ["open", "high", "low", "close"]

    if not all(c in d.columns for c in required):
        return pd.DataFrame()

    if "volume" not in d.columns:
        d["volume"] = 0

    d = d[required + ["volume"]].dropna(
        subset=required
    ).copy()

    d.index = pd.to_datetime(d.index)
    return d.sort_index()


def load_csv(upload):
    raw = pd.read_csv(upload)

    raw.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in raw.columns
    ]

    dt_col = next(
        (c for c in ["datetime", "date_time", "timestamp", "date"]
         if c in raw.columns),
        None
    )

    if dt_col is None:
        raise ValueError(
            "CSV needs Date, DateTime, Timestamp or date_time column."
        )

    if dt_col == "date" and "time" in raw.columns:
        idx = pd.to_datetime(
            raw["date"].astype(str) + " " + raw["time"].astype(str),
            errors="coerce",
            dayfirst=True
        )
    else:
        idx = pd.to_datetime(
            raw[dt_col],
            errors="coerce",
            dayfirst=True
        )

    aliases = {
        "open": ["open", "open_price"],
        "high": ["high", "high_price"],
        "low": ["low", "low_price"],
        "close": ["close", "close_price", "ltp", "last"],
        "volume": ["volume", "vol", "qty", "quantity"]
    }

    out = {}

    for key, names in aliases.items():
        col = next((n for n in names if n in raw.columns), None)
        if col:
            out[key] = pd.to_numeric(raw[col], errors="coerce")

    missing = [
        x for x in ["open", "high", "low", "close"]
        if x not in out
    ]

    if missing:
        raise ValueError(
            "CSV missing columns: " + ", ".join(missing)
        )

    df = pd.DataFrame(out, index=idx)

    if "volume" not in df:
        df["volume"] = 0

    df.index.name = "datetime"

    return (
        df[~df.index.isna()]
        .dropna(subset=["open", "high", "low", "close"])
        .sort_index()
    )


def resample_ohlc(df, rule):
    if df.empty or rule is None:
        return df.copy()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    result = df.resample(
        rule,
        label="right",
        closed="right"
    ).agg(agg)

    return result.dropna(
        subset=["open", "high", "low", "close"]
    )


def indicators(df):
    x = df.copy()
    c = x["close"]

    x["EMA9"] = c.ewm(span=9, adjust=False).mean()
    x["EMA21"] = c.ewm(span=21, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    ag = gain.ewm(alpha=1/14, adjust=False).mean()
    al = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = ag / al.replace(0, np.nan)
    x["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    x["MACD"] = (
        c.ewm(span=12, adjust=False).mean()
        - c.ewm(span=26, adjust=False).mean()
    )

    x["MACD_signal"] = x["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    prev = c.shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev).abs(),
            (x["low"] - prev).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATR"] = tr.rolling(14).mean()

    return x


def get_rating(df):
    if len(df) < 30:
        return {
            "Rating": "Insufficient data",
            "Buy": 0,
            "Sell": 0,
            "RSI": np.nan,
            "Close": float(df.close.iloc[-1]) if len(df) else np.nan,
            "Candles": len(df)
        }

    x = indicators(df)
    z = x.iloc[-1]

    checks = [
        z.EMA9 > z.EMA21,
        z.close > z.EMA50,
        z.MACD > z.MACD_signal,
        z.RSI > 55,
        z.close > z.EMA200
    ]

    buy = sum(bool(v) for v in checks)
    sell = len(checks) - buy

    if buy >= 4:
        rating = "Strong Buy"
    elif buy == 3:
        rating = "Buy"
    elif sell >= 4:
        rating = "Strong Sell"
    elif sell == 3:
        rating = "Sell"
    else:
        rating = "Neutral"

    return {
        "Rating": rating,
        "Buy": buy,
        "Sell": sell,
        "RSI": float(z.RSI),
        "Close": float(z.close),
        "Candles": len(df)
    }


def candle_pattern(df):
    if len(df) < 2:
        return "Not enough candles"

    p = df.iloc[-2]
    c = df.iloc[-1]

    body = abs(c.close - c.open)
    span = max(c.high - c.low, 1e-9)

    upper = c.high - max(c.open, c.close)
    lower = min(c.open, c.close) - c.low

    if body / span < 0.1:
        return "Doji"

    if c.close > c.open and lower > body * 2 and upper < body:
        return "Hammer"

    if c.close < c.open and upper > body * 2 and lower < body:
        return "Shooting Star"

    if (
        p.close < p.open
        and c.close > c.open
        and c.open <= p.close
        and c.close >= p.open
    ):
        return "Bullish Engulfing"

    if (
        p.close > p.open
        and c.close < c.open
        and c.open >= p.close
        and c.close <= p.open
    ):
        return "Bearish Engulfing"

    return "No basic pattern detected"


# MARKET SELECTION

market = st.sidebar.radio(
    "Choose market",
    ["NSE", "MCX"]
)

source = ""

if market == "NSE":

    st.sidebar.subheader("NSE instrument")

    quick = st.sidebar.selectbox(
        "Quick select",
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

    raw = (
        st.sidebar.text_input(
            "NSE symbol (example: TATAMOTORS)",
            "TATAMOTORS"
        )
        if quick == "Other"
        else quick
    )

    symbol = raw.strip().upper().replace(" ", "")

    ticker = (
        "^NSEI"
        if symbol in ["NIFTY", "NIFTY50"]
        else "^NSEBANK"
        if symbol in ["BANKNIFTY", "NIFTYBANK"]
        else (
            symbol
            if symbol.startswith("^") or symbol.endswith(".NS")
            else symbol + ".NS"
        )
    )

    with st.spinner("Loading NSE candles for multiple timeframes..."):

        try:
            one_min = fetch_yahoo(ticker, "1m", "7d")
            daily = fetch_yahoo(ticker, "1d", "5y")
            weekly = fetch_yahoo(ticker, "1wk", "10y")
            monthly = fetch_yahoo(ticker, "1mo", "max")

        except Exception as e:
            st.error(f"Yahoo Finance data error: {e}")
            st.stop()

    frames = {}

    for tf, rule in RESAMPLE_RULES.items():

        if tf == "Daily":
            frames[tf] = daily

        elif tf == "Weekly":
            frames[tf] = weekly

        elif tf == "Monthly":
            frames[tf] = monthly

        else:
            frames[tf] = resample_ohlc(one_min, rule)

    instrument = ticker

    source = (
        "Yahoo Finance unofficial feed; "
        "data delay and availability vary"
    )

    if one_min.empty and daily.empty:
        st.error(
            "No data returned. Try another symbol or check Yahoo Finance availability."
        )
        st.stop()

else:

    st.sidebar.subheader("MCX commodity CSV")

    st.write(
        "Upload an MCX OHLC CSV. The app will resample your uploaded "
        "candles into the available timeframes. Short timeframes depend "
        "on the interval of your original CSV."
    )

    uploaded = st.file_uploader(
        "Upload MCX OHLC CSV",
        type=["csv"]
    )

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

    if uploaded is None:
        st.warning("Upload an MCX OHLC CSV to continue.")

        st.code(
            "DateTime,Open,High,Low,Close,Volume\n"
            "2026-11-06 09:00,12300,12350,12280,12320,1500"
        )

        st.stop()

    try:
        base = load_csv(uploaded)

    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    frames = {
        tf: resample_ohlc(base, rule)
        for tf, rule in RESAMPLE_RULES.items()
    }

    instrument = commodity
    source = "User-uploaded MCX CSV; verify source and timestamp"


# MULTI-TIMEFRAME SUMMARY

st.subheader(
    f"{market} Technical Summary — {instrument}"
)

st.caption(f"Data source: {source}")

rows = []

for tf in TIMEFRAMES:

    r = get_rating(
        frames.get(tf, pd.DataFrame())
    )

    rows.append({
        "Timeframe": tf,
        "Overall Rating": r["Rating"],
        "Buy Signals": r["Buy"],
        "Sell Signals": r["Sell"],
        "RSI (14)": (
            round(r["RSI"], 1)
            if pd.notna(r["RSI"])
            else "—"
        ),
        "Candles": r["Candles"]
    })

summary = pd.DataFrame(rows)

st.dataframe(
    summary,
    hide_index=True,
    use_container_width=True
)


# SELECTED TIMEFRAME

st.subheader("Detailed timeframe analysis")

selected_tf = st.selectbox(
    "Choose timeframe to view chart and trade levels",
    TIMEFRAMES,
    index=3
)

df = frames.get(
    selected_tf,
    pd.DataFrame()
)

if df.empty or len(df) < 20:

    st.warning(
        f"Not enough candles available for {selected_tf}. "
        "This can happen with free data or a short MCX CSV. "
        "Try a higher timeframe or a longer-history data source."
    )

    st.stop()

x = indicators(df)

last = x.iloc[-1]

price = float(last.close)

atr = (
    float(last.ATR)
    if pd.notna(last.ATR)
    else 0.0
)

recent = x.tail(min(60, len(x)))

support = float(recent.low.min())
resistance = float(recent.high.max())

rating = get_rating(df)

st.caption(
    f"Selected: {selected_tf} | "
    f"Last candle: {df.index[-1]} | {source}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Last close", f"{price:,.2f}")
c2.metric("Overall rating", rating["Rating"])
c3.metric("RSI (14)", f"{last.RSI:.1f}")

c4.metric(
    "ATR (14)",
    f"{atr:,.2f}" if atr else "N/A"
)

p1, p2 = st.columns(2)

p1.metric(
    "Recent support",
    f"{support:,.2f}"
)

p2.metric(
    "Recent resistance",
    f"{resistance:,.2f}"
)

st.write(
    "**Last candle pattern:**",
    candle_pattern(df)
)


# CANDLESTICK CHART

fig = go.Figure(
    go.Candlestick(
        x=x.index,
        open=x.open,
        high=x.high,
        low=x.low,
        close=x.close,
        name="Price"
    )
)

for col in ["EMA9", "EMA21", "EMA50", "EMA200"]:

    fig.add_trace(
        go.Scatter(
            x=x.index,
            y=x[col],
            mode="lines",
            name=col
        )
    )

fig.add_hline(
    y=support,
    line_dash="dash",
    annotation_text="Recent support"
)

fig.add_hline(
    y=resistance,
    line_dash="dash",
    annotation_text="Recent resistance"
)

fig.update_layout(
    title=f"{instrument} — {selected_tf}",
    height=580,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# MACD CHART

macd = go.Figure()

macd.add_trace(
    go.Scatter(
        x=x.index,
        y=x.MACD,
        name="MACD"
    )
)

macd.add_trace(
    go.Scatter(
        x=x.index,
        y=x.MACD_signal,
        name="Signal"
    )
)

macd.update_layout(
    title="MACD",
    height=250,
    template="plotly_dark"
)

st.plotly_chart(
    macd,
    use_container_width=True
)


# TRADE SCENARIOS

st.subheader("Illustrative trade scenarios")

if atr > 0:

    a, b = st.columns(2)

    with a:

        st.markdown(
            "**Bullish scenario — wait for confirmation**"
        )

        st.write(
            f"Reference: {price:,.2f} | "
            f"Stop example: {price - 1.5 * atr:,.2f} | "
            f"Target 1: {price + 2 * atr:,.2f} | "
            f"Target 2: {price + 3 * atr:,.2f}"
        )

    with b:

        st.markdown(
            "**Bearish scenario — wait for confirmation**"
        )

        st.write(
            f"Reference: {price:,.2f} | "
            f"Stop example: {price + 1.5 * atr:,.2f} | "
            f"Target 1: {price - 2 * atr:,.2f} | "
            f"Target 2: {price - 3 * atr:,.2f}"
        )


# LATEST CANDLES

st.subheader("Latest candles")

st.dataframe(
    x[
        [
            "open", "high", "low", "close", "volume",
            "EMA9", "EMA21", "EMA50", "RSI", "MACD"
        ]
    ].tail(15).sort_index(ascending=False),
    use_container_width=True
)

st.warning(
    "Ratings are simple rule-based indicator counts, not validated "
    "predictions. Free Yahoo Finance 1-minute history is limited and "
    "may be unavailable. Resampled short intervals depend on those "
    "1-minute candles. MCX short intervals depend on the resolution "
    "of your uploaded CSV. Levels are illustrative, not trade instructions."
    )
