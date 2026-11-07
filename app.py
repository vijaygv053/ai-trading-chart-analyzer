import streamlit as st
from google import genai
from PIL import Image

st.set_page_config(
    page_title="AI Trading Chart Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Trading Chart Analyzer")
st.caption("Analyze charts, patterns, support, resistance and possible moves.")

st.sidebar.header("Trading Settings")

market = st.sidebar.selectbox(
    "Select Market",
    ["Indian Stocks", "NIFTY", "BANKNIFTY",
     "US Stocks", "Forex", "Crypto",
     "Gold", "Silver", "Commodities", "Other"]
)

symbol = st.sidebar.text_input(
    "Trading Symbol",
    placeholder="Example: NIFTY, BTCUSDT"
)

timeframe = st.sidebar.selectbox(
    "Chart Timeframe",
    ["1 minute", "3 minutes", "5 minutes",
     "15 minutes", "30 minutes", "1 hour",
     "4 hours", "Daily", "Weekly", "Monthly"]
)

style = st.sidebar.selectbox(
    "Trading Style",
    ["Scalping", "Intraday", "Overnight",
     "Swing Trading", "Positional Trading"]
)

st.subheader("Upload Your Trading Chart")

uploaded_file = st.file_uploader(
    "Upload chart screenshot or photo",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Trading Chart",
             use_container_width=True)

    if st.button("🔍 ANALYZE MY CHART", type="primary"):
        if not symbol.strip():
            st.warning("Please enter a trading symbol.")
        else:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                client = genai.Client(api_key=api_key)

                prompt = f"""
You are an advanced technical chart analysis assistant.

Analyze this uploaded trading chart.

Market: {market}
Symbol: {symbol}
Timeframe: {timeframe}
Trading style: {style}

Provide a detailed analysis with these sections:

1. Overall trend: Bullish, Bearish or Sideways.
2. Chart patterns detected.
3. Candlestick patterns detected.
4. Support levels S1, S2 and S3.
5. Resistance levels R1, R2 and R3.
6. Supply and demand zones.
7. Breakout and breakdown levels.
8. MACD, RSI, EMA and volume observations
   only when visible or supported by data.
9. Possible next upward movement.
10. Possible next downward movement.
11. Possible entry zone and confirmation conditions.
12. Stop loss and targets, if price levels
    are readable.
13. Pattern formation: Forming or Confirmed.
14. Reasons supporting your analysis.
15. What would invalidate the analysis.

IMPORTANT:
- Never invent price levels.
- If chart prices are unreadable, say so.
- Do not invent indicator values.
- Do not force a chart pattern when none exists.
- Give both bullish and bearish scenarios.
- Clearly explain uncertainty.
- Do not guarantee profits.
- Distinguish visual estimates from verified data.

Format the answer with clear headings and tables.
"""

                with st.spinner("AI is analyzing your chart..."):
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=[prompt, image]
                    )

                st.success("Chart analysis completed!")

                st.subheader("📊 AI Analysis Results")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"Analysis error: {e}")

st.divider()
st.caption(
    "For educational purposes only. "
    "Always verify signals before trading."
)
