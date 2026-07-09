"""
IGNITE - Candlestick Reversal / Continuation Signal PoC
=========================================================
A single-file Streamlit application that fetches recent OHLCV data via
yfinance, analyzes the anatomy of the most recently completed candle
(body, upper wick, lower wick, volume behaviour) and produces a simple
directional call: UP, DOWN, or HOLD.

This is a Proof of Concept for educational purposes only.
It is NOT financial advice and should not be used to make real trading
decisions.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deployable as-is to a Hugging Face Space (Streamlit SDK).
"""

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="IGNITE — Candlestick Signal PoC",
    page_icon="🔥",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
TICKERS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "NVIDIA (NVDA)": "NVDA",
    "EUR/USD (EURUSD=X)": "EURUSD=X",
    "S&P 500 ETF (SPY)": "SPY",
    "Gold Futures (GC=F)": "GC=F",
}

INTERVALS = {
    "1 minute": ("1m", "1d"),
    "5 minutes": ("5m", "5d"),
    "15 minutes": ("15m", "5d"),
}

# How many prior candles to use when computing the "average" volume that the
# latest candle's volume is compared against.
VOLUME_LOOKBACK = 10

# Thresholds — tunable "knobs" for the rule engine.
LONG_WICK_RATIO = 0.45       # wick must be >= 45% of the candle's total range
SMALL_OPPOSITE_WICK_RATIO = 0.15  # the "closing near the extreme" wick must be small
VOLUME_DROP_RATIO = 0.85     # current volume < 85% of recent average -> "decreasing"
VOLUME_SPIKE_RATIO = 1.25    # current volume > 125% of recent average -> "high"


# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class CandleMetrics:
    open: float
    high: float
    low: float
    close: float
    volume: float
    body: float
    upper_wick: float
    lower_wick: float
    total_range: float
    is_red: bool
    is_green: bool
    body_pct: float
    upper_wick_pct: float
    lower_wick_pct: float
    avg_prior_volume: float
    volume_ratio: float


@dataclass
class Signal:
    direction: str          # "UP", "DOWN", "HOLD"
    confidence: str         # "High", "Medium", "Low"
    reasons: List[str]


# ----------------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return df

    # yfinance sometimes returns a MultiIndex column set for single tickers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.reset_index()
    # The index column name varies ("Datetime" or "Date") depending on interval.
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "Time"})
    return df


# ----------------------------------------------------------------------------
# Core IGNITE analysis
# ----------------------------------------------------------------------------
def compute_candle_metrics(df: pd.DataFrame) -> CandleMetrics:
    """Compute body/wick/volume metrics for the most recently completed candle."""
    last = df.iloc[-1]
    o, h, l, c, v = (
        float(last["Open"]),
        float(last["High"]),
        float(last["Low"]),
        float(last["Close"]),
        float(last["Volume"]),
    )

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = max(h - l, 1e-9)  # avoid div-by-zero on flat candles

    is_red = c < o
    is_green = c > o

    prior = df.iloc[:-1].tail(VOLUME_LOOKBACK)
    avg_prior_volume = float(prior["Volume"].mean()) if not prior.empty else v
    volume_ratio = v / avg_prior_volume if avg_prior_volume > 0 else 1.0

    return CandleMetrics(
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        body=body,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        total_range=total_range,
        is_red=is_red,
        is_green=is_green,
        body_pct=body / total_range * 100,
        upper_wick_pct=upper_wick / total_range * 100,
        lower_wick_pct=lower_wick / total_range * 100,
        avg_prior_volume=avg_prior_volume,
        volume_ratio=volume_ratio,
    )


def run_ignite_algorithm(m: CandleMetrics) -> Signal:
    """
    IGNITE ruleset (PoC-level heuristics — not a robust trading strategy):

    Rule 1 (Bullish reversal / "UP"):
        Red candle, long lower wick (rejection of lower prices), and
        volume is decreasing vs recent average (selling pressure fading).

    Rule 2 (Bearish continuation / "DOWN"):
        Red candle, closes near its low (small lower wick, large body),
        and volume is elevated vs recent average (strong conviction selling).

    Rule 3 (Bearish reversal / "UP" mirror... actually bullish exhaustion):
        Green candle, long upper wick (rejection of higher prices), and
        volume is decreasing (buying pressure fading) -> lean "DOWN".

    Rule 4 (Bullish continuation / "DOWN" mirror):
        Green candle, closes near its high, high volume -> lean "UP".

    Anything else -> HOLD / Neutral.
    """
    reasons: List[str] = []

    volume_decreasing = m.volume_ratio < VOLUME_DROP_RATIO
    volume_high = m.volume_ratio > VOLUME_SPIKE_RATIO

    # --- Rule 1: Red candle, long lower wick, decreasing volume -> UP -------
    if m.is_red and m.lower_wick_pct >= LONG_WICK_RATIO * 100 and volume_decreasing:
        reasons.append(
            f"Candle is RED (close below open), but the lower wick is "
            f"{m.lower_wick_pct:.1f}% of the total range — buyers stepped in "
            f"and rejected the lows (hammer-like rejection)."
        )
        reasons.append(
            f"Volume is {(1 - m.volume_ratio) * 100:.1f}% below the "
            f"{VOLUME_LOOKBACK}-candle average, suggesting selling pressure "
            f"is fading rather than accelerating."
        )
        reasons.append(
            f"Upper wick is only {m.upper_wick_pct:.1f}% of the range and "
            f"body is {m.body_pct:.1f}%, consistent with a potential bullish reversal."
        )
        return Signal("UP", "Medium", reasons)

    # --- Rule 2: Strong red candle near low, high volume -> DOWN -----------
    if (
        m.is_red
        and m.lower_wick_pct <= SMALL_OPPOSITE_WICK_RATIO * 100
        and m.body_pct >= 50
        and volume_high
    ):
        reasons.append(
            f"Candle is RED and closed very near its low — lower wick is only "
            f"{m.lower_wick_pct:.1f}% of the total range, meaning sellers stayed "
            f"in control through the close."
        )
        reasons.append(
            f"Body accounts for {m.body_pct:.1f}% of the total range — a strong, "
            f"decisive move in one direction."
        )
        reasons.append(
            f"Volume is {(m.volume_ratio - 1) * 100:.1f}% above the "
            f"{VOLUME_LOOKBACK}-candle average, indicating high conviction "
            f"behind the sell-off."
        )
        return Signal("DOWN", "High", reasons)

    # --- Rule 3: Green candle, long upper wick, decreasing volume -> DOWN --
    if m.is_green and m.upper_wick_pct >= LONG_WICK_RATIO * 100 and volume_decreasing:
        reasons.append(
            f"Candle is GREEN, but the upper wick is {m.upper_wick_pct:.1f}% of "
            f"the range — price pushed higher then got rejected, a sign of "
            f"fading bullish momentum (shooting-star-like)."
        )
        reasons.append(
            f"Volume is {(1 - m.volume_ratio) * 100:.1f}% below the recent "
            f"average, meaning buyers are losing conviction."
        )
        return Signal("DOWN", "Medium", reasons)

    # --- Rule 4: Strong green candle near high, high volume -> UP ----------
    if (
        m.is_green
        and m.upper_wick_pct <= SMALL_OPPOSITE_WICK_RATIO * 100
        and m.body_pct >= 50
        and volume_high
    ):
        reasons.append(
            f"Candle is GREEN and closed very near its high — upper wick is only "
            f"{m.upper_wick_pct:.1f}% of the range, meaning buyers stayed in "
            f"control through the close."
        )
        reasons.append(
            f"Body accounts for {m.body_pct:.1f}% of the total range — a strong, "
            f"decisive up-move."
        )
        reasons.append(
            f"Volume is {(m.volume_ratio - 1) * 100:.1f}% above the "
            f"{VOLUME_LOOKBACK}-candle average, confirming strong buying interest."
        )
        return Signal("UP", "High", reasons)

    # --- Fallback: no rule fired -> HOLD ------------------------------------
    reasons.append(
        f"No high-confidence pattern detected. Body is {m.body_pct:.1f}% of range, "
        f"upper wick {m.upper_wick_pct:.1f}%, lower wick {m.lower_wick_pct:.1f}%."
    )
    reasons.append(
        f"Volume is running at {m.volume_ratio * 100:.0f}% of the "
        f"{VOLUME_LOOKBACK}-candle average — not enough of an extreme to signal "
        f"a directional edge."
    )
    return Signal("HOLD", "Low", reasons)


# ----------------------------------------------------------------------------
# Charting
# ----------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, m: CandleMetrics, signal: Signal) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["Time"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Highlight the analyzed (last) candle with a shaded vertical band + annotation.
    last_time = df["Time"].iloc[-1]
    color_map = {"UP": "#26a69a", "DOWN": "#ef5350", "HOLD": "#f2b705"}
    highlight_color = color_map.get(signal.direction, "#f2b705")

    fig.add_vrect(
        x0=last_time,
        x1=last_time,
        line_width=6,
        line_color=highlight_color,
        opacity=0.5,
    )

    fig.add_annotation(
        x=last_time,
        y=m.high,
        text=f"Analyzed candle → {signal.direction}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=highlight_color,
        font=dict(color=highlight_color, size=12),
        yshift=15,
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def render_badge(signal: Signal) -> None:
    palette = {
        "UP": ("#0e7c5f", "🟢 UP"),
        "DOWN": ("#8f1e1e", "🔴 DOWN"),
        "HOLD": ("#8a6d1a", "🟡 HOLD"),
    }
    bg_color, label = palette.get(signal.direction, ("#444444", signal.direction))

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding:22px;
            border-radius:14px;
            text-align:center;
            margin-bottom:10px;
        ">
            <div style="font-size:14px; color:#eeeeee; letter-spacing:2px;">
                IGNITE SIGNAL
            </div>
            <div style="font-size:40px; font-weight:800; color:#ffffff; margin-top:4px;">
                {label}
            </div>
            <div style="font-size:14px; color:#dddddd; margin-top:4px;">
                Confidence: {signal.confidence}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_row(m: CandleMetrics) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Body %", f"{m.body_pct:.1f}%")
    c2.metric("Upper Wick %", f"{m.upper_wick_pct:.1f}%")
    c3.metric("Lower Wick %", f"{m.lower_wick_pct:.1f}%")
    c4.metric("Volume vs Avg", f"{m.volume_ratio * 100:.0f}%")


# ----------------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------------
def main() -> None:
    st.title("🔥 IGNITE — Candlestick Signal PoC")
    st.caption(
        "Proof of concept only — analyzes body/wick/volume anatomy of the most "
        "recent candle to flag a directional lean. **Not financial advice.**"
    )

    with st.sidebar:
        st.header("Settings")
        ticker_label = st.selectbox("Ticker", list(TICKERS.keys()), index=0)
        ticker = TICKERS[ticker_label]

        interval_label = st.selectbox("Interval", list(INTERVALS.keys()), index=0)
        interval, period = INTERVALS[interval_label]

        refresh = st.button("🔄 Refresh Data", use_container_width=True)

        st.divider()
        st.caption(
            "Rules use fixed thresholds (wick ≥ 45% of range for a rejection "
            "signal; volume ±15-25% vs a 10-candle average). This is a "
            "simplified heuristic engine, not a backtested strategy."
        )

    if refresh:
        fetch_data.clear()

    with st.spinner(f"Fetching {ticker} @ {interval_label} data..."):
        df = fetch_data(ticker, interval, period)

    if df is None or df.empty or len(df) < VOLUME_LOOKBACK + 2:
        st.error(
            "Not enough data returned for this ticker/interval combination. "
            "Try a different interval (e.g. 5 minutes) or ticker — intraday "
            "1-minute data is only available for the last few trading days "
            "and may be limited outside market hours for equities."
        )
        return

    metrics = compute_candle_metrics(df)
    signal = run_ignite_algorithm(metrics)

    last_row = df.iloc[-1]
    st.subheader(f"{ticker_label} — Last candle: {last_row['Time']}")

    left, right = st.columns([1, 2.2])

    with left:
        render_badge(signal)
        st.markdown("**Why IGNITE made this call:**")
        for reason in signal.reasons:
            st.markdown(f"- {reason}")

        st.divider()
        st.markdown("**Raw candle values**")
        st.write(
            {
                "Open": round(metrics.open, 5),
                "High": round(metrics.high, 5),
                "Low": round(metrics.low, 5),
                "Close": round(metrics.close, 5),
                "Volume": int(metrics.volume),
                "Avg Prior Volume": round(metrics.avg_prior_volume, 2),
            }
        )

    with right:
        render_metrics_row(metrics)
        fig = build_chart(df.tail(60), metrics, signal)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ IGNITE is a rule-based heuristic PoC built for demonstration purposes. "
        "It does not account for broader market context, news, or multi-timeframe "
        "confluence. Do not use it as the sole basis for real trading decisions."
    )


if __name__ == "__main__":
    main()
