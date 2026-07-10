"""
IGNITE Hub — Global Market Intelligence
=============================================================
A unified, production-grade Streamlit application combining
Deep Macro Momentum Analysis and Candlestick Reversal Signal Engines.
Featuring a universal text search box for all yfinance assets.

Run:
    streamlit run app.py
"""

from __future__ import annotations
import textwrap
from typing import Optional, List
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IGNITE Hub — Global Search Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS (Light & Dark Theme Safe)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] * { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── KPI cards ────────────────────────────────────────────────────────── */
.kpi-card {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem 0.9rem;
    margin-bottom: 0.5rem;
    background: rgba(128,128,128,0.04);
}
.kpi-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    opacity: 0.5;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.7rem;
    opacity: 0.45;
    margin-top: 0.25rem;
}

/* ── Signal card ──────────────────────────────────────────────────────── */
.signal-card {
    border-radius: 14px;
    padding: 1.5rem 1.7rem;
    margin: 0.6rem 0 1.2rem;
    border: 1px solid transparent;
}
.sig-green  { background: rgba(22,163,74,0.12);  border-color: rgba(22,163,74,0.35); }
.sig-yellow { background: rgba(202,138,4,0.10);  border-color: rgba(202,138,4,0.35); }
.sig-red    { background: rgba(220,38,38,0.10);  border-color: rgba(220,38,38,0.35); }

.sig-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.28rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}
.badge-green  { background: rgba(22,163,74,0.2);  color: #16a34a; }
.badge-yellow { background: rgba(202,138,4,0.18); color: #b45309; }
.badge-red    { background: rgba(220,38,38,0.18); color: #dc2626; }

.sig-headline {
    font-size: 1.18rem;
    font-weight: 700;
    margin: 0 0 0.45rem;
}
.sig-body {
    font-size: 0.86rem;
    line-height: 1.65;
    opacity: 0.75;
    max-width: 680px;
}

/* ── Section divider label ────────────────────────────────────────────── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    opacity: 0.4;
    margin: 1.6rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(128,128,128,0.15);
}

/* ── Profile table ────────────────────────────────────────────────────── */
.profile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 0.6rem;
    margin-top: 0.3rem;
}
.profile-item {
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 9px;
    padding: 0.65rem 0.9rem;
    background: rgba(128,128,128,0.04);
}
.profile-key {
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.4;
    margin-bottom: 0.2rem;
}
.profile-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Summary box ──────────────────────────────────────────────────────── */
.summary-box {
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 9px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    line-height: 1.7;
    opacity: 0.75;
    margin-top: 0.4rem;
    max-height: 120px;
    overflow-y: auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES & CONFIG CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CandleMetrics:
    open: float; high: float; low: float; close: float; volume: float
    body: float; upper_wick: float; lower_wick: float; total_range: float
    is_red: bool; is_green: bool; body_pct: float; upper_wick_pct: float
    lower_wick_pct: float; avg_prior_volume: float; volume_ratio: float

@dataclass
class Signal:
    direction: str; confidence: str; reasons: List[str]

INTERVALS = {
    "5 minutes": ("5m", "5d"),
    "15 minutes": ("15m", "5d"),
    "Hourly": ("1h", "7d"),
    "Daily (Default Macro)": ("1d", "120d")
}

VOLUME_LOOKBACK = 10
LONG_WICK_RATIO = 0.45
SMALL_OPPOSITE_WICK_RATIO = 0.15
VOLUME_DROP_RATIO = 0.85
VOLUME_SPIKE_RATIO = 1.25

# ══════════════════════════════════════════════════════════════════════════════
# CORE API UTILITY FUNCTIONS (GLOBAL LIVE API FETCHING)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def search_symbols(query: str, max_results: int = 10) -> List[dict]:
    """Queries the live yfinance search dictionary index using plain text."""
    query = (query or "").strip()
    if len(query) < 1: return []
    try:
        result = yf.Search(query, max_results=max_results)
        quotes = result.quotes or []
        return [{"symbol": q.get("symbol"), "name": q.get("shortname") or q.get("longname") or q.get("symbol")} for q in quotes if q.get("symbol")]
    except Exception:
        return []

@st.cache_data(ttl=60, show_spinner=False)
def fetch_raw_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(tickers=ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    return df.rename(columns={time_col: "Time"})

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(ticker_symbol: str) -> dict:
    try: return yf.Ticker(ticker_symbol).info
    except: return {}

def calculate_ema(series: pd.Series, span: int) -> pd.Series: return series.ewm(span=span, adjust=False).mean()
def calculate_sma(series: pd.Series, window: int) -> pd.Series: return series.rolling(window=window).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))

def format_market_cap(cap: Optional[float]) -> tuple[str, str]:
    if cap is None: return "N/A", "Unknown"
    if cap >= 1e12: return f"${cap / 1e12:.2f} T", "Large-cap"
    if cap >= 1e9:
        val = cap / 1e9
        return f"${val:.2f} B", "Large-cap" if val >= 10 else ("Mid-cap" if val >= 2 else "Small-cap")
    return f"${cap:,.0f}", "Micro-cap"

# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHMIC RULE ENGINES
# ══════════════════════════════════════════════════════════════════════════════
def momentum_signal(price: float, ema20: float, rsi: float) -> dict:
    above_ema = price > ema20
    if above_ema and (40 <= rsi <= 65):
        return {
            "card_class": "sig-green", "badge_class": "badge-green", "badge_text": "🟢 BULLISH MOMENTUM",
            "headline": "Strong Upward Momentum Detected",
            "body": "Trading above EMA-20 with healthy RSI (40–65). Buyers control the structural trend safely."
        }
    elif rsi > 70:
        return {
            "card_class": "sig-red", "badge_class": "badge-red", "badge_text": "🔴 OVERBOUGHT WARNING",
            "headline": "Caution — Momentum Is Overextended",
            "body": f"RSI has touched {rsi:.1f}. High probability of technical cooling or profit booking execution."
        }
    elif rsi < 30 or not above_ema:
        return {
            "card_class": "sig-red", "badge_class": "badge-red", "badge_text": "🔴 BEARISH STRUCTURAL TREND",
            "headline": "Weak or Negative Trend Conditions",
            "body": "Asset trading under pressure below EMA-20 or in deeply weak oversold conditions."
        }
    else:
        return {
            "card_class": "sig-yellow", "badge_class": "badge-yellow", "badge_text": "🟡 NEUTRAL CONSOLIDATION",
            "headline": "Mixed Signals — No Clean Bias",
            "body": f"RSI sits at {rsi:.1f} in transition zone. Market consolidation structure remains in active play."
        }

def compute_candle_metrics(df: pd.DataFrame) -> CandleMetrics:
    last = df.iloc[-1]
    o, h, l, c, v = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"]), float(last["Volume"])
    body = abs(c - o)
    upper_wick, lower_wick = h - max(o, c), min(o, c) - l
    total_range = max(h - l, 1e-9)
    prior = df.iloc[:-1].tail(VOLUME_LOOKBACK)
    avg_prior_volume = float(prior["Volume"].mean()) if not prior.empty else v
    return CandleMetrics(
        open=o, high=h, low=l, close=c, volume=v, body=body, upper_wick=upper_wick, lower_wick=lower_wick,
        total_range=total_range, is_red=c < o, is_green=c > o, body_pct=(body/total_range)*100,
        upper_wick_pct=(upper_wick/total_range)*100, lower_wick_pct=(lower_wick/total_range)*100,
        avg_prior_volume=avg_prior_volume, volume_ratio=v/avg_prior_volume if avg_prior_volume > 0 else 1.0
    )

def run_ignite_algorithm(m: CandleMetrics) -> Signal:
    reasons = []
    vol_dec, vol_high = m.volume_ratio < VOLUME_DROP_RATIO, m.volume_ratio > VOLUME_SPIKE_RATIO
    if m.is_red and m.lower_wick_pct >= LONG_WICK_RATIO * 100 and vol_dec:
        reasons.extend([f"Red candle with massive lower wick ({m.lower_wick_pct:.1f}%) rejecting structural lows.", "Volume decreasing, confirming low selling conviction."])
        return Signal("UP", "Medium", reasons)
    if m.is_red and m.lower_wick_pct <= SMALL_OPPOSITE_WICK_RATIO * 100 and m.body_pct >= 50 and vol_high:
        reasons.extend(["Decisive flat red close near low with high body distribution dominance.", "Elevated high-conviction institutional distribution volume."])
        return Signal("DOWN", "High", reasons)
    if m.is_green and m.upper_wick_pct >= LONG_WICK_RATIO * 100 and vol_dec:
        reasons.extend([f"Green body structure rejected heavily at highs (Wick: {m.upper_wick_pct:.1f}%).", "Fading buying momentum liquidity profile."])
        return Signal("DOWN", "Medium", reasons)
    if m.is_green and m.upper_wick_pct <= SMALL_OPPOSITE_WICK_RATIO * 100 and m.body_pct >= 50 and vol_high:
        reasons.extend(["Clean marubozu expansion closing near highs securely.", "High accumulation volume transaction footprints detected."])
        return Signal("UP", "High", reasons)
    return Signal("HOLD", "Low", [f"Indecisive standard distribution frame. Body ratio holds {m.body_pct:.1f}% context.", "Volume metrics stable near historical baseline averages."])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN USER INTERFACE & SESSION ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔥 IGNITE Analytics Suite")
st.caption("Universal Engine: Plain-Text Asset Search, Macro Momentum Gauges & Candlestick Anatomy Parser")

st.warning("**⚠️ Educational Use Architecture Only — Not Core Financial Advice.** Parameters evaluate mathematical probabilities.", icon="⚠️")
st.divider()

# INITIAL SESSION STATES FOR THE APP HANDOFF
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = "RELIANCE.NS"
    st.session_state.current_title = "Reliance Industries Limited"

# UNIVERSAL PLAIN TEXT SEARCH INPUT CONTROL
st.markdown('<div class="section-label">Universal Global Stock Search</div>', unsafe_allow_html=True)
user_text_search = st.text_input(
    "Search for any global stock, crypto, or commodity by name:", 
    placeholder="Type company name here... (e.g., Tata Motors, Reliance, Apple, Google, Nifty, Gold)"
)

# ENGINE PROCESSOR FOR USER INPUTS
if user_text_search.strip():
    search_results = search_symbols(user_text_search)
    if search_results:
        st.markdown("**Select the exact match from the market directory:**")
        # Radio choice array for simple reading
        chosen_index = st.radio(
            "Found Matches:",
            range(len(search_results)),
            format_func=lambda i: f"{search_results[i]['name']} [{search_results[i]['symbol']}]",
            horizontal=True
        )
        if st.button("Analyze Selected Asset", use_container_width=True):
            st.session_state.current_ticker = search_results[chosen_index]["symbol"]
            st.session_state.current_title = search_results[chosen_index]["name"]
            st.success(f"Successfully connected to active pipeline stream for: {st.session_state.current_title}")
    else:
        st.error("No active corporate database entries matched that search text. Try adjusting terms.")

ticker = st.session_state.current_ticker
company_display_name = st.session_state.current_title

st.info(f"📊 **Currently Analyzing:** `{ticker}` — **{company_display_name}**")

# DUAL ROUTING DASHBOARD TABS
tab_macro, tab_ignite = st.tabs(["📈 Macro Momentum & Profile", "🔥 IGNITE Candlestick Analysis"])

# DATA PIPELINE BACKENDS
macro_df = fetch_raw_data(ticker, "1d", "120d")
info_dict = fetch_info(ticker)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: MACRO MOMENTUM ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_macro:
    if macro_df.empty or "Close" not in macro_df.columns:
        st.error(f"Macro historical timeline data is unavailable for `{ticker}` on standard daily durations.")
    else:
        close_series = macro_df["Close"].dropna()
        volume_series = macro_df["Volume"] if "Volume" in macro_df.columns else pd.Series(dtype=float)
        
        ema20 = calculate_ema(close_series, 20)
        sma50 = calculate_sma(close_series, 50)
        rsi = calculate_rsi(close_series, 14)
        
        l_pr, l_ema, l_rsi = float(close_series.iloc[-1]), float(ema20.iloc[-1]), float(rsi.iloc[-1])
        l_sma = float(sma50.iloc[-1]) if not sma50.isna().all() else None
        l_vol = int(volume_series.iloc[-1]) if not volume_series.empty else 0
        
        prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else l_pr
        change, chg_pct = l_pr - prev_close, ((l_pr - prev_close)/prev_close)*100
        
        # RENDERING RECONCILED PERFORMANCES
        st.markdown('<div class="section-label">Key Metrics Dashboard</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        
        currency_pfx = f"{info_dict.get('currency', '')} " if info_dict.get('currency') else ""
        cap_s, cap_l = format_market_cap(info_dict.get("marketCap"))
        
        k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Current Price</div><div class="kpi-value">{currency_pfx}{l_pr:,.2f}</div><div class="kpi-sub">Latest Session Close</div></div>', unsafe_allow_html=True)
        arrow, col = ("▲", "#16a34a") if change >= 0 else ("▼", "#dc2626")
        k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Today\'s Delta</div><div class="kpi-value" style="color:{col};">{arrow} {abs(change):,.2f}</div><div class="kpi-sub">{chg_pct:+.2f}%</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Market Cap</div><div class="kpi-value">{cap_s}</div><div class="kpi-sub">{cap_l}</div></div>', unsafe_allow_html=True)
        k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Volume Matrix</div><div class="kpi-value">{l_vol:,}</div><div class="kpi-sub">Transacted Units</div></div>', unsafe_allow_html=True)
        k5.markdown(f'<div class="kpi-card"><div class="kpi-label">EMA-20 Base</div><div class="kpi-value">{l_ema:,.2f}</div><div class="kpi-sub">{"▲ Above" if l_pr >= l_ema else "▼ Below"}</div></div>', unsafe_allow_html=True)
        k6.markdown(f'<div class="kpi-card"><div class="kpi-label">RSI-14 Index</div><div class="kpi-value">{l_rsi:.1f}</div><div class="kpi-sub">Normalized Velocity</div></div>', unsafe_allow_html=True)
        
        # MOMENTUM FRAME CLASSIFICATION
        st.markdown('<div class="section-label">Momentum Classification Signal</div>', unsafe_allow_html=True)
        sig = momentum_signal(l_pr, l_ema, l_rsi)
        st.markdown(
            f'<div class="signal-card {sig["card_class"]}">'
            f'<span class="sig-badge {sig["badge_class"]}">{sig["badge_text"]}</span>'
            f'<div class="sig-headline">{sig["headline"]}</div><div class="sig-body">{sig["body"]}</div></div>',
            unsafe_allow_html=True
        )
        
        # CHART INTERFACES
        st.markdown('<div class="section-label">Macro Price Distributions</div>', unsafe_allow_html=True)
        c_e, c_s = st.checkbox("Toggle Overlay: EMA-20", True), st.checkbox("Toggle Overlay: SMA-50", True)
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=macro_df["Time"].iloc[-60:], y=close_series.iloc[-60:], name="Close Price", line=dict(color="#3b82f6", width=2), fill="tozeroy", fillcolor="rgba(59,130,246,0.06)"))
        if c_e: fig_price.add_trace(go.Scatter(x=macro_df["Time"].iloc[-60:], y=ema20.iloc[-60:], name="EMA-20", line=dict(color="#f97316", width=1.5, dash="dot")))
        if c_s and l_sma: fig_price.add_trace(go.Scatter(x=macro_df["Time"].iloc[-60:], y=sma50.iloc[-60:], name="SMA-50", line=dict(color="#a78bfa", width=1.5, dash="dash")))
        fig_price.update_layout(hovermode="x unified", margin=dict(l=0,r=0,t=20,b=0), height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # PROFILE FIELDS RENDERING
        st.markdown('<div class="section-label">Enterprise Profile Summary</div>', unsafe_allow_html=True)
        p_items = [
            ("Sector Structuring", info_dict.get("sector", "N/A")), ("Industry Segment", info_dict.get("industry", "N/A")),
            ("Country Exchange", info_dict.get("country", "N/A")), ("Trailing Valuation P/E", f"{info_dict.get('trailingPE', 'N/A')}x"),
            ("Dividend Yield Status", f"{info_dict.get('dividendYield', 0)*100:.2f}%" if info_dict.get('dividendYield') else "N/A")
        ]
        grid_html = "".join(f'<div class="profile-item"><div class="profile-key">{k}</div><div class="profile-val">{v}</div></div>' for k,v in p_items)
        st.markdown(f'<div class="profile-grid">{grid_html}</div>', unsafe_allow_html=True)
        
        bs_summary = info_dict.get("longBusinessSummary")
        if bs_summary: st.markdown(f'<div class="summary-box">{textwrap.shorten(bs_summary, 700, placeholder="...")}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: IGNITE CANDLESTICK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_ignite:
    st.subheader("🔥 Granular Anatomy Extraction Logic")
    
    col_i, col_ref = st.columns([2, 2])
    with col_i:
        int_label = st.selectbox("Intraday Pipeline Resolution Interval", list(INTERVALS.keys()), index=3)
        sel_interval, sel_period = INTERVALS[int_label]
    with col_ref:
        st.write("")
        if st.button("Flush Cache Pipelines & Synchronize", use_container_width=True):
            fetch_raw_data.clear()
            
    with st.spinner("Processing Microstructural Sequences..."):
        ignite_df = fetch_raw_data(ticker, sel_interval, sel_period)
        
    if ignite_df is None or ignite_df.empty or len(ignite_df) < VOLUME_LOOKBACK + 2:
        st.error(f"Insufficient granularity ticks inside `{sel_interval}` parameters for asset `{ticker}`. Please switch interval mapping.")
    else:
        c_m = compute_candle_metrics(ignite_df)
        candle_signal = run_ignite_algorithm(c_m)
        
        i_left, i_right = st.columns([1.2, 2.2])
        
        with i_left:
            b_palette = {"UP": ("#0e7c5f", "🟢 BUY / REVERSAL CONTEXT"), "DOWN": ("#8f1e1e", "🔴 DISTRIBUTION EXHAUSTION"), "HOLD": ("#8a6d1a", "🟡 CONSOLIDATION HOLD")}
            bg_c, lbl_s = b_palette.get(candle_signal.direction, ("#444444", candle_signal.direction))
            
            st.markdown(
                f'<div style="background-color:{bg_c}; padding:20px; border-radius:12px; text-align:center; color:#fff;">'
                f'<div style="font-size:11px; letter-spacing:2px; opacity:0.8;">CORE PARSING MATRIX</div>'
                f'<div style="font-size:26px; font-weight:800; margin:6px 0;">{lbl_s}</div>'
                f'<div style="font-size:12px; opacity:0.9;">Confidence Vector: {candle_signal.confidence}</div></div>',
                unsafe_allow_html=True
            )
            
            st.markdown("#### Logic Framework Justifications:")
            for r in candle_signal.reasons:
                st.markdown(f"- {r}")
                
            st.markdown("---")
            st.write("**Raw Extraction Realities:**", {
                "Open": round(c_m.open, 4), "High": round(c_m.high, 4),
                "Low": round(c_m.low, 4), "Close": round(c_m.close, 4),
                "Relative Vol Scaling": f"{c_m.volume_ratio*100:.1f}%"
            })
            
        with i_right:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Body Distribution %", f"{c_m.body_pct:.1f}%")
            m2.metric("Upper Wick Target %", f"{c_m.upper_wick_pct:.1f}%")
            m3.metric("Lower Wick Base %", f"{c_m.lower_wick_pct:.1f}%")
            m4.metric("Volume Drift Delta", f"{c_m.volume_ratio*100:.0f}%")
            
            fig_cand = go.Figure()
            tail_df = ignite_df.tail(45)
            fig_cand.add_trace(go.Candlestick(x=tail_df["Time"], open=tail_df["Open"], high=tail_df["High"], low=tail_df["Low"], close=tail_df["Close"], name="OHLC Vitals"))
            
            t_last = tail_df["Time"].iloc[-1]
            fig_cand.add_vrect(x0=t_last, x1=t_last, line_width=4, line_color=bg_c, opacity=0.4)
            fig_cand.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10), height=400)
            st.plotly_chart(fig_cand, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption("🔒 Architecture Pipeline complete. Open search parameters activated across whole global index repositories via yfinance abstraction frameworks.")
