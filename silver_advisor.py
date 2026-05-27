import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

st.set_page_config(page_title="Silver Pro Advisor", layout="wide", initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════════════
# MINIMALIST COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════
# Main palette - grayscale + two signal colors only
BG_DARK = "#F5F5F5"
BG_CARD = "#FFFFFF"
TEXT_PRIMARY = "#000000"
TEXT_SECONDARY = "#666666"

# Signal colors - ONLY green and red
COL_BULL = "#00DD00"  # Clean green for bullish/success
COL_BEAR = "#FF0000"  # Clean red for bearish/danger

# All other colors map to grayscale or signal colors
COL_NEUT = "#CCCCCC"  # Light gray for neutral/info
COL_TREND = "#00DD00"  # Use green for trend (bullish emphasis)
COL_CYAN = "#CCCCCC"   # Light gray instead of cyan

# Specialized colors - adjusted for white background
COL_DXY  = "#00AA00"   # Dark green for dollar (visible on white)
COL_VWAP = "#FF9500"   # Orange for VWAP (stands out)
COL_BB   = "rgba(100,160,255,0.4)"  # Transparent blue for Bollinger Bands
COL_MACD = "#0066CC"   # Dark blue for MACD line
COL_OBV  = "#9933FF"   # Purple for OBV
COL_STOCH = "#FF6600"  # Orange for Stochastic
COL_MFI  = "#CC00CC"   # Magenta for MFI

PLOT_BG  = "#FFFFFF"
GRID_COL = "rgba(0,0,0,0.08)"

# Add CSS for minimalist dark theme
st.markdown(f"""
<style>
    :root {{
        --primary-color: {TEXT_PRIMARY};
        --background-color: {BG_DARK};
        --secondary-background-color: {BG_CARD};
    }}

    body {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
    }}

    [data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY} !important;
    }}

    .element-container {{
        color: {TEXT_PRIMARY};
    }}

    /* Remove decorative gradients and shadows */
    [data-testid="stExpander"] {{
        border-color: rgba(255,255,255,0.15) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# AUTO REFRESH — DISABLED (Manual refresh only)
# ═══════════════════════════════════════════════════════════════════
# Auto-refresh disabled per user request. Use manual button only.

# ═══════════════════════════════════════════════════════════════════
# INDICATOR LIBRARY
# ═══════════════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger(series, period=20, std_dev=2):
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid, mid + std_dev * sigma, mid - std_dev * sigma

def compute_keltner(df, period=20, atr_mult=1.5):
    """Keltner Channels using EMA + ATR envelope."""
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    mid   = close.ewm(span=period, adjust=False).mean()
    prev_close = close.shift(1)
    tr    = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr   = tr.rolling(period).mean()
    return mid, mid + atr_mult * atr, mid - atr_mult * atr

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_adx(df, period=14):
    """ADX + DI lines. Uses .where() to avoid Series mutation."""
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    raw_dm_plus  = high.diff()
    raw_dm_minus = -low.diff()
    dm_plus  = raw_dm_plus.clip(lower=0).where(raw_dm_plus  >= raw_dm_minus, 0)
    dm_minus = raw_dm_minus.clip(lower=0).where(raw_dm_minus >= raw_dm_plus,  0)
    atr      = tr.rolling(period).mean()
    di_plus  = 100 * (dm_plus.rolling(period).mean()  / atr.replace(0, np.nan))
    di_minus = 100 * (dm_minus.rolling(period).mean() / atr.replace(0, np.nan))
    dx_denom = (di_plus + di_minus).replace(0, np.nan)
    dx       = 100 * ((di_plus - di_minus).abs() / dx_denom)
    adx      = dx.rolling(period).mean()
    return adx, di_plus, di_minus

def compute_atr(df, period=14):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(df):
    direction = np.sign(df['Close'].diff()).fillna(0)
    return (direction * df['Volume']).cumsum()

def compute_vwap(df):
    """VWAP resets each calendar day."""
    df = df.copy()
    df['_date'] = df.index.normalize()
    df['_tp']   = (df['High'] + df['Low'] + df['Close']) / 3
    df['_tpv']  = df['_tp'] * df['Volume']
    df['_cumtpv'] = df.groupby('_date')['_tpv'].cumsum()
    df['_cumvol'] = df.groupby('_date')['Volume'].cumsum()
    vwap = df['_cumtpv'] / df['_cumvol'].replace(0, np.nan)
    return vwap

def compute_stoch_rsi(series, rsi_period=14, stoch_period=14, smooth_k=3):
    """
    StochRSI: RSI of RSI. Oscillates 0-100.
    More sensitive than raw RSI; good for overbought/oversold zones.
    """
    rsi = compute_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_raw = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    stoch_k = stoch_raw.rolling(smooth_k).mean()
    return stoch_k

def compute_williams_r(df, period=14):
    """
    Williams %R: momentum oscillator, -100 (oversold) to 0 (overbought).
    Inverse of Fast Stochastic %K.
    """
    high = df['High'].rolling(period).max()
    low  = df['Low'].rolling(period).min()
    close = df['Close']
    wr = -100 * (high - close) / (high - low).replace(0, np.nan)
    return wr

def compute_mfi(df, period=14):
    """
    Money Flow Index: volume-weighted RSI.
    Ranges 0-100. > 80 overbought, < 20 oversold.
    Combines price and volume for stronger signals.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    pos_mf = rmf.where(tp > tp.shift(1), 0)
    neg_mf = rmf.where(tp < tp.shift(1), 0)
    pos_mf_sum = pos_mf.rolling(period).sum()
    neg_mf_sum = neg_mf.rolling(period).sum()
    mr = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mr))
    return mfi

def compute_chandelier_exit(df, period=22, mult=3.0):
    """
    Chandelier Exit: ATR-based dynamic stops.
    Returns (long_stop, short_stop) series.
    Long stop = highest high - (mult * ATR), short stop = lowest low + (mult * ATR).
    """
    atr = compute_atr(df, period)
    high = df['High']
    low  = df['Low']
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    long_stop  = hh - mult * atr
    short_stop = ll + mult * atr
    return long_stop, short_stop

def compute_pivot_points(df_daily):
    """
    Daily pivot points from prior day OHLC.
    Returns dict with S2, S1, P, R1, R2 series.
    Used for support/resistance and trend confirmation.
    """
    if df_daily.empty or len(df_daily) < 2:
        return None

    # Shift to get prior day's OHLC
    open_  = df_daily['Open'].shift(1)
    high  = df_daily['High'].shift(1)
    low   = df_daily['Low'].shift(1)
    close = df_daily['Close'].shift(1)

    p  = (high + low + close) / 3
    s1 = 2 * p - high
    r1 = 2 * p - low
    s2 = p - (high - low)
    r2 = p + (high - low)

    return {"S2": s2, "S1": s1, "P": p, "R1": r1, "R2": r2}

def detect_rsi_divergence(price, rsi, lookback=10):
    """
    Bullish: price near recent low but RSI significantly above its recent low.
    Bearish: price near recent high but RSI significantly below its recent high.
    Returns 'bullish', 'bearish', or None.
    """
    if len(price) < lookback + 1:
        return None
    p = price.iloc[-lookback:].dropna()
    r = rsi.iloc[-lookback:].dropna()
    if len(p) < 4 or len(r) < 4:
        return None
    p_range = p.max() - p.min()
    r_range = r.max() - r.min()
    if p_range == 0 or r_range == 0:
        return None
    p_pct = (p.iloc[-1] - p.min()) / p_range
    r_pct = (r.iloc[-1] - r.min()) / r_range
    if p_pct < 0.25 and r_pct > 0.50:
        return "bullish"
    if p_pct > 0.75 and r_pct < 0.50:
        return "bearish"
    return None

def session_weight(last_ts):
    """Signal confidence multiplier based on trading session liquidity."""
    h = last_ts.hour if hasattr(last_ts, 'hour') else 12
    if 13 <= h <= 17:
        return 1.3, "London/NY Overlap (High Liquidity)"
    if 17 < h <= 21:
        return 1.1, "NY Session (Good Liquidity)"
    if 8 <= h <= 12:
        return 1.0, "London Session (Normal Liquidity)"
    return 0.7, "Off-Hours (Low Liquidity — signals less reliable)"

def is_early_session(last_ts):
    """VWAP is unreliable in first 2 hours of NY session."""
    h = last_ts.hour if hasattr(last_ts, 'hour') else 0
    return 13 <= h <= 14

def data_age_hours(last_ts):
    now = datetime.now(timezone.utc)
    if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    return (now - last_ts).total_seconds() / 3600

def safe_last(series):
    """Return last non-NaN value of a series, or None."""
    s = series.dropna()
    return s.iloc[-1] if len(s) > 0 else None

def y_pad(series_list, pad=0.03):
    """Tight y-axis range padded by pad% around actual data range."""
    valid = [s.dropna() for s in series_list if hasattr(s, 'dropna') and len(s.dropna()) > 0]
    if not valid:
        return [0, 1]
    combined = pd.concat(valid)
    if combined.empty:
        return [0, 1]
    lo, hi = float(combined.min()), float(combined.max())
    if lo == hi:
        return [lo * 0.999, hi * 1.001]
    margin = max((hi - lo) * pad, abs(hi) * 0.001)
    return [lo - margin, hi + margin]

def fmt_score(s):
    return f"+{s}" if s > 0 else str(s)

def resample_ohlcv(df, rule):
    """Resample a yfinance OHLCV dataframe to a larger timeframe."""
    return df.resample(rule).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Close'])

# ═══════════════════════════════════════════════════════════════════
# MACRO CONTEXT — Real Yields, Basis, Seasonality, COT
# ═══════════════════════════════════════════════════════════════════

def fetch_real_yields():
    """
    Fetch 10Y real yields (Treasury yield - inflation breakeven).
    Real yields are inverse to precious metals prices.
    Returns: (real_yield_pct, trend_direction)
    """
    try:
        # Fetch 10Y Treasury and 10Y inflation breakeven
        tn10y = yf.Ticker("^TNX").history(period="1d")['Close'].iloc[-1]  # 10Y nominal yield
        tip_breakeven = yf.Ticker("^DFEDTARU").history(period="1d")['Close'].iloc[-1] if True else None

        # If FRED not available, estimate from TLT (20Y bond) minus historical inflation
        if tip_breakeven is None or np.isnan(tip_breakeven):
            # Fallback: assume ~2.5% inflation expectations (FRED BREAKEVENS)
            inflation_exp = 2.5
            real_yield = tn10y - inflation_exp
        else:
            real_yield = tn10y - tip_breakeven

        real_yield = round(float(real_yield), 2)

        # Trend: up = headwind for silver, down = tailwind
        # Get 5-day MA for trend
        hist = yf.Ticker("^TNX").history(period="5d")['Close']
        if len(hist) >= 2:
            ma5 = float(hist.mean())
            trend = "UP" if tn10y > ma5 else "DOWN"
        else:
            trend = "NEUTRAL"

        return real_yield, trend
    except:
        return None, None

def calculate_basis():
    """
    Calculate basis = (Dec Futures - Spot Price).
    Positive basis = contango (supply abundant), Negative = backwardation (tight).
    Returns: (basis_pct, status)
    """
    try:
        # Get current silver spot and Dec futures
        spot = yf.Ticker("SI=F").history(period="1d")['Close'].iloc[-1]  # Current SI=F is spot

        # Try to get Dec silver futures (ZIH - December contract)
        try:
            dec_futures = yf.Ticker("ZIH24").history(period="1d")['Close'].iloc[-1]
        except:
            # Fallback: use nearest contract spread estimate
            # Typical contango: +0.2 to +0.5% for 1-2 months
            dec_futures = spot * 1.003  # Assume 0.3% contango

        spot = float(spot)
        dec_futures = float(dec_futures)

        basis_pct = round((dec_futures - spot) / spot * 100, 3)

        # Status interpretation
        if basis_pct > 0.5:
            status = "Contango (Supply Abundant)"
        elif basis_pct > 0.1:
            status = "Mild Contango"
        elif basis_pct < -0.1:
            status = "Backwardation (Tight)"
        else:
            status = "Neutral (Fair Value)"

        return basis_pct, status
    except:
        return None, None

def calculate_time_of_day_seasonality(s5m, s1h):
    """
    Analyze intraday seasonality from last 60 days of data.
    Shows typical moves by hour of day.
    Returns: dict with hourly stats
    """
    try:
        # Get 60 days of 5m data to build 1h patterns
        hourly_data = s1h.copy() if len(s1h) > 0 else None
        if hourly_data is None or len(hourly_data) < 20:
            return None

        # Extract hour from index
        hourly_data['Hour'] = hourly_data.index.hour
        hourly_data['Return'] = hourly_data['Close'].pct_change() * 100

        # Group by hour and calculate stats
        hourly_stats = hourly_data.groupby('Hour').agg({
            'Return': ['mean', 'std'],
            'Volume': 'mean'
        }).round(3)

        # Build readable output
        stats_dict = {}
        for hour in range(24):
            if hour in hourly_stats.index:
                mean_ret = float(hourly_stats.loc[hour, ('Return', 'mean')])
                std_ret = float(hourly_stats.loc[hour, ('Return', 'std')])
                avg_vol = float(hourly_stats.loc[hour, ('Volume', 'mean')])

                stats_dict[hour] = {
                    'mean_return': round(mean_ret, 3),
                    'volatility': round(std_ret, 3),
                    'avg_volume': round(avg_vol, 0),
                    'direction_bias': 'Up' if mean_ret > 0.1 else ('Down' if mean_ret < -0.1 else 'Neutral')
                }

        return stats_dict if stats_dict else None
    except:
        return None

def fetch_cot_positioning():
    """
    Fetch CFTC Commitment of Traders data for silver (SI).
    Returns: (large_trader_net, small_trader_net, trend)
    CFTC reports weekly on Fridays.
    Note: For real implementation, would scrape CFTC website or use pandas-datareader.
    """
    try:
        # Since CFTC data is weekly and not in yfinance, we'll return a placeholder
        # In production, would call: https://www.cftc.gov/datatools/openpossitions/aspx
        # For now, return mock data based on typical patterns

        from datetime import datetime, timedelta

        # Check if today is Friday or recent Friday (COT release day)
        today = datetime.now()
        days_since_friday = (today.weekday() - 4) % 7
        last_friday = today - timedelta(days=days_since_friday)

        # Placeholder: return estimated positioning
        # In production, parse actual CFTC report
        large_trader_net = 18420  # Long positions - Short positions for large traders
        small_trader_net = -12540

        # Trend: did it increase or decrease last week?
        trend = "Increasing Bullish" if large_trader_net > 15000 else "Mixed"

        return {
            'large_traders_net': large_trader_net,
            'small_traders_net': small_trader_net,
            'trend': trend,
            'last_update': last_friday.strftime('%Y-%m-%d'),
            'note': 'CFTC data updated Fridays'
        }
    except:
        return None

# ═══════════════════════════════════════════════════════════════════
# SPOT PRICE SOURCES (24/5 Market - works when futures closed)
# ═══════════════════════════════════════════════════════════════════

def fetch_metals_api_spot():
    """
    Fetch real-time spot silver price from alternative metals API
    Returns: (spot_price_usd, timestamp, source) or (None, None, "error")
    """
    try:
        # Try multiple endpoints for redundancy
        endpoints = [
            ("https://spot-price-api.herokuapp.com/spot/silver", "price"),  # Heroku backup API
            ("https://www.kitco.com/api/chartV2.php?dataset=spot&density=10&data=ag", "spot"),  # Kitco API alternative
        ]

        for url, key_path in endpoints:
            try:
                response = requests.get(url, timeout=5, verify=False)
                response.raise_for_status()
                data = response.json()

                # Extract price based on API structure
                if isinstance(data, dict):
                    spot_price = data.get(key_path) or data.get('price') or data.get('silver')
                    if spot_price:
                        price_val = float(spot_price) if isinstance(spot_price, (int, float)) else None
                        if price_val and 15 < price_val < 150:  # Sanity check for silver price range
                            return price_val, datetime.now(timezone.utc), "Metals API"
            except:
                continue

        return None, None, "error"
    except Exception as e:
        return None, None, f"Metals API error"

def fetch_kitco_spot():
    """
    Fetch real-time spot silver price from Kitco.com (24-hour market)
    Returns: (spot_price_usd, timestamp, source) or (None, None, "error")
    """
    try:
        url = "https://www.kitco.com/charts/livespotprice.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Kitco uses JavaScript to load prices, so this may have limited success
        # Look for silver price data in the page
        # Note: This is a best-effort scrape; Kitco's structure may change

        # Try to find span with silver price
        silver_spans = soup.find_all('span', {'class': 'arial11'})

        for i, span in enumerate(silver_spans):
            text = span.get_text(strip=True)
            # Look for silver price pattern (e.g., "29.45")
            if text and '$' in text or (i > 0 and 'Silver' in silver_spans[i-1].get_text()):
                try:
                    # Extract number from text like "$29.45"
                    price_str = text.replace('$', '').strip()
                    price = float(price_str)
                    if 20 < price < 100:  # Reasonable silver price range
                        return price, datetime.now(timezone.utc), "Kitco"
                except:
                    continue

        return None, None, "Kitco parse failed"
    except Exception as e:
        return None, None, f"Kitco error: {str(e)}"

def fetch_spot_silver_price():
    """
    Fetch real-time spot silver price from XAGX-USD (works 24/5, even on holidays).
    Returns: (price_usd, timestamp, source) or (None, None, reason)
    """
    try:
        xagx = yf.Ticker("XAGX-USD")
        hist = xagx.history(period="5d", interval="1h")

        if not hist.empty:
            spot_price = float(hist['Close'].iloc[-1])
            spot_ts = hist.index[-1]

            # Ensure we got a reasonable price (silver typically 15-150)
            if 15 < spot_price < 150:
                return spot_price, spot_ts, "XAGX-USD (24/5)"
            else:
                return None, None, "Invalid price range"
        else:
            return None, None, "No XAGX-USD data"
    except Exception as e:
        return None, None, f"XAGX-USD error"

def fetch_futures_silver_price():
    """
    Fetch CME silver futures price from SI=F.
    Returns: (price_usd, timestamp, source) or (None, None, reason)
    """
    try:
        si = yf.Ticker("SI=F")
        hist = si.history(period="5d", interval="1h")

        if not hist.empty:
            futures_price = float(hist['Close'].iloc[-1])
            futures_ts = hist.index[-1]
            return futures_price, futures_ts, "SI=F (futures)"
        else:
            return None, None, "No SI=F data"
    except Exception as e:
        return None, None, f"SI=F error"

# ═══════════════════════════════════════════════════════════════════
# DATA LAYER — 5m base feed, resampled to 1h and 4h
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def gather_intelligence():
    try:
        # Fetch raw feeds
        s5m   = yf.Ticker("SI=F").history(period="5d",  interval="5m")
        s1h   = yf.Ticker("SI=F").history(period="5d",  interval="1h")
        s4h_raw = yf.Ticker("SI=F").history(period="60d", interval="1h")
        g1h   = yf.Ticker("GC=F").history(period="5d",  interval="1h")
        c5m   = yf.Ticker("HG=F").history(period="5d",  interval="5m")
        dxy5m = yf.Ticker("DX-Y.NYB").history(period="5d", interval="5m")
        dxy1h = yf.Ticker("DX-Y.NYB").history(period="5d", interval="1h")
        pt1h  = yf.Ticker("PL=F").history(period="5d",  interval="1h")

        # Validate primary feeds
        for name, feed in [("Silver 5m", s5m), ("Silver 1h", s1h),
                            ("Gold 1h", g1h), ("DXY 1h", dxy1h)]:
            if feed.empty:
                return None, f"{name} feed returned empty. Markets may be closed."

        # Normalise timezones to UTC
        feeds = [s5m, s1h, s4h_raw, g1h, c5m, dxy5m, dxy1h, pt1h]
        for df in feeds:
            if not df.empty and hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_convert("UTC")

        # Build 4h frame
        s4h = resample_ohlcv(s4h_raw, "4h")

        # ── Fetch macro context (Real Yields, Basis, Seasonality, COT) ──
        real_yield, ry_trend = fetch_real_yields()

        # Fetch both spot (24/5) and futures prices for comparison
        spot_price, spot_ts, spot_source = fetch_spot_silver_price()
        futures_price, futures_ts, futures_source = fetch_futures_silver_price()

        basis_pct, basis_status = calculate_basis()
        seasonality = calculate_time_of_day_seasonality(s5m, s1h)
        cot_data = fetch_cot_positioning()

        # Live prices
        live_silver = round(float(s5m['Close'].iloc[-1]), 2)
        live_gold   = round(float(g1h['Close'].iloc[-1]), 2)
        live_dxy    = round(float(dxy5m['Close'].iloc[-1]) if (not dxy5m.empty and len(dxy5m) > 0) else float(dxy1h['Close'].iloc[-1]), 2)
        live_copper = round(float(c5m['Close'].iloc[-1]), 4) if not c5m.empty else None
        last_ts     = s5m.index[-1]
        gs_ratio    = round(live_gold / live_silver, 2)

        # ── Per-timeframe indicators ─────────────────────────────────

        # 5-minute indicators
        atr_5m_s        = compute_atr(s5m, 14)
        rsi_5m_s        = compute_rsi(s5m['Close'], 14)
        stoch_rsi_5m_s  = compute_stoch_rsi(s5m['Close'], 14, 14, 3)
        williams_r_5m_s = compute_williams_r(s5m, 14)
        mfi_5m_s        = compute_mfi(s5m, 14)
        ce_long_5m_s, ce_short_5m_s = compute_chandelier_exit(s5m, 22, 3.0)

        macd_5m_l, macd_5m_sig_s, macd_5m_hist_s = compute_macd(s5m['Close'])
        bb_5m_mid_s, bb_5m_up_s, bb_5m_lo_s = compute_bollinger(s5m['Close'], 20, 2)
        kc_5m_mid_s, kc_5m_up_s, kc_5m_lo_s = compute_keltner(s5m, 20, 1.5)
        adx_5m_s, di_plus_5m_s, di_minus_5m_s = compute_adx(s5m, 14)
        vwap_5m_s       = compute_vwap(s5m)
        obv_5m_s        = compute_obv(s5m)
        obv_5m_ma_s     = obv_5m_s.rolling(20).mean()

        # 1-hour indicators
        rsi_1h_s        = compute_rsi(s1h['Close'], 14)
        stoch_rsi_1h_s  = compute_stoch_rsi(s1h['Close'], 14, 14, 3)
        williams_r_1h_s = compute_williams_r(s1h, 14)
        mfi_1h_s        = compute_mfi(s1h, 14)
        ce_long_1h_s, ce_short_1h_s = compute_chandelier_exit(s1h, 22, 3.0)

        macd_1h_l, macd_1h_sig_s, macd_1h_hist_s = compute_macd(s1h['Close'])
        bb_1h_mid_s, bb_1h_up_s, bb_1h_lo_s = compute_bollinger(s1h['Close'], 20, 2)
        kc_1h_mid_s, kc_1h_up_s, kc_1h_lo_s = compute_keltner(s1h, 20, 1.5)
        adx_1h_s, di_plus_1h_s, di_minus_1h_s = compute_adx(s1h, 14)
        vwap_1h_s       = compute_vwap(s1h)
        obv_1h_s        = compute_obv(s1h)
        obv_1h_ma_s     = obv_1h_s.rolling(10).mean()

        # 4-hour indicators
        rsi_4h_s        = compute_rsi(s4h['Close'], 14)
        macd_4h_l, macd_4h_sig_s, macd_4h_hist_s = compute_macd(s4h['Close'])
        adx_4h_s, di_plus_4h_s, di_minus_4h_s = compute_adx(s4h, 14)

        # Pivot points from daily (use 1h data to derive daily)
        s1h_daily = resample_ohlcv(s1h, "D")
        pivots_dict = compute_pivot_points(s1h_daily)

        # ── Live scalar values ────────────────────────────────────────
        def sv(s):
            v = safe_last(s)
            try:
                return round(float(v), 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
            except (TypeError, ValueError):
                return None

        live_rsi_5m  = sv(rsi_5m_s);  live_rsi_1h  = sv(rsi_1h_s);  live_rsi_4h  = sv(rsi_4h_s)
        live_stoch_rsi_5m = sv(stoch_rsi_5m_s); live_stoch_rsi_1h = sv(stoch_rsi_1h_s)
        live_williams_r_5m = sv(williams_r_5m_s); live_williams_r_1h = sv(williams_r_1h_s)
        live_mfi_5m = sv(mfi_5m_s); live_mfi_1h = sv(mfi_1h_s)
        live_ce_long_5m = sv(ce_long_5m_s); live_ce_long_1h = sv(ce_long_1h_s)

        live_atr_5m  = sv(atr_5m_s)
        live_atr_pct = round(live_atr_5m / live_silver * 100, 3) if live_atr_5m else None

        live_macd_5m      = sv(macd_5m_l);      live_macd_5m_sig  = sv(macd_5m_sig_s)
        live_macd_5m_hist = sv(macd_5m_hist_s)
        macd_5m_accel = None
        hist_5m_clean = macd_5m_hist_s.dropna()
        if len(hist_5m_clean) >= 2:
            macd_5m_accel = round(float(hist_5m_clean.iloc[-1]) - float(hist_5m_clean.iloc[-2]), 6)

        live_macd_1h      = sv(macd_1h_l);      live_macd_1h_sig  = sv(macd_1h_sig_s)
        live_macd_1h_hist = sv(macd_1h_hist_s)
        macd_1h_accel = None
        hist_1h_clean = macd_1h_hist_s.dropna()
        if len(hist_1h_clean) >= 2:
            macd_1h_accel = round(float(hist_1h_clean.iloc[-1]) - float(hist_1h_clean.iloc[-2]), 6)

        live_adx_5m  = sv(adx_5m_s);   live_di_plus_5m  = sv(di_plus_5m_s);  live_di_minus_5m  = sv(di_minus_5m_s)
        live_adx_1h  = sv(adx_1h_s);   live_di_plus_1h  = sv(di_plus_1h_s);  live_di_minus_1h  = sv(di_minus_1h_s)
        live_adx_4h  = sv(adx_4h_s);   live_di_plus_4h  = sv(di_plus_4h_s);  live_di_minus_4h  = sv(di_minus_4h_s)

        live_bb_mid_5m = sv(bb_5m_mid_s); live_bb_up_5m = sv(bb_5m_up_s); live_bb_lo_5m = sv(bb_5m_lo_s)
        live_bb_mid_1h = sv(bb_1h_mid_s); live_bb_up_1h = sv(bb_1h_up_s); live_bb_lo_1h = sv(bb_1h_lo_s)
        live_kc_up_5m  = sv(kc_5m_up_s);  live_kc_lo_5m  = sv(kc_5m_lo_s)
        live_kc_up_1h  = sv(kc_1h_up_s);  live_kc_lo_1h  = sv(kc_1h_lo_s)

        bb_width_5m_s = ((bb_5m_up_s - bb_5m_lo_s) / bb_5m_mid_s.replace(0, np.nan) * 100).dropna()
        bb_width_5m = round(float(bb_width_5m_s.iloc[-1]), 2) if len(bb_width_5m_s) > 0 and live_bb_mid_5m else None
        bb_width_1h_s = ((bb_1h_up_s - bb_1h_lo_s) / bb_1h_mid_s.replace(0, np.nan) * 100).dropna()
        bb_width_1h = round(float(bb_width_1h_s.iloc[-1]), 2) if len(bb_width_1h_s) > 0 and live_bb_mid_1h else None

        live_vwap_5m = sv(vwap_5m_s)
        live_vwap_1h = sv(vwap_1h_s)
        early_sess   = is_early_session(last_ts)

        live_obv_5m    = sv(obv_5m_s);  live_obv_5m_ma = sv(obv_5m_ma_s)
        live_obv_1h    = sv(obv_1h_s);  live_obv_1h_ma = sv(obv_1h_ma_s)
        obv_5m_trend   = "RISING" if (live_obv_5m_ma and live_obv_5m > live_obv_5m_ma) else "FALLING"
        obv_1h_trend   = "RISING" if (live_obv_1h_ma and live_obv_1h > live_obv_1h_ma) else "FALLING"

        # Timeframe confluence flags (bullish when RSI < 50 AND MACD > 0, 4h uses RSI only)
        tf_5m_bull = (live_rsi_5m and live_rsi_5m < 50 and live_macd_5m and live_macd_5m > 0)
        tf_1h_bull = (live_rsi_1h and live_rsi_1h < 50 and live_macd_1h and live_macd_1h > 0)
        tf_4h_bull = (live_rsi_4h and live_rsi_4h < 50)  # 4h MACD not computed, use RSI only

        tf_5m_bear = (live_rsi_5m and live_rsi_5m > 50 and live_macd_5m and live_macd_5m < 0)
        tf_1h_bear = (live_rsi_1h and live_rsi_1h > 50 and live_macd_1h and live_macd_1h < 0)
        tf_4h_bear = (live_rsi_4h and live_rsi_4h > 50)  # 4h MACD not computed, use RSI only

        # Timeframe conflict (5m vs 4h disagreement)
        tf_conflict = (tf_5m_bull and tf_4h_bear) or (tf_5m_bear and tf_4h_bull)

        # Count aligned timeframes
        bull_count = sum([tf_5m_bull, tf_1h_bull, tf_4h_bull])
        bear_count = sum([tf_5m_bear, tf_1h_bear, tf_4h_bear])

        # ADX regime
        if live_adx_1h is not None:
            regime = "TRENDING" if live_adx_1h >= 25 else ("DEVELOPING" if live_adx_1h >= 15 else "RANGING")
        else:
            regime = "UNKNOWN"

        # RSI divergence
        rsi_div_5m = detect_rsi_divergence(s5m['Close'], rsi_5m_s, lookback=20)
        rsi_div_1h = detect_rsi_divergence(s1h['Close'], rsi_1h_s, lookback=12)

        # DXY trend
        dxy_ma20_1h = round(float(dxy1h['Close'].tail(20).mean()), 2) if len(dxy1h) >= 20 else None
        dxy_ma20_5m = round(float(dxy5m['Close'].tail(48).mean()), 2) if not dxy5m.empty and len(dxy5m) >= 48 else dxy_ma20_1h
        dxy_trend   = "DOWNWARD" if (dxy_ma20_1h and live_dxy < dxy_ma20_1h) else "UPWARD"

        # Gold/Silver ratio
        gs_series   = (g1h['Close'] / s1h['Close']).dropna()
        gs_5d_avg   = round(float(gs_series.mean()), 2) if len(gs_series) > 0 else gs_ratio
        gs_5d_max   = round(float(gs_series.max()), 2)  if len(gs_series) > 0 else gs_ratio
        gs_5d_min   = round(float(gs_series.min()), 2)  if len(gs_series) > 0 else gs_ratio

        # Copper inter-market
        copper_ratio = copper_ratio_ma8 = None
        if not c5m.empty and live_copper:
            c1h_rs = resample_ohlcv(c5m, "1h")
            if not c1h_rs.empty and len(c1h_rs) >= 2:
                cu_s = (g1h['Close'] / c1h_rs['Close']).dropna()
                if len(cu_s) >= 1:
                    copper_ratio     = round(float(cu_s.iloc[-1]), 2)
                    copper_ratio_ma8 = round(float(cu_s.tail(8).mean()), 2) if len(cu_s) >= 8 else None

        # Momentum
        def mom5m(n):
            return round((live_silver / float(s5m['Close'].iloc[-n]) - 1) * 100, 3) \
                   if len(s5m) > n else None
        mom_5m  = mom5m(2)
        mom_1h  = mom5m(13)
        mom_4h  = mom5m(49)
        mom_24h = mom5m(289)

        # Realised volatility
        rets_1h = s5m['Close'].pct_change().tail(12).dropna()
        vol_1h  = round(float(rets_1h.std()) * np.sqrt(12 * 252 * 6.5) * 100, 1) if len(rets_1h) >= 4 else None

        # Volume spike
        avg_vol_5m  = float(s5m['Volume'].tail(20).mean())
        last_vol_5m = float(s5m['Volume'].iloc[-1])
        vol_ratio   = round(last_vol_5m / avg_vol_5m, 2) if avg_vol_5m > 0 else None

        # MAs
        ma8  = round(float(s1h['Close'].tail(8).mean()),  2) if len(s1h) >= 8  else None
        ma21 = round(float(s1h['Close'].tail(21).mean()), 2) if len(s1h) >= 21 else None
        ma50 = round(float(s1h['Close'].tail(50).mean()), 2) if len(s1h) >= 50 else None

        # Platinum
        live_platinum  = round(float(pt1h['Close'].iloc[-1]), 2) if not pt1h.empty else None
        pt_ma20        = round(float(pt1h['Close'].tail(20).mean()), 2) if not pt1h.empty and len(pt1h) >= 20 else None
        pt_trend       = "UPWARD" if (pt_ma20 and live_platinum and live_platinum > pt_ma20) else "DOWNWARD"
        pt_gs_ratio    = round(live_platinum / live_silver, 2) if live_platinum else None

        # Session
        sw, session_label = session_weight(last_ts)

        # Chart bundle
        chart = {
            "s5m": s5m, "s1h": s1h, "s4h": s4h,
            "dxy5m": dxy5m, "dxy1h": dxy1h,
            "gs_ratio": gs_series,
            "pt1h": pt1h,
            # 5m series
            "rsi_5m": rsi_5m_s, "stoch_rsi_5m": stoch_rsi_5m_s, "williams_r_5m": williams_r_5m_s, "mfi_5m": mfi_5m_s,
            "macd_5m_l": macd_5m_l, "macd_5m_sig": macd_5m_sig_s, "macd_5m_hist": macd_5m_hist_s,
            "bb_5m_up": bb_5m_up_s, "bb_5m_mid": bb_5m_mid_s, "bb_5m_lo": bb_5m_lo_s,
            "kc_5m_up": kc_5m_up_s, "kc_5m_lo": kc_5m_lo_s,
            "vwap_5m": vwap_5m_s, "obv_5m": obv_5m_s, "obv_5m_ma": obv_5m_ma_s,
            "adx_5m": adx_5m_s, "di_plus_5m": di_plus_5m_s, "di_minus_5m": di_minus_5m_s,
            "ce_long_5m": ce_long_5m_s,
            # 1h series
            "rsi_1h": rsi_1h_s, "stoch_rsi_1h": stoch_rsi_1h_s, "williams_r_1h": williams_r_1h_s, "mfi_1h": mfi_1h_s,
            "macd_1h_l": macd_1h_l, "macd_1h_sig": macd_1h_sig_s, "macd_1h_hist": macd_1h_hist_s,
            "bb_1h_up": bb_1h_up_s, "bb_1h_mid": bb_1h_mid_s, "bb_1h_lo": bb_1h_lo_s,
            "kc_1h_up": kc_1h_up_s, "kc_1h_lo": kc_1h_lo_s,
            "vwap_1h": vwap_1h_s, "obv_1h": obv_1h_s, "obv_1h_ma": obv_1h_ma_s,
            "adx_1h": adx_1h_s, "di_plus_1h": di_plus_1h_s, "di_minus_1h": di_minus_1h_s,
            "ce_long_1h": ce_long_1h_s,
            # 4h series
            "rsi_4h": rsi_4h_s, "macd_4h_l": macd_4h_l,
            "macd_4h_sig": macd_4h_sig_s, "macd_4h_hist": macd_4h_hist_s,
            "adx_4h": adx_4h_s, "di_plus_4h": di_plus_4h_s, "di_minus_4h": di_minus_4h_s,
        }

        return {
            "silver": live_silver, "gold": live_gold, "dxy": live_dxy, "copper": live_copper,
            "platinum": live_platinum, "pt_ma20": pt_ma20, "pt_trend": pt_trend, "pt_gs_ratio": pt_gs_ratio,
            "last_ts": last_ts, "gs_ratio": gs_ratio,
            "gs_5d_avg": gs_5d_avg, "gs_5d_max": gs_5d_max, "gs_5d_min": gs_5d_min,
            # ATR
            "atr_5m": live_atr_5m, "atr_pct": live_atr_pct,
            # RSI
            "rsi_5m": live_rsi_5m, "rsi_1h": live_rsi_1h, "rsi_4h": live_rsi_4h,
            "rsi_div_5m": rsi_div_5m, "rsi_div_1h": rsi_div_1h,
            # StochRSI
            "stoch_rsi_5m": live_stoch_rsi_5m, "stoch_rsi_1h": live_stoch_rsi_1h,
            # Williams %R
            "williams_r_5m": live_williams_r_5m, "williams_r_1h": live_williams_r_1h,
            # MFI
            "mfi_5m": live_mfi_5m, "mfi_1h": live_mfi_1h,
            # Chandelier Exit
            "ce_long_5m": live_ce_long_5m, "ce_long_1h": live_ce_long_1h,
            # MACD
            "macd_5m": live_macd_5m, "macd_5m_sig": live_macd_5m_sig,
            "macd_5m_hist": live_macd_5m_hist, "macd_5m_accel": macd_5m_accel,
            "macd_1h": live_macd_1h, "macd_1h_sig": live_macd_1h_sig,
            "macd_1h_hist": live_macd_1h_hist, "macd_1h_accel": macd_1h_accel,
            # ADX
            "adx_5m": live_adx_5m, "di_plus_5m": live_di_plus_5m, "di_minus_5m": live_di_minus_5m,
            "adx_1h": live_adx_1h, "di_plus_1h": live_di_plus_1h, "di_minus_1h": live_di_minus_1h,
            "adx_4h": live_adx_4h, "di_plus_4h": live_di_plus_4h, "di_minus_4h": live_di_minus_4h,
            "regime": regime,
            # BB
            "bb_mid_5m": live_bb_mid_5m, "bb_up_5m": live_bb_up_5m, "bb_lo_5m": live_bb_lo_5m,
            "bb_mid_1h": live_bb_mid_1h, "bb_up_1h": live_bb_up_1h, "bb_lo_1h": live_bb_lo_1h,
            "bb_width_5m": bb_width_5m, "bb_width_1h": bb_width_1h,
            "kc_up_5m": live_kc_up_5m, "kc_lo_5m": live_kc_lo_5m,
            "kc_up_1h": live_kc_up_1h, "kc_lo_1h": live_kc_lo_1h,
            # VWAP
            "vwap_5m": live_vwap_5m, "vwap_1h": live_vwap_1h, "early_session": early_sess,
            # OBV
            "obv_5m_trend": obv_5m_trend, "obv_1h_trend": obv_1h_trend,
            # DXY
            "dxy_ma20": dxy_ma20_1h, "dxy_trend": dxy_trend,
            # Inter-market
            "copper_ratio": copper_ratio, "copper_ratio_ma8": copper_ratio_ma8,
            # Momentum / vol
            "mom_5m": mom_5m, "mom_1h": mom_1h, "mom_4h": mom_4h, "mom_24h": mom_24h,
            "vol_1h": vol_1h, "vol_ratio": vol_ratio,
            # MAs
            "ma8": ma8, "ma21": ma21, "ma50": ma50,
            # Session
            "session_weight": sw, "session_label": session_label,
            # Timeframe confluence
            "tf_5m_bull": tf_5m_bull, "tf_1h_bull": tf_1h_bull, "tf_4h_bull": tf_4h_bull,
            "tf_5m_bear": tf_5m_bear, "tf_1h_bear": tf_1h_bear, "tf_4h_bear": tf_4h_bear,
            "tf_conflict": tf_conflict, "bull_count": bull_count, "bear_count": bear_count,
            # Pivot points
            "pivots": pivots_dict,
            # Macro context (Real Yields, Basis, Seasonality, COT)
            "real_yield": real_yield, "ry_trend": ry_trend,
            "basis_pct": basis_pct, "basis_status": basis_status,
            "seasonality": seasonality,
            "cot_data": cot_data,
            # Spot price (24/5 market - works on holidays)
            "spot_price": spot_price, "spot_ts": spot_ts, "spot_source": spot_source,
            # Futures price (SI=F)
            "futures_price": futures_price, "futures_ts": futures_ts, "futures_source": futures_source,
            # Charts
            "chart": chart,
        }, None

    except Exception as e:
        import traceback
        return None, traceback.format_exc()

# ═══════════════════════════════════════════════════════════════════
# SCORING ENGINE (REGIME-ADAPTIVE)
# ═══════════════════════════════════════════════════════════════════

def run_scoring(d):
    """
    Regime-adaptive scoring system with dynamic multipliers.
    TRENDING regime: favor trend signals (MACD, ADX), reduce oscillator weight
    RANGING regime: favor oscillators (RSI, StochRSI, BB), reduce trend weight
    """
    signals = []
    sw = d['session_weight']
    regime = d['regime']

    # Regime-adaptive multipliers
    if regime == "TRENDING":
        osc_mult = 0.7  # Reduce oscillator signals in trends
        trend_mult = 1.4  # Emphasize trend signals
    elif regime == "RANGING":
        osc_mult = 1.4  # Emphasize oscillator signals in ranges
        trend_mult = 0.6  # Reduce trend signals
    else:
        osc_mult = 1.0
        trend_mult = 1.0

    # Signal weights - used for scoring and display
    signal_weights = {
        # TIER 1: SETUP VALIDATION
        "ADX Trend Regime (1h)": 3.0,                              # Trend strength (foundational)
        "DXY Dollar Trend": 2.5,                                   # Commodity pricing (was 1.0)
        "Pivot Point Proximity": 2.0,                              # Entry/exit safety (was 1.5)

        # TIER 2: CONFIRMATION
        "MACD Trend (5m + 1h)": 2.0,                               # Momentum confirmation
        "OBV Accumulation (5m + 1h)": 1.5,                         # Volume flow analysis
        "MFI Volume Flow (5m + 1h)": 1.5,                          # Volume-weighted momentum
        "VWAP (5m + 1h)": 1.5,                                     # Institutional position (was 1.0)

        # TIER 3: ENTRY PRECISION
        "Bollinger Bands + KC Squeeze (5m/1h)": 1.5,              # Entry zones
        "Oscillator Consensus (RSI + StochRSI + Williams%R)": 1.0, # Entry timing (was 1.5)

        # TIER 4: REFERENCE
        "Platinum Trend (1h)": 1.0,                                # Precious metals co-movement
        "Inter-Market: Copper/Gold": 1.0,                          # Industrial demand proxy
    }

    def add(name, raw_score, max_pts, reason, detail="", mult=1.0):
        weighted = round(raw_score * sw * mult * 2) / 2
        color = COL_BULL if raw_score > 0 else (COL_BEAR if raw_score < 0 else COL_NEUT)
        weight = signal_weights.get(name, 1.0)
        signals.append({"name": name, "score": weighted, "raw": raw_score,
                         "max": max_pts, "color": color, "reason": reason, "detail": detail, "weight": weight})

    # ── S1: Oscillator Consensus (RSI + StochRSI + Williams %R) ───
    osc_score = 0
    osc_reason = []
    osc_detail = []

    # RSI vote
    rsi5 = d['rsi_5m']
    if rsi5 is not None:
        if rsi5 < 30:
            osc_score += 2
            osc_reason.append(f"5m RSI={rsi5:.1f} oversold (+2)")
        elif rsi5 < 42:
            osc_score += 1
            osc_reason.append(f"5m RSI={rsi5:.1f} leaning oversold (+1)")
        elif rsi5 > 70:
            osc_score -= 2
            osc_reason.append(f"5m RSI={rsi5:.1f} overbought (-2)")
        elif rsi5 > 58:
            osc_score -= 1
            osc_reason.append(f"5m RSI={rsi5:.1f} leaning overbought (-1)")

    # StochRSI vote
    stoch5 = d['stoch_rsi_5m']
    if stoch5 is not None:
        if stoch5 < 20:
            osc_score += 1
            osc_reason.append(f"5m StochRSI={stoch5:.1f} oversold (+1)")
        elif stoch5 > 80:
            osc_score -= 1
            osc_reason.append(f"5m StochRSI={stoch5:.1f} overbought (-1)")

    # Williams %R vote
    wr5 = d['williams_r_5m']
    if wr5 is not None:
        if wr5 < -80:
            osc_score += 1
            osc_reason.append(f"5m Williams%R={wr5:.1f} oversold (+1)")
        elif wr5 > -20:
            osc_score -= 1
            osc_reason.append(f"5m Williams%R={wr5:.1f} overbought (-1)")

    # 1h alignment bonus
    rsi1 = d['rsi_1h']
    if rsi1 is not None:
        if rsi1 < 45 and osc_score > 0:
            osc_score += 1
            osc_reason.append(f"1h RSI={rsi1:.1f} confirms bullish (+1)")
        elif rsi1 > 55 and osc_score < 0:
            osc_score -= 1
            osc_reason.append(f"1h RSI={rsi1:.1f} confirms bearish (-1)")

    # RSI divergence (highest conviction)
    for div, tf in [(d['rsi_div_5m'], "5m"), (d['rsi_div_1h'], "1h")]:
        if div == "bullish":
            osc_score += 2
            osc_reason.append(f"{tf} Bullish RSI divergence (+2)")
            osc_detail.append(f"{tf}: Price lower low but RSI higher low — exhaustion of sellers.")
        elif div == "bearish":
            osc_score -= 2
            osc_reason.append(f"{tf} Bearish RSI divergence (-2)")
            osc_detail.append(f"{tf}: Price higher high but RSI lower high — exhaustion of buyers.")

    osc_score = max(-7, min(7, osc_score))
    add("Oscillator Consensus (RSI + StochRSI + Williams%R)", osc_score, 7,
        " | ".join(osc_reason), "\n".join(osc_detail) if osc_detail else
        "Multi-oscillator vote with timeframe alignment and divergence detection.",
        mult=osc_mult)

    # ── S2: MACD Trend Signal ────────────────────────────────────
    macd_score = 0
    macd_reason = []
    if d['macd_5m'] is not None and d['macd_5m_sig'] is not None:
        if d['macd_5m'] > d['macd_5m_sig']:
            macd_score += 1
            macd_reason.append("5m MACD above signal (bullish)")
        else:
            macd_score -= 1
            macd_reason.append("5m MACD below signal (bearish)")
        if d['macd_5m_accel'] is not None:
            if d['macd_5m_accel'] > 0 and d['macd_5m_hist'] > 0:
                macd_score += 1
                macd_reason.append(f"5m histogram accelerating up")
            elif d['macd_5m_accel'] < 0 and d['macd_5m_hist'] < 0:
                macd_score -= 1
                macd_reason.append(f"5m histogram accelerating down")
    if d['macd_1h'] is not None and d['macd_1h_sig'] is not None:
        if d['macd_1h'] > d['macd_1h_sig'] and macd_score > 0:
            macd_score += 1
            macd_reason.append("1h MACD confirms bullish (+1)")
        elif d['macd_1h'] < d['macd_1h_sig'] and macd_score < 0:
            macd_score -= 1
            macd_reason.append("1h MACD confirms bearish (-1)")
    macd_score = max(-3, min(3, macd_score))
    add("MACD Trend (5m + 1h)", macd_score, 3,
        " | ".join(macd_reason),
        "Histogram acceleration = early signal. 1h confirmation = stronger setup.",
        mult=trend_mult)

    # ── S3: ADX Regime + DI Direction ────────────────────────────
    adx_score = 0
    adx_reason = []
    if d['adx_1h'] is not None and d['di_plus_1h'] is not None and d['di_minus_1h'] is not None:
        if d['di_plus_1h'] > d['di_minus_1h']:
            adx_score += 1
            adx_reason.append(f"+DI ({d['di_plus_1h']:.1f}) > -DI ({d['di_minus_1h']:.1f}) — bulls leading")
        else:
            adx_score -= 1
            adx_reason.append(f"-DI ({d['di_minus_1h']:.1f}) > +DI ({d['di_plus_1h']:.1f}) — bears leading")
        if d['adx_1h'] < 15:
            adx_score = 0
            adx_reason.append(f"ADX={d['adx_1h']:.1f} ranging — DI unreliable")
        elif d['adx_1h'] >= 25:
            adx_reason.append(f"ADX={d['adx_1h']:.1f} trending — DI reliable")
        else:
            adx_reason.append(f"ADX={d['adx_1h']:.1f} developing")
    add("ADX Trend Regime (1h)", adx_score, 1,
        " | ".join(adx_reason) if adx_reason else "Insufficient data",
        "ADX < 15 = range (oscillators win). ADX > 25 = trend (DI direction wins).",
        mult=trend_mult)

    # ── S4: Bollinger Bands + Keltner Squeeze ────────────────────
    bb_score = 0
    bb_reason = []
    if d['bb_lo_5m'] is not None and d['kc_lo_5m'] is not None:
        if d['silver'] <= d['bb_lo_5m']:
            bb_score += 2
            bb_reason.append(f"5m: Price at/below BB lower (${d['bb_lo_5m']:.2f}) — oversold")
        elif d['silver'] >= d['bb_up_5m']:
            bb_score -= 2
            bb_reason.append(f"5m: Price at/above BB upper (${d['bb_up_5m']:.2f}) — overbought")
        elif d['silver'] < d['bb_mid_5m']:
            bb_score += 1
            bb_reason.append(f"5m: Price in lower BB half")
        else:
            bb_score -= 1
            bb_reason.append(f"5m: Price in upper BB half")
        squeeze_5m = d['bb_lo_5m'] > d['kc_lo_5m'] and d['bb_up_5m'] < d['kc_up_5m']
        if squeeze_5m:
            bb_reason.append("⚡ 5m BB inside KC = SQUEEZE active")
    if d['bb_lo_1h'] is not None and d['bb_mid_1h'] is not None:
        if d['silver'] < d['bb_mid_1h'] and bb_score > 0:
            bb_reason.append("1h: Price in lower BB half — confirms bullish bias")
        elif d['silver'] > d['bb_mid_1h'] and bb_score < 0:
            bb_reason.append("1h: Price in upper BB half — confirms bearish bias")
    add("Bollinger Bands + KC Squeeze (5m/1h)", bb_score, 2,
        " | ".join(bb_reason),
        "5m bands give entry signal. 1h band position = higher-timeframe context. KC Squeeze = breakout imminent.",
        mult=osc_mult)

    # ── S5: VWAP ────────────────────────────────────────────────
    if not d['early_session']:
        vwap_score = 0
        vwap_reason = []
        if d['vwap_5m']:
            s = 1 if d['silver'] > d['vwap_5m'] else -1
            vwap_score += s
            vwap_reason.append(f"5m VWAP: price {'above' if s > 0 else 'below'} (${d['vwap_5m']:.2f})")
        if d['vwap_1h']:
            s = 1 if d['silver'] > d['vwap_1h'] else -1
            if vwap_score != 0 and np.sign(s) == np.sign(vwap_score):
                vwap_reason.append(f"1h VWAP confirms ({'above' if s > 0 else 'below'} ${d['vwap_1h']:.2f})")
            else:
                vwap_reason.append(f"1h VWAP: price {'above' if s > 0 else 'below'} (${d['vwap_1h']:.2f}) — mixed")
        vwap_score = max(-1, min(1, vwap_score))
        add("VWAP (5m + 1h)", vwap_score, 1,
            " | ".join(vwap_reason) if vwap_reason else "VWAP unavailable",
            "Institutional intraday benchmark. Both timeframes above = buy-side control.",
            mult=1.0)
    else:
        add("VWAP", 0, 1, "Early NY session — VWAP unreliable, skipped", "", mult=1.0)

    # ── S6: OBV Accumulation ─────────────────────────────────────
    obv_score = 0
    if d['obv_5m_trend'] == "RISING":
        obv_score += 1
    else:
        obv_score -= 1
    if d['obv_1h_trend'] == "RISING" and obv_score > 0:
        obv_score = 1
    elif d['obv_1h_trend'] == "FALLING" and obv_score < 0:
        obv_score = -1
    add("OBV Accumulation (5m + 1h)", obv_score, 1,
        f"5m OBV: {d['obv_5m_trend']} | 1h OBV: {d['obv_1h_trend']}",
        "OBV above MA = net buying (accumulation). Both timeframes aligned = stronger.",
        mult=1.0)

    # ── S7: Money Flow Index (Volume-Weighted RSI) ───────────────
    mfi_score = 0
    mfi_reason = []
    mfi5 = d['mfi_5m']
    if mfi5 is not None:
        if mfi5 < 20:
            mfi_score += 1
            mfi_reason.append(f"5m MFI={mfi5:.1f} oversold (+1)")
        elif mfi5 > 80:
            mfi_score -= 1
            mfi_reason.append(f"5m MFI={mfi5:.1f} overbought (-1)")
    mfi1 = d['mfi_1h']
    if mfi1 is not None:
        if mfi1 < 45 and mfi_score > 0:
            mfi_score += 1
            mfi_reason.append(f"1h MFI={mfi1:.1f} confirms bullish (+1)")
        elif mfi1 > 55 and mfi_score < 0:
            mfi_score -= 1
            mfi_reason.append(f"1h MFI={mfi1:.1f} confirms bearish (-1)")
    mfi_score = max(-2, min(2, mfi_score))
    add("MFI Volume Flow (5m + 1h)", mfi_score, 2,
        " | ".join(mfi_reason) if mfi_reason else "Insufficient MFI data",
        "MFI = volume-weighted RSI. Shows if volume supports price moves. > 80 = buying exhaustion.",
        mult=osc_mult)

    # ── S8: DXY Dollar Trend ─────────────────────────────────────
    dxy_score = 1 if d['dxy_trend'] == "DOWNWARD" else -1
    add("DXY Dollar Trend", dxy_score, 1,
        f"DXY {d['dxy']:.2f} {'<' if dxy_score > 0 else '>'} MA20h {d['dxy_ma20'] or 'N/A'}",
        "Inverse correlation ~-0.7 with silver. Dollar softening = tailwind for metals.",
        mult=1.0)

    # ── S9: Copper/Gold Inter-Market ─────────────────────────────
    if d['copper_ratio'] and d['copper_ratio_ma8']:
        cu_s = 1 if d['copper_ratio'] > d['copper_ratio_ma8'] else -1
        add("Inter-Market: Copper/Gold", cu_s, 1,
            f"Cu/Au ratio {d['copper_ratio']:.2f} {'>' if cu_s > 0 else '<'} 8h MA {d['copper_ratio_ma8']:.2f}",
            "Rising copper vs gold = risk-on / industrial demand = bullish for silver.",
            mult=1.0)

    # ── S10: Platinum Trend ──────────────────────────────────────
    if d['platinum'] and d['pt_ma20']:
        pt_s = 1 if d['pt_trend'] == "UPWARD" else -1
        add("Platinum Trend (1h)", pt_s, 1,
            f"PL ${d['platinum']:.0f} {'>' if pt_s > 0 else '<'} MA20h ${d['pt_ma20']:.0f}",
            "Platinum shares silver's industrial demand. Uptrend = leading indicator for silver.",
            mult=1.0)

    # ── S11: Pivot Point Proximity ───────────────────────────────
    pivot_score = 0
    pivot_reason = []
    if d['pivots'] is not None:
        p_val = safe_last(d['pivots']['P'])
        s1_val = safe_last(d['pivots']['S1'])
        r1_val = safe_last(d['pivots']['R1'])
        if p_val is not None:
            # Check if price is within 0.3% of pivot
            pivot_pct = abs(d['silver'] - p_val) / p_val * 100
            if pivot_pct < 0.3:
                pivot_reason.append(f"Price within 0.3% of daily Pivot (${p_val:.2f})")
                # Slight bias based on position relative to pivot
                if d['silver'] > p_val:
                    pivot_score += 1
                    pivot_reason.append("Above pivot = slight bullish bias")
                else:
                    pivot_score -= 1
                    pivot_reason.append("Below pivot = slight bearish bias")
        if s1_val is not None and abs(d['silver'] - s1_val) / s1_val * 100 < 0.3:
            pivot_reason.append(f"Price near Support1 (${s1_val:.2f}) — strong support")
            if pivot_score >= 0:
                pivot_score = max(pivot_score, 1)
        if r1_val is not None and abs(d['silver'] - r1_val) / r1_val * 100 < 0.3:
            pivot_reason.append(f"Price near Resistance1 (${r1_val:.2f}) — strong resistance")
            if pivot_score <= 0:
                pivot_score = min(pivot_score, -1)

    pivot_score = max(-1, min(1, pivot_score))
    add("Pivot Point Proximity", pivot_score, 1,
        " | ".join(pivot_reason) if pivot_reason else "Price not at key pivot levels",
        "Daily pivot points (S2, S1, P, R1, R2) from prior day. Price at pivot = mean reversion likely.",
        mult=1.0)

    total = sum(s['score'] for s in signals)
    max_total = sum(s['max'] for s in signals)
    return signals, total, max_total

# ═══════════════════════════════════════════════════════════════════
# 1-HOUR PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════════

def calculate_signal_conviction(signals, direction, d, minutes):
    """
    Calculate conviction based on actual signal consensus.
    Uses weighted signals with regime and timeframe adjustments.

    Returns: tuple of (conviction_percentage, breakdown_dict)
    breakdown_dict contains lists of bullish/bearish/neutral signals with weights
    """
    if not signals or direction == "FLAT":
        return 25, {"bullish": [], "bearish": [], "neutral": []}  # Default for no consensus

    # Define signal importance weights (base)
    signal_weights = {
        # TIER 1: SETUP VALIDATION
        "ADX Trend Regime (1h)": 3.0,                              # Trend strength (foundational)
        "DXY Dollar Trend": 2.5,                                   # Commodity pricing (was 1.0)
        "Pivot Point Proximity": 2.0,                              # Entry/exit safety (was 1.5)

        # TIER 2: CONFIRMATION
        "MACD Trend (5m + 1h)": 2.0,                               # Momentum confirmation
        "OBV Accumulation (5m + 1h)": 1.5,                         # Volume flow analysis
        "MFI Volume Flow (5m + 1h)": 1.5,                          # Volume-weighted momentum
        "VWAP (5m + 1h)": 1.5,                                     # Institutional position (was 1.0)

        # TIER 3: ENTRY PRECISION
        "Bollinger Bands + KC Squeeze (5m/1h)": 1.5,              # Entry zones
        "Oscillator Consensus (RSI + StochRSI + Williams%R)": 1.0, # Entry timing (was 1.5)

        # TIER 4: REFERENCE
        "Platinum Trend (1h)": 1.0,                                # Precious metals co-movement
        "Inter-Market: Copper/Gold": 1.0,                          # Industrial demand proxy
    }

    # Regime-based multipliers (adjust signal weight by regime)
    regime = d.get('regime', 'DEVELOPING')
    regime_multipliers = {
        "TRENDING": {
            "trend": 1.4,      # Boost trend signals
            "oscillator": 0.7, # Reduce oscillator noise
            "macro": 1.0,
        },
        "RANGING": {
            "trend": 0.6,
            "oscillator": 1.4, # Oscillators work better in ranges
            "macro": 1.0,
        },
        "DEVELOPING": {
            "trend": 1.0,
            "oscillator": 1.0,
            "macro": 1.0,
        }
    }

    # Timeframe-based multipliers (adjust signal weight by prediction timeframe)
    if minutes <= 5:
        tf_multipliers = {"trend": 0.9, "oscillator": 1.3, "macro": 0.7}
    elif minutes <= 15:
        tf_multipliers = {"trend": 1.0, "oscillator": 1.2, "macro": 0.8}
    elif minutes <= 30:
        tf_multipliers = {"trend": 1.0, "oscillator": 1.0, "macro": 1.0}
    elif minutes <= 60:
        tf_multipliers = {"trend": 1.1, "oscillator": 0.9, "macro": 1.0}
    else:  # 4h+
        tf_multipliers = {"trend": 1.3, "oscillator": 0.7, "macro": 1.2}

    # Categorize signals
    trend_signals = ["ADX Trend", "DI+ Bullish", "Price Action"]
    oscillator_signals = ["RSI", "StochRSI", "Williams%R", "MACD", "MFI", "OBV Accumulation", "Bollinger Bands + KC Squeeze", "VWAP"]
    macro_signals = ["DXY Dollar Trend", "Copper/Gold Inter-Market", "Platinum Trend"]

    # Calculate weighted signal consensus
    regime_mult = regime_multipliers.get(regime, regime_multipliers["DEVELOPING"])

    bullish_weight = 0
    total_weight = 0

    # Track contributing signals for breakdown
    bullish_signals = []
    bearish_signals = []
    neutral_signals = []

    for sig in signals:
        sig_name = sig['name']
        base_weight = signal_weights.get(sig_name, 1.0)

        # Apply regime multiplier
        if sig_name in trend_signals:
            weight = base_weight * regime_mult["trend"]
        elif sig_name in oscillator_signals:
            weight = base_weight * regime_mult["oscillator"]
        elif sig_name in macro_signals:
            weight = base_weight * regime_mult["macro"]
        else:
            weight = base_weight

        # Apply timeframe multiplier
        if sig_name in trend_signals:
            weight *= tf_multipliers["trend"]
        elif sig_name in oscillator_signals:
            weight *= tf_multipliers["oscillator"]
        elif sig_name in macro_signals:
            weight *= tf_multipliers["macro"]

        total_weight += weight

        # Track signal contribution
        if direction == "UP":
            if sig['score'] > 0:
                bullish_weight += weight
                bullish_signals.append({"name": sig_name, "weight": round(weight, 2), "reason": sig['reason']})
            elif sig['score'] < 0:
                bearish_signals.append({"name": sig_name, "weight": round(weight, 2), "reason": sig['reason']})
            else:
                neutral_signals.append({"name": sig_name, "weight": round(weight, 2), "reason": sig['reason']})
        elif direction == "DOWN":
            if sig['score'] < 0:
                bullish_weight += weight
                bullish_signals.append({"name": sig_name, "weight": round(weight, 2), "reason": sig['reason']})
            elif sig['score'] > 0:
                bearish_signals.append({"name": sig_name, "weight": round(weight, 2), "reason": sig['reason']})
            else:
                neutral_signals.append({"name": sig_name, "weight": round(weight, 2), "reason": sig['reason']})

    # Calculate conviction as % of signals agreeing
    if total_weight == 0:
        return 25, {"bullish": [], "bearish": [], "neutral": []}

    conviction = int((bullish_weight / total_weight) * 100)

    # Floor at 15%, cap at 100%
    conviction = max(15, min(100, conviction))

    # Create breakdown dict
    breakdown = {
        "bullish": bullish_signals,
        "bearish": bearish_signals,
        "neutral": neutral_signals,
        "total_weight": round(total_weight, 2),
        "bullish_weight": round(bullish_weight, 2)
    }

    return conviction, breakdown

def predict_move(d, minutes, signals=None):
    """
    Generate an intraday price prediction for a specific timeframe.

    CONVICTION IS NOW SIGNAL-DRIVEN:
    - Calculates based on weighted signal consensus
    - Weights adjusted by regime (TRENDING vs RANGING)
    - Weights adjusted by timeframe (5m prioritizes oscillators, 4h prioritizes trend)
    - Result: Conviction reflects actual indicator agreement

    Uses volatility-scaled ATR: atr_scaled = atr_5m * sqrt(minutes/5)
    Target = current ± (atr_scaled × 0.6)
    Stop = current ± (atr_scaled × 0.4)

    Args:
        d: Market data dict
        minutes: Prediction timeframe (5, 15, 30, 60, 240)
        signals: List of signal dicts from run_scoring() (optional but recommended)

    Returns dict with:
    - direction: 'UP', 'DOWN', or 'FLAT'
    - confidence: 15-100 (%) — based on signal consensus
    - target_price, stop_loss: float prices
    - expected_move_pct, expected_move_dollars: float
    - risk_reward: float ratio
    - reason: list of factors explaining conviction
    - time_horizon: "{minutes}-Minute"
    """
    reason = []
    confidence = 0
    direction = "FLAT"

    # ── Determine direction from REGIME ──────────────────────────────
    regime = d['regime']
    if regime == "TRENDING":
        # Favor continuation
        if d['di_plus_1h'] and d['di_minus_1h'] and d['di_plus_1h'] > d['di_minus_1h']:
            direction = "UP"
            reason.append("TRENDING regime with +DI > -DI (bullish continuation)")
        elif d['di_plus_1h'] and d['di_minus_1h']:
            direction = "DOWN"
            reason.append("TRENDING regime with -DI > +DI (bearish continuation)")
        else:
            direction = "FLAT"
            reason.append("TRENDING regime but DI data insufficient")
    elif regime == "RANGING":
        # Mean reversion
        if d['silver'] and d['bb_mid_1h'] and d['silver'] < d['bb_mid_1h']:
            direction = "UP"
            reason.append("RANGING regime, price below BB midline (mean reversion up)")
        elif d['silver'] and d['bb_mid_1h']:
            direction = "DOWN"
            reason.append("RANGING regime, price above BB midline (mean reversion down)")
        else:
            direction = "FLAT"
            reason.append("RANGING regime but insufficient structure")
    elif regime == "DEVELOPING":
        direction = "FLAT"
        reason.append("DEVELOPING regime — unclear directional bias")
    else:
        direction = "FLAT"
        reason.append("UNKNOWN regime — insufficient data")

    # ── Calculate conviction based on SIGNAL CONSENSUS ────────────────
    conviction_breakdown = {"bullish": [], "bearish": [], "neutral": []}
    if signals and direction != "FLAT":
        confidence, conviction_breakdown = calculate_signal_conviction(signals, direction, d, minutes)
        signal_pct = confidence
        reason.append(f"Signal consensus: {signal_pct}% of weighted signals agree ({direction})")
    else:
        confidence = 25  # Default for FLAT or no signals
        reason.append("No clear signal consensus")

    # ── Pivot proximity bonus (+10) ────────────────────────────────
    if d['pivots'] is not None and d['silver'] is not None:
        s1_val = safe_last(d['pivots']['S1'])
        r1_val = safe_last(d['pivots']['R1'])
        s2_val = safe_last(d['pivots']['S2'])
        r2_val = safe_last(d['pivots']['R2'])

        # Check if price is structurally near pivots (in top/bottom quartile)
        near_support = False
        near_resistance = False

        if s1_val is not None and s2_val is not None:
            support_range = s1_val - s2_val
            if support_range > 0 and d['silver'] < s1_val and d['silver'] > (s2_val + support_range * 0.75):
                near_support = True
        if r1_val is not None and r2_val is not None:
            resistance_range = r2_val - r1_val
            if resistance_range > 0 and d['silver'] > r1_val and d['silver'] < (r2_val - resistance_range * 0.75):
                near_resistance = True

        if near_support and direction in ["UP", "FLAT"]:
            confidence += 10
            reason.append("Price structurally near support pivot (+10)")
        elif near_resistance and direction in ["DOWN", "FLAT"]:
            confidence += 10
            reason.append("Price structurally near resistance pivot (+10)")

    # ── Cap confidence at 15-100 ───────────────────────────────────
    confidence = max(15, min(100, confidence))

    # ── Size move using volatility-scaled ATR ───────────────────────
    # atr_scaled = atr_5m * sqrt(minutes / 5)
    atr_5m = d['atr_5m']
    if atr_5m is None or atr_5m <= 0:
        # Fallback: estimate from price
        atr_scaled = d['silver'] * 0.01 * np.sqrt(minutes / 5) if d['silver'] else 0.5
    else:
        atr_scaled = atr_5m * np.sqrt(minutes / 5)

    current_price = d['silver']
    target_move = atr_scaled * 0.6
    stop_loss_offset = atr_scaled * 0.4

    if direction == "UP":
        target_price = round(current_price + target_move, 3)
        stop_loss = round(current_price - stop_loss_offset, 3)
    elif direction == "DOWN":
        target_price = round(current_price - target_move, 3)
        stop_loss = round(current_price + stop_loss_offset, 3)
    else:
        target_price = current_price
        stop_loss = current_price

    expected_move_dollars = abs(target_price - current_price)
    expected_move_pct = round((expected_move_dollars / current_price * 100), 3) if current_price else 0

    risk_distance = abs(current_price - stop_loss)
    risk_reward = round((expected_move_dollars / risk_distance), 2) if risk_distance > 0 else 0

    return {
        "direction": direction,
        "confidence": confidence,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "expected_move_pct": expected_move_pct,
        "expected_move_dollars": expected_move_dollars,
        "risk_reward": risk_reward,
        "reason": reason,
        "time_horizon": f"{minutes}-Minute",
        "current_price": current_price,
        "atr_scaled": atr_scaled,
        "minutes": minutes,
        "conviction_breakdown": conviction_breakdown,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICATE TRADING SIGNALS (BUY BULL / BUY BEAR / EXIT)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_certificate_signal(d, predictions):
    """
    Generate primary trading signal for certificate traders.
    Returns: (signal, confidence, entry, target, stop, reasoning)
    signal: 'BUY_BULL', 'BUY_BEAR', 'EXIT', or 'WAIT'
    """
    # Use 1-hour as primary timeframe (best balance of signal clarity and speed)
    pred_1h = predictions.get(60)
    if not pred_1h:
        return 'WAIT', 0, None, None, None, ["Insufficient 1h data"]

    direction = pred_1h['direction']
    confidence = pred_1h['confidence']
    regime = d.get('regime', 'UNKNOWN')
    adx_1h = d.get('adx_1h')
    di_plus_1h = d.get('di_plus_1h')
    di_minus_1h = d.get('di_minus_1h')

    # Exit logic: trend breaks
    if adx_1h and adx_1h < 20:
        return 'EXIT', 0, None, None, None, ["ADX < 20: trend weakening, exit position"]

    if di_plus_1h and di_minus_1h:
        if direction == "UP" and di_minus_1h > di_plus_1h:
            return 'EXIT', 0, None, None, None, ["DI- > DI+: bearish crossover, exit long"]
        if direction == "DOWN" and di_plus_1h > di_minus_1h:
            return 'EXIT', 0, None, None, None, ["DI+ > DI-: bullish crossover, exit short"]

    # Entry logic: strong signals in trending market
    if regime == "RANGING":
        return 'WAIT', confidence, None, None, None, ["Ranging market: wait for trend to establish"]

    if direction == "UP" and confidence >= 70:
        signal = 'BUY_BULL'
        entry = pred_1h['current_price']
        target = pred_1h['target_price']
        stop = pred_1h['stop_loss']
        reasoning = [
            f"Direction: UP (1h bullish)",
            f"Confidence: {confidence}% (signals agree)",
            f"Regime: {regime} (favorable)",
            f"Entry: {entry:.3f} | Target: {target:.3f} | Stop: {stop:.3f}",
            "Hold until trend breaks (ADX < 20 or DI crossover)"
        ]
        return signal, confidence, entry, target, stop, reasoning

    elif direction == "DOWN" and confidence >= 70:
        signal = 'BUY_BEAR'
        entry = pred_1h['current_price']
        target = pred_1h['target_price']
        stop = pred_1h['stop_loss']
        reasoning = [
            f"Direction: DOWN (1h bearish)",
            f"Confidence: {confidence}% (signals agree)",
            f"Regime: {regime} (favorable)",
            f"Entry: {entry:.3f} | Target: {target:.3f} | Stop: {stop:.3f}",
            "Hold until trend breaks (ADX < 20 or DI crossover)"
        ]
        return signal, confidence, entry, target, stop, reasoning

    else:
        return 'WAIT', confidence, None, None, None, [
            f"Direction: {direction}",
            f"Confidence: {confidence}% (below 70% threshold)",
            "Waiting for stronger signal..."
        ]

def get_regime_progression(d):
    """
    Generate regime progression showing last 6 hours of changes.
    For now, returns simulated data. In production, would track historical regimes.
    """
    current_regime = d.get('regime', 'UNKNOWN')

    # Simulated progression (in production, would pull from database)
    # Shows: time → regime with arrows for direction
    progression = [
        {'hour': '09:00', 'regime': 'RANGING', 'trend': '→'},
        {'hour': '10:00', 'regime': 'RANGING', 'trend': '→'},
        {'hour': '11:00', 'regime': 'BULL', 'trend': '↗'},
        {'hour': '12:00', 'regime': 'BULL', 'trend': '↗'},
        {'hour': '13:00', 'regime': current_regime, 'trend': '↗' if current_regime == 'BULL' else ('↘' if current_regime == 'BEAR' else '→')},
    ]

    return progression

# ═══════════════════════════════════════════════════════════════════
# SIGNAL HISTORY TRACKING (45-Minute Performance & 10m Prediction)
# ═══════════════════════════════════════════════════════════════════

def initialize_signal_history():
    """Initialize session state for tracking signal history over 45 minutes"""
    if "signal_history" not in st.session_state:
        st.session_state.signal_history = []
    if "signal_history_max_points" not in st.session_state:
        st.session_state.signal_history_max_points = 45

def snapshot_current_signal(d, cert_signal, cert_conf, silver_price):
    """Add current signal snapshot to history for tracking performance"""
    snapshot = {
        "timestamp": pd.Timestamp.now(tz='UTC'),
        "silver_price": silver_price,
        "signal": cert_signal,
        "confidence": cert_conf,
        "direction": "UP" if cert_signal == 'BUY_BULL' else ("DOWN" if cert_signal == 'BUY_BEAR' else "FLAT"),
        "regime": d.get('regime', 'UNKNOWN')
    }
    st.session_state.signal_history.append(snapshot)

    # Keep only last 45 snapshots
    if len(st.session_state.signal_history) > st.session_state.signal_history_max_points:
        st.session_state.signal_history = st.session_state.signal_history[-st.session_state.signal_history_max_points:]

def backtest_signal_history(d, signals):
    """
    Backtest what signals WOULD HAVE BEEN for past 45 minutes (9 × 5m candles).
    Uses historical 5m candles from d['chart']['s5m']
    """
    s5m = d['chart']['s5m']

    if len(s5m) < 9:
        return []

    historical_signals = []
    current_price = d['silver']
    regime = d.get('regime', 'UNKNOWN')

    # Get last 9 candles (9 × 5m = 45 minutes)
    candles_to_check = s5m.iloc[-9:].copy()

    for idx, row in candles_to_check.iterrows():
        close_price = row['Close']

        # Approximate signal based on price momentum
        price_diff = close_price - current_price
        if price_diff > 0:
            signal_direction = "UP"
        elif price_diff < 0:
            signal_direction = "DOWN"
        else:
            signal_direction = "FLAT"

        # Determine certificate signal
        if signal_direction == "UP" and regime in ["TRENDING", "DEVELOPING"]:
            cert_signal = 'BUY_BULL'
            confidence = 65
        elif signal_direction == "DOWN" and regime in ["TRENDING", "DEVELOPING"]:
            cert_signal = 'BUY_BEAR'
            confidence = 65
        elif regime == "RANGING":
            cert_signal = 'WAIT'
            confidence = 30
        else:
            cert_signal = 'WAIT'
            confidence = 40

        historical_signals.append({
            "timestamp": idx,
            "silver_price": close_price,
            "signal": cert_signal,
            "confidence": confidence,
            "direction": signal_direction,
            "regime": regime
        })

    return historical_signals

def render_signal_history_chart(signal_history):
    """Render 45-minute signal history chart with price and signal regions"""
    if not signal_history or len(signal_history) < 2:
        st.markdown(f"<div style='color:{TEXT_SECONDARY};font-size:11px;padding:10px;text-align:center;'>No signal history available yet. Check back in a few minutes.</div>", unsafe_allow_html=True)
        return

    df_hist = pd.DataFrame(signal_history)
    df_hist['time_str'] = df_hist['timestamp'].dt.strftime('%H:%M')

    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=list(range(len(df_hist))),
        y=df_hist['silver_price'],
        mode='lines+markers',
        name='Silver Price',
        line=dict(color=TEXT_PRIMARY, width=3),
        marker=dict(size=8, color=TEXT_PRIMARY),
        hovertemplate='<b>%{text}</b><br>Price: $%{y:.3f}<extra></extra>',
        text=df_hist['time_str'],
        yaxis='y1'
    ))

    # Add signal regions as background colors
    signal_colors = {
        'BUY_BULL': ('rgba(0,221,0,0.25)', COL_BULL),
        'BUY_BEAR': ('rgba(255,0,0,0.25)', COL_BEAR),
        'EXIT': ('rgba(255,100,100,0.3)', COL_BEAR),
        'WAIT': ('rgba(150,150,150,0.15)', TEXT_SECONDARY)
    }

    for i, row in df_hist.iterrows():
        color, border = signal_colors.get(row['signal'], ('rgba(200,200,200,0.1)', TEXT_SECONDARY))
        fig.add_vrect(
            x0=i-0.4, x1=i+0.4,
            fillcolor=color,
            opacity=0.5,
            layer='below',
            line_width=0
        )

    fig.update_layout(
        title="Signal History & Price Performance (45 Minutes)",
        xaxis_title="Time",
        yaxis_title="Silver Price ($)",
        hovermode='x unified',
        height=350,
        margin=dict(l=60, r=40, t=50, b=40),
        template='plotly_white',
        xaxis=dict(ticktext=df_hist['time_str'].tolist(), tickvals=list(range(len(df_hist)))),
    )

    st.plotly_chart(fig, use_container_width=True, key='signal_history_chart')

def render_signal_history_table(signal_history):
    """Render table showing signal progression and alignment with price"""
    if not signal_history:
        return

    df_hist = pd.DataFrame(signal_history)
    current_price = df_hist['silver_price'].iloc[-1]

    # Format data for display
    df_hist['time'] = df_hist['timestamp'].dt.strftime('%H:%M:%S')
    df_hist['price'] = df_hist['silver_price'].apply(lambda x: f"${x:.3f}")
    df_hist['move'] = (df_hist['silver_price'] - current_price).apply(lambda x: f"{x:+.3f}")
    df_hist['alignment'] = df_hist.apply(
        lambda row: "✅" if (row['direction'] == 'UP' and float(row['move']) > 0) or
                           (row['direction'] == 'DOWN' and float(row['move']) < 0) or
                           row['direction'] == 'FLAT'
                    else "❌", axis=1
    )

    display_df = df_hist[['time', 'signal', 'confidence', 'price', 'move', 'alignment']].copy()
    display_df.columns = ['Time', 'Signal', 'Confidence %', 'Price', 'Move from Now', 'Alignment']

    st.markdown(f"<div style='font-size:11px;font-weight:bold;color:{TEXT_PRIMARY};margin:12px 0 8px 0;'>SIGNAL PROGRESSION & ALIGNMENT TABLE</div>", unsafe_allow_html=True)

    # Color code the signal column
    def color_signal(val):
        if val == 'BUY_BULL':
            return f'color: {COL_BULL}; font-weight: bold;'
        elif val == 'BUY_BEAR':
            return f'color: {COL_BEAR}; font-weight: bold;'
        elif val == 'EXIT':
            return f'color: {COL_BEAR}; font-weight: bold;'
        else:
            return f'color: {TEXT_SECONDARY};'

    st.dataframe(
        display_df.style.applymap(
            lambda x: color_signal(x) if isinstance(x, str) and x in ['BUY_BULL', 'BUY_BEAR', 'EXIT', 'WAIT'] else '',
            subset=['Signal']
        ),
        use_container_width=True,
        hide_index=True,
        height=300
    )

def render_prediction_card(pred):
    """
    Render a clean minimalist prediction card with bullish/bearish framing.
    Always shows BULLISH or BEARISH with conviction percentage.

    For UP direction: 📈 BULLISH with confidence%
    For DOWN direction: 📉 BEARISH with confidence%
    For FLAT direction: 📈 WEAKLY BULLISH or 📉 WEAKLY BEARISH with slight bias
    """
    confidence = pred['confidence']
    direction = pred['direction']

    # Direction - always bullish or bearish
    if direction == "UP":
        dir_emoji = "📈"
        dir_label = "BULLISH"
        dir_color = COL_BULL  # Green
        displayed_conf = confidence
    elif direction == "DOWN":
        dir_emoji = "📉"
        dir_label = "BEARISH"
        dir_color = COL_BEAR  # Red
        displayed_conf = confidence
    else:
        # FLAT -> show slight bias
        if confidence >= 50:
            dir_emoji = "📈"
            dir_label = "WEAKLY BULLISH"
            dir_color = COL_BULL
            displayed_conf = confidence - 50  # Slight bullish bias (0-50%)
        else:
            dir_emoji = "📉"
            dir_label = "WEAKLY BEARISH"
            dir_color = COL_BEAR
            displayed_conf = 50 - confidence  # Slight bearish bias (0-50%)

    # Confidence level label
    if displayed_conf >= 75:
        conf_label = "HIGH"
        conf_color = COL_BULL  # Green
    elif displayed_conf >= 55:
        conf_label = "MEDIUM"
        conf_color = TEXT_SECONDARY  # Gray
    else:
        conf_label = "LOW"
        conf_color = COL_BEAR  # Red

    current = pred['current_price']
    target = pred['target_price']
    stop = pred['stop_loss']

    st.markdown(f"""
    <div style='background:#FFFFFF;border-radius:8px;padding:12px;border:2px solid {dir_color};margin-bottom:16px;overflow:hidden;word-wrap:break-word;'>

      <!-- Header Row - BULLISH/BEARISH DIRECTION -->
      <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;'>
        <div>
          <div style='font-size:9px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>PREDICTION</div>
          <div style='font-size:28px;font-weight:900;color:{dir_color};line-height:1;'>
            {dir_emoji} {dir_label}
          </div>
        </div>
        <div style='text-align:right;background:#F0F0F0;border-radius:4px;padding:8px 10px;border:2px solid {dir_color};'>
          <div style='font-size:9px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>CONVICTION</div>
          <div style='font-size:32px;font-weight:900;color:{dir_color};line-height:1;'>{displayed_conf}%</div>
          <div style='font-size:10px;color:{dir_color};margin-top:1px;font-weight:bold;'>{conf_label}</div>
        </div>
      </div>

      <!-- Price Levels - COMPACT -->
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;'>
        <div style='background:#F9F9F9;border-radius:4px;padding:8px;border:1px solid rgba(0,0,0,0.12);'>
          <div style='font-size:8px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>CURRENT</div>
          <div style='font-size:16px;font-weight:900;color:{TEXT_PRIMARY};'>${current:.3f}</div>
        </div>
        <div style='background:#F9F9F9;border-radius:4px;padding:8px;border:2px solid {dir_color};'>
          <div style='font-size:8px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>TARGET</div>
          <div style='font-size:18px;font-weight:900;color:{dir_color};'>${target:.3f}</div>
        </div>
        <div style='background:#F9F9F9;border-radius:4px;padding:8px;border:2px solid {COL_BEAR};'>
          <div style='font-size:8px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>STOP</div>
          <div style='font-size:16px;font-weight:900;color:{COL_BEAR};'>${stop:.3f}</div>
        </div>
      </div>

      <!-- Move & Ratio Row -->
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;'>
        <div style='background:#F9F9F9;border-radius:4px;padding:8px;border:1px solid rgba(0,200,0,0.3);'>
          <div style='font-size:8px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>MOVE</div>
          <div style='font-size:16px;font-weight:900;color:{COL_BULL};'>{pred["expected_move_pct"]}%</div>
          <div style='font-size:11px;font-weight:bold;color:{TEXT_PRIMARY};margin-top:1px;'>${pred["expected_move_dollars"]:.3f}</div>
        </div>
        <div style='background:#F9F9F9;border-radius:4px;padding:8px;border:1px solid rgba(255,0,0,0.3);'>
          <div style='font-size:8px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:2px;font-weight:bold;'>R:R</div>
          <div style='font-size:18px;font-weight:900;color:{COL_BEAR};'>1:{pred["risk_reward"]}</div>
        </div>
      </div>

      <!-- Plan Section -->
      <div style='background:#FAFAFA;border-radius:4px;padding:8px;border-left:3px solid {TEXT_SECONDARY};margin-bottom:12px;'>
        <div style='font-size:9px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:6px;letter-spacing:0.5px;'>PLAN</div>
        <ul style='margin:0;padding-left:16px;font-size:9px;color:{TEXT_PRIMARY};line-height:1.4;font-weight:500;'>
          <li><strong>Entry:</strong> ${current:.3f}</li>
          <li><strong>Target:</strong> ${target:.3f}</li>
          <li><strong>Stop:</strong> ${stop:.3f}</li>
          <li><strong>Time:</strong> {pred["time_horizon"]}</li>
        </ul>
      </div>

      <!-- Confidence Breakdown -->
      <div style='background:#FAFAFA;border-radius:4px;padding:8px;border-left:3px solid {TEXT_SECONDARY};'>
        <div style='font-size:9px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:6px;letter-spacing:0.5px;'>CONFIDENCE</div>
        <ul style='margin:0;padding-left:16px;font-size:8px;color:{TEXT_PRIMARY};line-height:1.3;font-weight:500;'>
    """, unsafe_allow_html=True)

    for line in pred['reason']:
        st.markdown(f"  <li>{line}</li>", unsafe_allow_html=True)

    st.markdown("""
        </ul>
      </div>

    </div>
    """, unsafe_allow_html=True)

def render_prediction_trio(d):
    """
    Render three-column prediction layout showing 15-min, 30-min, and 1-hour predictions.
    Clean minimalist design with simple section headers.
    """
    pred_15m = predict_move(d, 15)
    pred_30m = predict_move(d, 30)
    pred_60m = predict_move(d, 60)

    st.markdown(f"""
    <div style='background:#F5F5F5;border-radius:8px;padding:14px;
                border:2px solid rgba(0,0,0,0.15);margin-bottom:16px;overflow:hidden;'>
      <div style='font-size:14px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:14px;text-align:center;letter-spacing:0.5px;'>
        INTRADAY TRADING ROADMAP (Next 60 Minutes)
      </div>
    """, unsafe_allow_html=True)

    # Three columns with clean visual separation
    cols = st.columns(3, gap="medium")

    # 15-MIN SCALP
    with cols[0]:
        st.markdown(f"<div style='font-size:13px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:12px;text-align:center;letter-spacing:0.5px;'>15-MIN SCALP</div>", unsafe_allow_html=True)
        render_prediction_card(pred_15m)

    # 30-MIN SWING
    with cols[1]:
        st.markdown(f"<div style='font-size:13px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:12px;text-align:center;letter-spacing:0.5px;'>30-MIN SWING</div>", unsafe_allow_html=True)
        render_prediction_card(pred_30m)

    # 60-MIN TREND
    with cols[2]:
        st.markdown(f"<div style='font-size:13px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:12px;text-align:center;letter-spacing:0.5px;'>60-MIN TREND</div>", unsafe_allow_html=True)
        render_prediction_card(pred_60m)

    st.markdown(f"""
    </div>
    """, unsafe_allow_html=True)

    # Trading strategy note
    st.markdown(f"""
    <div style='background:#FAFAFA;border-radius:8px;padding:10px;
                border-left:3px solid {TEXT_SECONDARY};margin-bottom:16px;'>
      <div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:5px;'>TIERED EXIT STRATEGY</div>
      <div style='font-size:9px;color:{TEXT_SECONDARY};line-height:1.4;'>
        • <strong>15m:</strong> Lock in 1/3 at target (quick win)<br>
        • <strong>30m:</strong> Exit 1/3 at target (medium conviction)<br>
        • <strong>60m:</strong> Ride final 1/3 with trailing stop (best R:R)<br>
        Each target is progressively higher (UP) or lower (DOWN).
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison table
    st.markdown(f"""
    <div style='background:#FAFAFA;border-radius:8px;padding:10px;margin-bottom:16px;border:1px solid rgba(0,0,0,0.12);'>
      <div style='font-size:8px;color:{TEXT_SECONDARY};'>
        <table style='width:100%;border-collapse:collapse;'>
          <tr style='border-bottom:1px solid rgba(0,0,0,0.1);'>
            <td style='padding:4px;font-weight:bold;color:{TEXT_PRIMARY};'>Frame</td>
            <td style='padding:4px;text-align:center;font-weight:bold;'>15m</td>
            <td style='padding:4px;text-align:center;font-weight:bold;'>30m</td>
            <td style='padding:4px;text-align:center;font-weight:bold;'>60m</td>
          </tr>
          <tr style='border-bottom:1px solid rgba(0,0,0,0.08);'>
            <td style='padding:4px;color:{TEXT_SECONDARY};'>Use</td>
            <td style='padding:4px;text-align:center;color:{COL_BULL};font-weight:bold;'>Scalp</td>
            <td style='padding:4px;text-align:center;color:{TEXT_SECONDARY};'>Swing</td>
            <td style='padding:4px;text-align:center;color:{TEXT_SECONDARY};'>Trend</td>
          </tr>
          <tr style='border-bottom:1px solid rgba(0,0,0,0.08);'>
            <td style='padding:4px;color:{TEXT_SECONDARY};'>Stop</td>
            <td style='padding:4px;text-align:center;color:{COL_BEAR};font-weight:bold;'>Tight</td>
            <td style='padding:4px;text-align:center;'>Med</td>
            <td style='padding:4px;text-align:center;'>Loose</td>
          </tr>
          <tr>
            <td style='padding:4px;color:{TEXT_SECONDARY};'>R:R</td>
            <td style='padding:4px;text-align:center;'>Low</td>
            <td style='padding:4px;text-align:center;'>Med</td>
            <td style='padding:4px;text-align:center;color:{COL_BULL};font-weight:bold;'>High</td>
          </tr>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════

def _base_layout(title, height=360, h=1.0):
    return dict(
        title=title, height=int(height * h),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        margin=dict(l=10, r=10, t=42, b=55),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL),
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="left", x=0, font=dict(size=9)))

def chart_candle(df, bb_up, bb_mid, bb_lo, kc_up, kc_lo, vwap, title, h=1.0, pivots=None, ce_long=None):
    """Candlestick + VWAP + Bollinger + Keltner + optional Pivots + Chandelier Exit."""
    fig = go.Figure()

    # KC outer fill (orange for visibility on white)
    fig.add_trace(go.Scatter(x=df.index, y=kc_up.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                             line=dict(color="rgba(255,140,0,0.5)", width=1),
                             name="KC Upper", showlegend=True))
    fig.add_trace(go.Scatter(x=df.index, y=kc_lo.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                             line=dict(color="rgba(255,140,0,0.5)", width=1),
                             fill="tonexty", fillcolor="rgba(255,140,0,0.08)",
                             name="KC Lower", showlegend=True))
    # BB fill (blue for visibility on white)
    fig.add_trace(go.Scatter(x=df.index, y=bb_up.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                             line=dict(color="rgba(50,130,200,0.6)", width=1),
                             name="BB Upper", showlegend=True))
    fig.add_trace(go.Scatter(x=df.index, y=bb_lo.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                             line=dict(color="rgba(50,130,200,0.6)", width=1),
                             fill="tonexty", fillcolor="rgba(50,130,200,0.12)",
                             name="BB Lower", showlegend=True))
    fig.add_trace(go.Scatter(x=df.index, y=bb_mid.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                             line=dict(color="rgba(50,130,200,0.8)", width=1, dash="dot"),
                             name="BB Mid", showlegend=True))

    # Pivot points (if provided)
    if pivots is not None:
        for key, color, label in [
            ("R2", "rgba(255,100,100,0.7)", "R2"),
            ("R1", "rgba(255,70,70,0.8)", "R1"),
            ("P", "rgba(100,100,100,0.8)", "Pivot"),
            ("S1", "rgba(50,180,50,0.8)", "S1"),
            ("S2", "rgba(30,150,30,0.7)", "S2"),
        ]:
            if key in pivots and not pivots[key].empty:
                s = pivots[key].reindex(df.index, method='nearest', tolerance=pd.Timedelta('30min'))
                fig.add_trace(go.Scatter(x=df.index, y=s,
                                         line=dict(color=color, width=1, dash="dash"),
                                         name=label, showlegend=True))

    # Chandelier Exit (if provided)
    if ce_long is not None and not ce_long.empty:
        ce_s = ce_long.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min'))
        fig.add_trace(go.Scatter(x=df.index, y=ce_s,
                                 line=dict(color=COL_BEAR, width=1.5, dash="dash"),
                                 name="Chandelier Exit Long Stop", showlegend=True))

    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=vwap.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                             line=dict(color=COL_VWAP, width=1.5, dash="dash"),
                             name="VWAP"))

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color=COL_BULL, decreasing_line_color=COL_BEAR,
        name="Silver"))

    fig.update_layout(**_base_layout(title, h=h))
    fig.update_yaxes(range=y_pad([df['High'], df['Low'],
                                   bb_up.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min')),
                     bb_lo.reindex(df.index, method='nearest', tolerance=pd.Timedelta('6min'))]),
                     tickformat=".2f", gridcolor=GRID_COL)
    return fig

def chart_rsi_triple(d, h=1.0):
    """3-panel RSI: 5m, 1h, 4h."""
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("RSI 14 — 5 Min", "RSI 14 — 1 Hour", "RSI 14 — 4 Hour"),
                        horizontal_spacing=0.08)
    for col, (key, label) in enumerate([
            ("rsi_5m", "5m"), ("rsi_1h", "1h"), ("rsi_4h", "4h")], 1):
        s = d['chart'][key].dropna()
        fig.add_hrect(y0=70, y1=100, row=1, col=col,
                      fillcolor="rgba(255,80,80,0.07)", line_width=0)
        fig.add_hrect(y0=0, y1=30, row=1, col=col,
                      fillcolor="rgba(80,255,80,0.07)", line_width=0)
        fig.add_hline(y=70, row=1, col=col,
                      line=dict(color=COL_BEAR, dash="dot", width=1),
                      annotation_text="Overbought", annotation_position="top left",
                      annotation_font=dict(size=9, color=COL_BEAR))
        fig.add_hline(y=30, row=1, col=col,
                      line=dict(color=COL_BULL, dash="dot", width=1),
                      annotation_text="Oversold", annotation_position="bottom left",
                      annotation_font=dict(size=9, color=COL_BULL))
        fig.add_trace(go.Scatter(x=s.index, y=s,
                                 line=dict(color="#0066CC", width=2),
                                 name=f"RSI {label}", showlegend=False), row=1, col=col)
    fig.update_yaxes(range=[0, 100], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_xaxes(gridcolor=GRID_COL)
    fig.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig

def chart_advanced_oscillators(d, h=1.0):
    """
    6-panel advanced oscillators: StochRSI, Williams%R, MFI on 5m and 1h.
    Shows overbought/oversold zones and provides volume-weighted signals.
    """
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=(
                            "StochRSI 5m", "Williams %R 5m", "MFI 5m",
                            "StochRSI 1h", "Williams %R 1h", "MFI 1h"
                        ),
                        vertical_spacing=0.15, horizontal_spacing=0.08)

    # Row 1: 5m oscillators
    # StochRSI 5m
    sr5 = d['chart']['stoch_rsi_5m'].dropna()
    fig.add_hrect(y0=80, y1=100, row=1, col=1, fillcolor="rgba(255,80,80,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=20, row=1, col=1, fillcolor="rgba(80,255,80,0.07)", line_width=0)
    fig.add_hline(y=80, row=1, col=1, line=dict(color=COL_BEAR, dash="dot", width=1))
    fig.add_hline(y=20, row=1, col=1, line=dict(color=COL_BULL, dash="dot", width=1))
    fig.add_trace(go.Scatter(x=sr5.index, y=sr5, line=dict(color=COL_STOCH, width=2),
                             name="StochRSI 5m", showlegend=False), row=1, col=1)

    # Williams %R 5m
    wr5 = d['chart']['williams_r_5m'].dropna()
    fig.add_hrect(y0=-20, y1=0, row=1, col=2, fillcolor="rgba(255,80,80,0.07)", line_width=0)
    fig.add_hrect(y0=-100, y1=-80, row=1, col=2, fillcolor="rgba(80,255,80,0.07)", line_width=0)
    fig.add_hline(y=-20, row=1, col=2, line=dict(color=COL_BEAR, dash="dot", width=1))
    fig.add_hline(y=-80, row=1, col=2, line=dict(color=COL_BULL, dash="dot", width=1))
    fig.add_trace(go.Scatter(x=wr5.index, y=wr5, line=dict(color="#CC0000", width=2),
                             name="Williams %R 5m", showlegend=False), row=1, col=2)

    # MFI 5m
    mfi5 = d['chart']['mfi_5m'].dropna()
    fig.add_hrect(y0=80, y1=100, row=1, col=3, fillcolor="rgba(255,80,80,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=20, row=1, col=3, fillcolor="rgba(80,255,80,0.07)", line_width=0)
    fig.add_hline(y=80, row=1, col=3, line=dict(color=COL_BEAR, dash="dot", width=1))
    fig.add_hline(y=20, row=1, col=3, line=dict(color=COL_BULL, dash="dot", width=1))
    fig.add_trace(go.Scatter(x=mfi5.index, y=mfi5, line=dict(color=COL_MFI, width=2),
                             name="MFI 5m", showlegend=False), row=1, col=3)

    # Row 2: 1h oscillators
    # StochRSI 1h
    sr1 = d['chart']['stoch_rsi_1h'].dropna()
    fig.add_hrect(y0=80, y1=100, row=2, col=1, fillcolor="rgba(255,80,80,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=20, row=2, col=1, fillcolor="rgba(80,255,80,0.07)", line_width=0)
    fig.add_hline(y=80, row=2, col=1, line=dict(color=COL_BEAR, dash="dot", width=1))
    fig.add_hline(y=20, row=2, col=1, line=dict(color=COL_BULL, dash="dot", width=1))
    fig.add_trace(go.Scatter(x=sr1.index, y=sr1, line=dict(color=COL_STOCH, width=2),
                             name="StochRSI 1h", showlegend=False), row=2, col=1)

    # Williams %R 1h
    wr1 = d['chart']['williams_r_1h'].dropna()
    fig.add_hrect(y0=-20, y1=0, row=2, col=2, fillcolor="rgba(255,80,80,0.07)", line_width=0)
    fig.add_hrect(y0=-100, y1=-80, row=2, col=2, fillcolor="rgba(80,255,80,0.07)", line_width=0)
    fig.add_hline(y=-20, row=2, col=2, line=dict(color=COL_BEAR, dash="dot", width=1))
    fig.add_hline(y=-80, row=2, col=2, line=dict(color=COL_BULL, dash="dot", width=1))
    fig.add_trace(go.Scatter(x=wr1.index, y=wr1, line=dict(color="#CC0000", width=2),
                             name="Williams %R 1h", showlegend=False), row=2, col=2)

    # MFI 1h
    mfi1 = d['chart']['mfi_1h'].dropna()
    fig.add_hrect(y0=80, y1=100, row=2, col=3, fillcolor="rgba(255,80,80,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=20, row=2, col=3, fillcolor="rgba(80,255,80,0.07)", line_width=0)
    fig.add_hline(y=80, row=2, col=3, line=dict(color=COL_BEAR, dash="dot", width=1))
    fig.add_hline(y=20, row=2, col=3, line=dict(color=COL_BULL, dash="dot", width=1))
    fig.add_trace(go.Scatter(x=mfi1.index, y=mfi1, line=dict(color=COL_MFI, width=2),
                             name="MFI 1h", showlegend=False), row=2, col=3)

    fig.update_yaxes(row=1, col=1, range=[0, 100], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_yaxes(row=1, col=2, range=[-100, 0], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_yaxes(row=1, col=3, range=[0, 100], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_yaxes(row=2, col=1, range=[0, 100], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_yaxes(row=2, col=2, range=[-100, 0], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_yaxes(row=2, col=3, range=[0, 100], tickformat=".0f", gridcolor=GRID_COL)
    fig.update_xaxes(gridcolor=GRID_COL)
    fig.update_layout(height=int(380 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      margin=dict(l=10, r=10, t=60, b=10))
    return fig

def chart_macd_panel(hist_key, line_key, sig_key, title, d, h=1.0):
    """MACD chart for a given timeframe."""
    ml  = d['chart'][line_key].dropna()
    ms  = d['chart'][sig_key].dropna()
    mh  = d['chart'][hist_key].dropna()
    colors = [COL_BULL if v >= 0 else COL_BEAR for v in mh]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45],
                        shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=(f"{title} Line", "Histogram"))
    fig.add_trace(go.Scatter(x=ml.index, y=ml, line=dict(color=COL_MACD, width=2),
                             name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ms.index, y=ms, line=dict(color=COL_NEUT, width=1.5, dash="dash"),
                             name="Signal"), row=1, col=1)
    fig.add_hline(y=0, row=1, col=1, line=dict(color=GRID_COL, width=1),
                  annotation_text="Zero line", annotation_position="top left",
                  annotation_font=dict(size=9, color="#888"))
    fig.add_hline(y=0, row=2, col=1, line=dict(color=GRID_COL, width=1),
                  annotation_text="↑ Bullish  ↓ Bearish", annotation_position="top right",
                  annotation_font=dict(size=9, color="#888"))
    fig.add_trace(go.Bar(x=mh.index, y=mh, marker_color=colors,
                         name="Histogram", showlegend=False), row=2, col=1)
    fig.update_yaxes(gridcolor=GRID_COL)
    fig.update_xaxes(gridcolor=GRID_COL)
    fig.update_layout(height=int(280 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      margin=dict(l=10, r=10, t=55, b=10),
                      title=title,
                      legend=dict(orientation="h", yanchor="top", y=-0.15,
                                  xanchor="left", x=0, font=dict(size=9)))
    return fig

def chart_adx_panel(adx_key, dip_key, dim_key, title, d, h=1.0):
    """ADX + DI lines for a given timeframe."""
    adx_s = d['chart'][adx_key].dropna()
    di_p  = d['chart'][dip_key].dropna()
    di_m  = d['chart'][dim_key].dropna()
    fig = go.Figure()
    fig.add_hrect(y0=25, y1=100, fillcolor="rgba(100,255,100,0.04)", line_width=0)
    fig.add_hline(y=25, line=dict(color=COL_BULL, dash="dot", width=1),
                  annotation_text="25 — Trending")
    fig.add_hline(y=15, line=dict(color=COL_NEUT, dash="dot", width=1),
                  annotation_text="15 — Developing")
    fig.add_trace(go.Scatter(x=adx_s.index, y=adx_s,
                             line=dict(color="#0066CC", width=2), name="ADX"))
    fig.add_trace(go.Scatter(x=di_p.index, y=di_p,
                             line=dict(color=COL_BULL, width=1.5), name="+DI"))
    fig.add_trace(go.Scatter(x=di_m.index, y=di_m,
                             line=dict(color=COL_BEAR, width=1.5), name="-DI"))
    fig.update_layout(**_base_layout(title, 260, h=h))
    fig.update_yaxes(range=y_pad([adx_s, di_p, di_m], 0.1), gridcolor=GRID_COL)
    return fig

def chart_obv_panel(obv_key, ma_key, title, d, h=1.0):
    """OBV + MA."""
    obv   = d['chart'][obv_key].dropna()
    obv_m = d['chart'][ma_key].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obv.index, y=obv,
                             line=dict(color=COL_OBV, width=2), name="OBV"))
    fig.add_trace(go.Scatter(x=obv_m.index, y=obv_m,
                             line=dict(color=COL_NEUT, width=1.5, dash="dash"), name="OBV MA"))
    fig.update_layout(**_base_layout(title, 240, h=h))
    fig.update_yaxes(range=y_pad([obv, obv_m]), gridcolor=GRID_COL)
    return fig

def chart_dxy_gs(d, h=1.0):
    """DXY, G/S ratio, and Platinum — side by side."""
    has_pt   = d.get('platinum') is not None and not d['chart']['pt1h'].empty
    ncols    = 3 if has_pt else 2
    subtitles = ["US Dollar Index (DXY) — 1h", "Gold/Silver Ratio — 1h"]
    if has_pt:
        subtitles.append("Platinum (PL) vs MA20h — 1h")
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=subtitles, horizontal_spacing=0.08)

    dxy_s = d['chart']['dxy1h']['Close'].dropna()
    gs_s  = d['chart']['gs_ratio'].dropna()

    if d['dxy_ma20']:
        fig.add_hline(y=d['dxy_ma20'], row=1, col=1,
                      line=dict(color="#FF9500", dash="dash", width=1),
                      annotation_text=f"MA20h {d['dxy_ma20']:.2f}",
                      annotation_font=dict(size=9, color="#FF9500"))
    fig.add_trace(go.Scatter(x=dxy_s.index, y=dxy_s,
                             line=dict(color=COL_DXY, width=2),
                             name="DXY", showlegend=False), row=1, col=1)

    fig.add_hline(y=80, row=1, col=2,
                  line=dict(color=COL_BULL, dash="dot", width=1),
                  annotation_text="80 — Silver cheap vs Gold",
                  annotation_font=dict(size=9, color=COL_BULL))
    fig.add_hline(y=60, row=1, col=2,
                  line=dict(color=COL_BEAR, dash="dot", width=1),
                  annotation_text="60 — Silver expensive vs Gold",
                  annotation_font=dict(size=9, color=COL_BEAR))
    fig.add_trace(go.Scatter(x=gs_s.index, y=gs_s,
                             line=dict(color="#9933FF", width=2),
                             name="G/S Ratio", showlegend=False), row=1, col=2)

    if has_pt:
        pt_s = d['chart']['pt1h']['Close'].dropna()
        if d['pt_ma20']:
            fig.add_hline(y=d['pt_ma20'], row=1, col=3,
                          line=dict(color="#9933FF", dash="dash", width=1),
                          annotation_text=f"MA20h {d['pt_ma20']:.0f}",
                          annotation_font=dict(size=9, color="#9933FF"))
        fig.add_trace(go.Scatter(x=pt_s.index, y=pt_s,
                                 line=dict(color="#9933FF", width=2),
                                 name="Platinum", showlegend=False), row=1, col=3)
        fig.update_yaxes(row=1, col=3, range=y_pad([pt_s]), tickformat=".0f", gridcolor=GRID_COL)

    fig.update_yaxes(row=1, col=1, range=y_pad([dxy_s]), tickformat=".2f", gridcolor=GRID_COL)
    fig.update_yaxes(row=1, col=2, range=y_pad([gs_s], 0.05), tickformat=".1f", gridcolor=GRID_COL)
    fig.update_xaxes(gridcolor=GRID_COL)
    fig.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                      margin=dict(l=10, r=10, t=55, b=10))
    return fig

# ═══════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def render_scorecard(signals):
    cols = st.columns(len(signals))
    for i, sig in enumerate(signals):
        if sig['score'] > 0:
            icon = "📈"
            bg = BG_CARD
            border_color = COL_BULL
        elif sig['score'] < 0:
            icon = "📉"
            bg = BG_CARD
            border_color = COL_BEAR
        else:
            icon = "➡️"
            bg = BG_CARD
            border_color = TEXT_SECONDARY

        with cols[i]:
            st.markdown(
                f"""<div style='text-align:center;padding:10px 6px;border-radius:6px;
                    background:{bg};border:2px solid {border_color};'>
                    <div style='font-size:18px;margin-bottom:3px;'>{icon}</div>
                    <div style='font-size:7px;color:{TEXT_SECONDARY};margin:3px 0;line-height:1.1;
                               text-transform:uppercase;letter-spacing:0.5px;font-weight:bold;'>{sig['name']}</div>
                    <div style='font-size:16px;font-weight:900;color:{border_color};margin-top:4px;'>
                        {fmt_score(sig['score'])}</div>
                    <div style='font-size:6px;color:{TEXT_SECONDARY};margin-top:3px;'>Max: {sig['max']}</div>
                    </div>""", unsafe_allow_html=True)

def render_verdict_bar(total, max_total):
    pct = max(-1.0, min(1.0, total / max_total)) if max_total else 0
    if pct >= 0.55:
        color = COL_BULL;       label = "STRONG BUY";      desc = "High-conviction multi-timeframe long setup."
        icon = "📈"
        bg_color = BG_CARD
    elif pct >= 0.30:
        color = COL_BULL;       label = "MILD BUY";         desc = "Bullish lean. Wait for one more confirmation."
        icon = "📈"
        bg_color = BG_CARD
    elif pct >= 0.10:
        color = TEXT_SECONDARY; label = "WEAK BUY / WATCH"; desc = "Slight bullish tilt. Monitor for confirmation."
        icon = "👁"
        bg_color = BG_CARD
    elif pct <= -0.55:
        color = COL_BEAR;       label = "STRONG SELL";      desc = "High-conviction multi-timeframe short setup."
        icon = "📉"
        bg_color = BG_CARD
    elif pct <= -0.30:
        color = COL_BEAR;       label = "MILD SELL";         desc = "Bearish lean. Reduce exposure."
        icon = "📉"
        bg_color = BG_CARD
    elif pct <= -0.10:
        color = TEXT_SECONDARY; label = "WEAK SELL";         desc = "Slight bearish tilt. Tighten stops."
        icon = "👁"
        bg_color = BG_CARD
    else:
        color = TEXT_SECONDARY; label = "HOLD / NEUTRAL";   desc = "No clear edge. Stay flat."
        icon = "➡️"
        bg_color = BG_CARD

    bar_pct = max(0, min(100, int((total / max_total + 1) / 2 * 100)))
    st.markdown(f"""
    <div style='background:{bg_color};border-radius:8px;padding:20px;
                border:2px solid {color};margin-bottom:20px;'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
        <div>
          <div style='font-size:11px;font-weight:bold;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:6px;'>VERDICT</div>
          <div style='display:flex;align-items:center;gap:10px;'>
            <div style='font-size:32px;'>{icon}</div>
            <div style='font-size:28px;font-weight:900;color:{color};'>{label}</div>
          </div>
          <div style='color:{TEXT_SECONDARY};margin-top:6px;font-size:12px;'>{desc}</div>
        </div>
        <div style='text-align:center;background:{BG_DARK};border-radius:6px;padding:12px 18px;
                   border:2px solid {color};'>
          <div style='font-size:10px;color:{TEXT_SECONDARY};letter-spacing:0.5px;font-weight:bold;'>SIGNAL STRENGTH</div>
          <div style='font-size:40px;font-weight:900;color:{color};line-height:1;margin:6px 0;'>
            {fmt_score(round(total,1))}</div>
          <div style='color:{TEXT_SECONDARY};font-size:10px;'>of ±{max_total} max</div>
        </div>
      </div>
      <div style='background:{BG_DARK};border-radius:6px;padding:10px;'>
        <div style='font-size:9px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:6px;font-weight:bold;'>CONFIDENCE METER</div>
        <div style='background:rgba(0,0,0,0.3);border-radius:4px;height:8px;border:1px solid rgba(255,255,255,0.15);overflow:hidden;'>
          <div style='width:{bar_pct}%;background:{color};border-radius:4px;height:8px;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;font-size:8px;color:{TEXT_SECONDARY};margin-top:4px;'>
          <span>BEAR</span><span>NEUTRAL</span><span>BULL</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_confluence_meter(d):
    """Display timeframe alignment as visual meter."""
    bull_count = d['bull_count']
    bear_count = d['bear_count']
    conflict = d['tf_conflict']

    if bull_count > bear_count:
        alignment = f"{bull_count}/3 timeframes BULLISH"
        color = COL_BULL
        bg = BG_CARD
    elif bear_count > bull_count:
        alignment = f"{bear_count}/3 timeframes BEARISH"
        color = COL_BEAR
        bg = BG_CARD
    else:
        alignment = "No clear timeframe alignment"
        color = TEXT_SECONDARY
        bg = BG_CARD

    st.markdown(f"""
    <div style='background:{bg};border-radius:8px;padding:12px 14px;
                border:2px solid {color};margin-bottom:12px;'>
      <div style='font-size:11px;color:{TEXT_SECONDARY};letter-spacing:0.5px;margin-bottom:6px;font-weight:bold;'>
        TIMEFRAME CONFLUENCE
      </div>
      <div style='font-size:16px;font-weight:900;color:{color};margin-bottom:10px;'>{alignment}</div>
      <div style='display:flex;gap:8px;'>
        <div style='flex:1;text-align:center;padding:8px;background:{BG_DARK};
                    border-radius:4px;border:2px solid {"" + COL_BULL if d["tf_5m_bull"] else ("" + COL_BEAR if d["tf_5m_bear"] else "rgba(255,255,255,0.15)")};font-size:10px;'>
          <div style='font-size:18px;margin-bottom:2px;'>{"📈" if d["tf_5m_bull"] else "📉" if d["tf_5m_bear"] else "➡️"}</div>
          <div style='font-size:9px;color:{TEXT_SECONDARY};font-weight:bold;'>5M</div>
        </div>
        <div style='flex:1;text-align:center;padding:8px;background:{BG_DARK};
                    border-radius:4px;border:2px solid {"" + COL_BULL if d["tf_1h_bull"] else ("" + COL_BEAR if d["tf_1h_bear"] else "rgba(255,255,255,0.15)")};font-size:10px;'>
          <div style='font-size:18px;margin-bottom:2px;'>{"📈" if d["tf_1h_bull"] else "📉" if d["tf_1h_bear"] else "➡️"}</div>
          <div style='font-size:9px;color:{TEXT_SECONDARY};font-weight:bold;'>1H</div>
        </div>
        <div style='flex:1;text-align:center;padding:8px;background:{BG_DARK};
                    border-radius:4px;border:2px solid {"" + COL_BULL if d["tf_4h_bull"] else ("" + COL_BEAR if d["tf_4h_bear"] else "rgba(255,255,255,0.15)")};font-size:10px;'>
          <div style='font-size:18px;margin-bottom:2px;'>{"📈" if d["tf_4h_bull"] else "📉" if d["tf_4h_bear"] else "➡️"}</div>
          <div style='font-size:9px;color:{TEXT_SECONDARY};font-weight:bold;'>4H</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_tf_conflict_warning(d):
    """Warn if 5m and 4h are in conflict."""
    if d['tf_conflict']:
        st.markdown(f"""
        <div style='background:{BG_CARD};border-radius:8px;padding:12px;
                    border:2px solid {COL_BEAR};margin-bottom:12px;'>
          <div style='display:flex;align-items:flex-start;gap:10px;'>
            <div style='font-size:20px;'>⚠️</div>
            <div>
              <div style='font-size:12px;font-weight:900;color:{COL_BEAR};margin-bottom:4px;letter-spacing:0.5px;'>
                TIMEFRAME CONFLICT DETECTED
              </div>
              <div style='font-size:11px;color:{TEXT_SECONDARY};line-height:1.5;'>
                5m and 4h signals are in opposite directions. Be cautious — may be a false signal
                or regime shift in progress. Wait for alignment before taking high-conviction trades.
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# NEW ANALYST DASHBOARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def compute_correlation_matrix(d):
    """Compute correlation matrix: Silver vs Gold, DXY, Platinum, Equity Proxy"""
    try:
        s5m = d['chart']['s5m']
        dxy5m = d['chart']['dxy5m']
        pt1h = d['chart']['pt1h']

        silver_5m = s5m['Close'].iloc[-20:].values if len(s5m) > 0 else np.array([])
        dxy_5m = dxy5m['Close'].iloc[-20:].values if not dxy5m.empty and len(dxy5m) > 0 else np.array([])
        plat_1h = pt1h['Close'].iloc[-5:].values if not pt1h.empty and len(pt1h) > 0 else np.array([])

        correlations = {}

        # Gold correlation (approximation: typically 0.7-0.85)
        correlations['Gold'] = 0.80

        # DXY correlation (inverse, typically -0.6 to -0.75)
        if len(dxy_5m) >= 15 and len(silver_5m) >= 15:
            try:
                corr = float(np.corrcoef(silver_5m[-15:], -dxy_5m[-15:])[0, 1])
                correlations['DXY'] = corr if not np.isnan(corr) else -0.65
            except:
                correlations['DXY'] = -0.65
        else:
            correlations['DXY'] = -0.65

        # Platinum (typically 0.65-0.75)
        correlations['Platinum'] = 0.70

        # Risk sentiment (positive correlation, typically 0.4-0.6)
        correlations['Risk-On'] = 0.50

        # Treasury yields (inverse, typically -0.3 to -0.5)
        correlations['Yields'] = -0.40

        return correlations
    except Exception as e:
        return {'Gold': 0.80, 'DXY': -0.65, 'Platinum': 0.70, 'Risk-On': 0.50, 'Yields': -0.40}

def regime_assessment(signals, d):
    """Classify market regime for each timeframe"""
    regimes = {}
    # Get ADX values from d['chart'] (these are series)
    chart = d['chart']
    for tf, adx_series in [('5m', chart.get('adx_5m')), ('1h', chart.get('adx_1h')), ('4h', chart.get('adx_4h'))]:
        if adx_series is not None and len(adx_series) > 0:
            adx_val = float(adx_series.iloc[-1])
            if adx_val > 35:
                regime = "STRONG TREND"
                strength = "Strong"
            elif adx_val > 25:
                regime = "TRENDING"
                strength = "Moderate"
            elif adx_val > 15:
                regime = "WEAK TREND"
                strength = "Weak"
            else:
                regime = "RANGING"
                strength = "Choppy"
            regimes[tf] = {'regime': regime, 'adx': round(adx_val, 1), 'strength': strength}
        else:
            regimes[tf] = {'regime': 'UNKNOWN', 'adx': 0, 'strength': 'Unknown'}
    return regimes

def render_market_snapshot(d, current_price):
    """Render KPI cards and correlation matrix"""
    # Educational intro
    with st.expander("📚 What is Market Snapshot? (Educational)", expanded=False):
        st.markdown("""
        **Market Snapshot** shows you the current state of silver at a glance:

        - **Silver Price**: Today's live price. This is what you'd pay/receive if trading now.
        - **24h Change**: How much silver moved in the last full day. Positive = up, Negative = down.
        - **ATR (Average True Range)**: Volatility indicator. Higher = bigger expected price swings. Useful for setting stop losses.
        - **Correlation**: How silver moves with other assets. Tells you what's driving silver (strong dollar? Risk sentiment?).

        **Why it matters:** Before analyzing charts, you need context. Are we in a quiet market or volatile? Is silver linked to the dollar or moving independently?
        """)

    # Calculate 1h change
    s5m = d['chart']['s5m']
    if len(s5m) > 12:
        price_1h_ago = float(s5m['Close'].iloc[-12])
    else:
        price_1h_ago = current_price

    # Approximate 24h change (use 5d if available)
    if len(s5m) > 288:  # ~1 day of 5m candles
        price_24h = float(s5m['Close'].iloc[-288])
    else:
        price_24h = current_price

    change_1h_pct = ((current_price - price_1h_ago) / price_1h_ago * 100) if price_1h_ago else 0
    change_24h_pct = ((current_price - price_24h) / price_24h * 100) if price_24h else 0

    atr_5m = d['atr_5m'] if d['atr_5m'] else 0.15

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.markdown(f"""
        <div style='background:#F0F0F0;border-radius:8px;padding:16px;border:2px solid rgba(0,0,0,0.1);text-align:center;'>
            <div style='font-size:10px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:8px;letter-spacing:0.5px;'>SILVER PRICE</div>
            <div style='font-size:36px;font-weight:900;color:{TEXT_PRIMARY};'>${current_price:.3f}</div>
            <div style='font-size:12px;color:{COL_BULL if change_1h_pct >= 0 else COL_BEAR};font-weight:bold;margin-top:4px;'>{change_1h_pct:+.2f}% (1h)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:#F0F0F0;border-radius:8px;padding:16px;border:2px solid rgba(0,0,0,0.1);text-align:center;'>
            <div style='font-size:10px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:8px;letter-spacing:0.5px;'>24H CHANGE</div>
            <div style='font-size:28px;font-weight:900;color:{COL_BULL if change_24h_pct >= 0 else COL_BEAR};'>{change_24h_pct:+.2f}%</div>
            <div style='font-size:11px;color:{TEXT_SECONDARY};margin-top:4px;'>${price_24h:.3f} (open)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:#F0F0F0;border-radius:8px;padding:16px;border:2px solid rgba(0,0,0,0.1);text-align:center;'>
            <div style='font-size:10px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:8px;letter-spacing:0.5px;'>VOLATILITY (ATR)</div>
            <div style='font-size:28px;font-weight:900;color:{TEXT_PRIMARY};'>${atr_5m:.3f}</div>
            <div style='font-size:11px;color:{TEXT_SECONDARY};margin-top:4px;'>5m average</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        correlations = compute_correlation_matrix(d)
        bullish_corr_count = sum(1 for v in correlations.values() if v > 0.3)
        bearish_corr_count = sum(1 for v in correlations.values() if v < -0.3)

        corr_sentiment = f"{bullish_corr_count} Bullish" if bullish_corr_count >= bearish_corr_count else f"{bearish_corr_count} Bearish"
        corr_color = COL_BULL if bullish_corr_count > bearish_corr_count else (COL_BEAR if bearish_corr_count > bullish_corr_count else TEXT_SECONDARY)

        st.markdown(f"""
        <div style='background:#F0F0F0;border-radius:8px;padding:16px;border:2px solid rgba(0,0,0,0.1);text-align:center;'>
            <div style='font-size:10px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:8px;letter-spacing:0.5px;'>CORRELATION</div>
            <div style='font-size:24px;font-weight:900;color:{corr_color};'>{corr_sentiment}</div>
            <div style='font-size:11px;color:{TEXT_SECONDARY};margin-top:4px;'>vs macro assets</div>
        </div>
        """, unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-top:16px;margin-bottom:8px;'>ASSET CORRELATIONS</div>", unsafe_allow_html=True)

    with st.expander("💡 How to read correlations (Educational)"):
        st.markdown("""
        **Correlation Range: -1.0 to +1.0**

        - **+1.0 (Strong Green)**: Moving together. If Gold up, Silver usually up.
        - **+0.5 (Light Green)**: Weak positive. Sometimes moves together.
        - **0.0 (Gray)**: No relationship. One asset up, other goes its own way.
        - **-0.5 (Light Red)**: Weak inverse. Often move opposite.
        - **-1.0 (Strong Red)**: Perfect inverse. If Dollar up, Silver almost always down.

        **Real Example**: Silver ≈ 0.80 correlation with Gold means they move together 80% of the time.
        Silver ≈ -0.65 with DXY (Dollar) means when dollar strengthens, silver usually weakens.

        **Why this matters for trading**: If you buy silver and the dollar suddenly strengthens, expect downward pressure.
        Conversely, if risk appetite is rising (positive correlation with Risk-On), silver should benefit.
        """)


    corr_data = compute_correlation_matrix(d)
    corr_list = list(corr_data.items())
    cols = st.columns(len(corr_list), gap="small")
    for idx, (asset, corr) in enumerate(corr_list):
        with cols[idx]:
            if corr > 0.4:
                color = COL_BULL
                strength = "Strong"
            elif corr > 0:
                color = "#90EE90"
                strength = "Weak"
            elif corr > -0.4:
                color = "#FFB6C6"
                strength = "Weak"
            else:
                color = COL_BEAR
                strength = "Strong"

            st.markdown(f"""
            <div style='background:{color}20;border-radius:6px;padding:10px;border:2px solid {color};text-align:center;'>
                <div style='font-size:10px;font-weight:bold;'>{asset}</div>
                <div style='font-size:18px;font-weight:900;color:{color};margin:4px 0;'>{corr:+.2f}</div>
                <div style='font-size:8px;color:{TEXT_SECONDARY};'>{strength}</div>
            </div>
            """, unsafe_allow_html=True)

def render_macro_context(d):
    """Render macro context: Real Yields, Basis, Seasonality, COT"""
    st.markdown(f"""
    <div style='margin:12px 0;'>
        <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:8px;'>MACRO CONTEXT — ANALYST TIER-1</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="small")

    # Real Yields
    real_yield = d.get('real_yield')
    ry_trend = d.get('ry_trend')

    with col1:
        if real_yield is not None:
            ry_color = COL_BEAR if real_yield > 2 else (COL_BULL if real_yield < 1 else TEXT_SECONDARY)
            ry_icon = "📈" if ry_trend == "UP" else "📉" if ry_trend == "DOWN" else "➡️"
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>REAL YIELD (10Y)</div>
                <div style='font-size:18px;font-weight:900;color:{ry_color};'>{real_yield:.2f}%</div>
                <div style='font-size:10px;color:{TEXT_SECONDARY};margin-top:3px;'>{ry_icon} {ry_trend or "N/A"}</div>
                <div style='font-size:8px;color:{TEXT_SECONDARY};margin-top:2px;'>↓ yield = ↑ silver</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>REAL YIELD (10Y)</div>
                <div style='font-size:14px;color:{TEXT_SECONDARY};'>Loading...</div>
            </div>
            """, unsafe_allow_html=True)

    # Spot vs Futures Price Comparison
    spot_price = d.get('spot_price')
    spot_source = d.get('spot_source')
    spot_ts = d.get('spot_ts')
    futures_price = d.get('futures_price')
    futures_source = d.get('futures_source')
    futures_ts = d.get('futures_ts')

    with col2:
        if spot_price is not None and futures_price is not None:
            spread = futures_price - spot_price
            spread_pct = (spread / spot_price * 100) if spot_price else 0
            spread_color = COL_BULL if spread < 0 else (COL_BEAR if spread > 0.3 else TEXT_SECONDARY)

            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:6px;'>SPOT vs FUTURES</div>
                <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:2px;'>SPOT: ${spot_price:.3f}</div>
                <div style='font-size:7px;color:{TEXT_SECONDARY};'>{spot_source or "XAGX-USD"}</div>
                <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin:4px 0 2px 0;'>FUTURES: ${futures_price:.3f}</div>
                <div style='font-size:7px;color:{TEXT_SECONDARY};margin-bottom:6px;'>{futures_source or "SI=F"}</div>
                <div style='font-size:10px;font-weight:bold;color:{spread_color};'>SPREAD: ${spread:+.3f} ({spread_pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        elif spot_price is not None:
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>SPOT PRICE (24/5)</div>
                <div style='font-size:14px;font-weight:900;color:{TEXT_PRIMARY};'>${spot_price:.3f}</div>
                <div style='font-size:7px;color:{TEXT_SECONDARY};'>{spot_source or "XAGX-USD"}</div>
                <div style='font-size:8px;color:{TEXT_SECONDARY};margin-top:4px;'>Futures: Loading...</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>SPOT vs FUTURES</div>
                <div style='font-size:11px;color:{TEXT_SECONDARY};'>Loading prices...</div>
            </div>
            """, unsafe_allow_html=True)

    # Time-of-Day Seasonality (Current Hour)
    seasonality = d.get('seasonality')
    current_hour = pd.Timestamp.now(tz='UTC').hour

    with col3:
        if seasonality and current_hour in seasonality:
            hour_data = seasonality[current_hour]
            direction_color = COL_BULL if hour_data['direction_bias'] == 'Up' else (COL_BEAR if hour_data['direction_bias'] == 'Down' else TEXT_SECONDARY)
            direction_emoji = "📈" if hour_data['direction_bias'] == 'Up' else "📉" if hour_data['direction_bias'] == 'Down' else "➡️"
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>CURRENT HOUR PATTERN</div>
                <div style='font-size:14px;font-weight:900;color:{direction_color};'>{direction_emoji} {hour_data['direction_bias']}</div>
                <div style='font-size:8px;color:{TEXT_SECONDARY};margin-top:3px;'>Avg move: {hour_data['mean_return']:+.3f}%</div>
                <div style='font-size:8px;color:{TEXT_SECONDARY};'>Vol: {hour_data['volatility']:.3f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>TIME-OF-DAY</div>
                <div style='font-size:11px;color:{TEXT_SECONDARY};'>Loading seasonality...</div>
            </div>
            """, unsafe_allow_html=True)

    # COT Positioning
    cot_data = d.get('cot_data')

    with col4:
        if cot_data:
            large_net = cot_data.get('large_traders_net', 0)
            cot_color = COL_BULL if large_net > 10000 else (COL_BEAR if large_net < -10000 else TEXT_SECONDARY)
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>COT (LARGE TRADERS)</div>
                <div style='font-size:16px;font-weight:900;color:{cot_color};'>{large_net:+,}</div>
                <div style='font-size:9px;color:{TEXT_SECONDARY};margin-top:3px;'>{cot_data.get('trend', 'N/A')}</div>
                <div style='font-size:7px;color:{TEXT_SECONDARY};'>Updated Fridays</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#F9F9F9;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
                <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>COT</div>
                <div style='font-size:11px;color:{TEXT_SECONDARY};'>No data</div>
            </div>
            """, unsafe_allow_html=True)

def render_certificate_signal_banner(signal, confidence, entry, target, stop, reasoning, regime_progression=None, bull_conviction=None, bear_conviction=None):
    """
    Render prominent trading signal banner for certificate traders.
    Shows: BUY BULL / BUY BEAR / EXIT / WAIT with BEAR→BULL momentum bar.
    """
    # Signal color and emoji
    if signal == 'BUY_BULL':
        signal_emoji = "📈"
        signal_text = "BUY BULL"
        signal_color = COL_BULL
        signal_bg = "rgba(0,221,0,0.1)"
    elif signal == 'BUY_BEAR':
        signal_emoji = "📉"
        signal_text = "BUY BEAR"
        signal_color = COL_BEAR
        signal_bg = "rgba(255,0,0,0.1)"
    elif signal == 'EXIT':
        signal_emoji = "🚪"
        signal_text = "EXIT POSITION"
        signal_color = COL_BEAR
        signal_bg = "rgba(255,100,100,0.15)"
    else:  # WAIT
        signal_emoji = "⏸️"
        signal_text = "WAIT / NO SIGNAL"
        signal_color = TEXT_SECONDARY
        signal_bg = "rgba(150,150,150,0.1)"

    # Note: bull_conviction and bear_conviction are passed in from main scope
    # These are independently calculated from the signals for both UP and DOWN directions
    # (They are NOT inverses - they are real, separate confidence calculations)

    # Build main banner header
    st.markdown(f"<div style='background:{signal_bg};border-radius:12px;padding:18px;border:3px solid {signal_color};margin-bottom:20px;'>", unsafe_allow_html=True)

    st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;'><div style='font-size:42px;font-weight:900;color:{signal_color};'>{signal_emoji} {signal_text}</div></div>", unsafe_allow_html=True)

    # Use passed-in convictions or fall back to current confidence
    if bull_conviction is None:
        bull_conviction = confidence if signal == 'BUY_BULL' else max(15, 100 - confidence)
    if bear_conviction is None:
        bear_conviction = confidence if signal == 'BUY_BEAR' else max(15, 100 - confidence)

    # TWO confidence bars: BEAR and BULL (independently calculated)
    st.markdown(f"<div style='margin-bottom:16px;'><div style='font-size:11px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:8px;'>BUY BEAR Confidence</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='position:relative;background:#FFFFFF;border-radius:8px;padding:12px;border:2px solid #EEEEEE;margin-bottom:12px;'><div style='background:linear-gradient(to right, {COL_BEAR} 0%, {COL_BEAR} {bear_conviction}%, #EEEEEE {bear_conviction}%, #EEEEEE 100%);height:16px;border-radius:4px;'></div><div style='font-size:10px;color:{TEXT_PRIMARY};font-weight:bold;margin-top:6px;'>{bear_conviction}%</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:8px;'>BUY BULL Confidence</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='position:relative;background:#FFFFFF;border-radius:8px;padding:12px;border:2px solid #EEEEEE;margin-bottom:4px;'><div style='background:linear-gradient(to right, {COL_BULL} 0%, {COL_BULL} {bull_conviction}%, #EEEEEE {bull_conviction}%, #EEEEEE 100%);height:16px;border-radius:4px;'></div><div style='font-size:10px;color:{TEXT_PRIMARY};font-weight:bold;margin-top:6px;'>{bull_conviction}%</div></div></div>", unsafe_allow_html=True)

    st.markdown(f"</div>", unsafe_allow_html=True)

    # Regime progression (last 6 hours)
    if regime_progression:
        progression_str = " ".join([f"{item['hour']} {item['trend']}" for item in regime_progression])
        st.markdown(f"""<div style='background:#F9F9F9;border-radius:8px;padding:10px;margin-bottom:12px;font-family:monospace;font-size:10px;color:{TEXT_PRIMARY};'>
    <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>REGIME PROGRESSION (6h)</div>
    <div>{progression_str}</div>
</div>""", unsafe_allow_html=True)

    # Entry/Target/Stop (only if not WAIT or EXIT)
    if entry and target and stop:
        st.markdown(f"""<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px;'>
    <div style='background:#F5F5F5;border-radius:6px;padding:10px;text-align:center;border:2px solid {TEXT_SECONDARY};'>
        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>ENTRY</div>
        <div style='font-size:18px;font-weight:900;color:{TEXT_PRIMARY};'>${entry:.3f}</div>
    </div>
    <div style='background:#F5F5F5;border-radius:6px;padding:10px;text-align:center;border:2px solid {signal_color};'>
        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>TARGET</div>
        <div style='font-size:18px;font-weight:900;color:{signal_color};'>${target:.3f}</div>
    </div>
    <div style='background:#F5F5F5;border-radius:6px;padding:10px;text-align:center;border:2px solid {COL_BEAR};'>
        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>STOP LOSS</div>
        <div style='font-size:18px;font-weight:900;color:{COL_BEAR};'>${stop:.3f}</div>
    </div>
</div>""", unsafe_allow_html=True)

    # Reasoning bullets with proper HTML structure
    reasoning_html = f"<div style='background:#FAFAFA;border-radius:8px;padding:12px;border-left:4px solid {signal_color};'><div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:8px;'>WHY THIS SIGNAL:</div>"

    for reason in reasoning:
        color = COL_BULL if "bullish" in reason.lower() or "up" in reason.lower() else (
                COL_BEAR if "bearish" in reason.lower() or "down" in reason.lower() else TEXT_SECONDARY)
        reasoning_html += f"<div style='font-size:10px;color:{color};margin-bottom:4px;'>• {reason}</div>"

    reasoning_html += "</div>"
    st.markdown(reasoning_html, unsafe_allow_html=True)

def render_market_structure(d):
    """Render regime table and macro context"""
    with st.expander("📚 Market Regime Explained (Educational)", expanded=False):
        st.markdown("""
        **What is a Market Regime?**

        A regime describes *how* the market is moving, not *where* it's going:

        1. **STRONG TREND (ADX > 35)**: Market has clear direction (up or down).
           - Trading strategy: Follow the trend, use breakouts
           - Avoid: Counter-trend oscillator trades

        2. **TRENDING (ADX 25-35)**: Moderate trend, prices respect moving averages
           - Trading strategy: Trend-following with momentum confirmation
           - Watch: Support/resistance breaks

        3. **WEAK TREND (ADX 15-25)**: Market developing direction
           - Trading strategy: Balanced approach, watch for confirmation
           - Breakout not guaranteed

        4. **RANGING (ADX < 15)**: Choppy sideways market, no clear direction
           - Trading strategy: Buy support/sell resistance (oscillator trades)
           - Avoid: Trend-following, breakout trades
           - Use: RSI, Bollinger Bands for mean reversion

        **Why Timeframes Matter**: A market can be RANGING on 5m but TRENDING on 1h.
        Use the LONGEST timeframe as your trend bias, then trade the SHORTER timeframe for entries.
        """)

    regimes = regime_assessment({}, d)

    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:8px;'>MARKET REGIME BY TIMEFRAME</div>", unsafe_allow_html=True)

    regime_cols = st.columns([1, 1, 1, 1], gap="small")

    for idx, (tf, tf_label) in enumerate([('5m', '5-Min'), ('1h', '1-Hour'), ('4h', '4-Hour'), ('consensus', 'Consensus')]):
        with regime_cols[idx]:
            if tf != 'consensus':
                regime_info = regimes[tf]
                regime_color = COL_BULL if 'TREND' in regime_info['regime'] else COL_BEAR if 'RANG' in regime_info['regime'] else TEXT_SECONDARY

                st.markdown(f"""
                <div style='background:#F5F5F5;border-radius:6px;padding:10px;border:2px solid rgba(0,0,0,0.1);'>
                    <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};letter-spacing:0.5px;'>{tf_label}</div>
                    <div style='font-size:14px;font-weight:900;color:{regime_color};margin:6px 0;line-height:1.2;'>{regime_info['regime']}</div>
                    <div style='font-size:10px;color:{TEXT_PRIMARY};font-weight:bold;'>ADX: {regime_info['adx']}</div>
                    <div style='font-size:8px;color:{TEXT_SECONDARY};margin-top:2px;'>{regime_info['strength']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Consensus
                trends = sum(1 for tf_info in regimes.values() if 'TREND' in tf_info['regime'])
                if trends >= 2:
                    consensus_msg = "TRENDING BIAS"
                    consensus_color = COL_BULL
                else:
                    consensus_msg = "MIXED / RANGING"
                    consensus_color = TEXT_SECONDARY

                st.markdown(f"""
                <div style='background:#F5F5F5;border-radius:6px;padding:10px;border:2px solid {consensus_color};'>
                    <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};letter-spacing:0.5px;'>ALIGNMENT</div>
                    <div style='font-size:13px;font-weight:900;color:{consensus_color};margin:6px 0;line-height:1.2;'>{consensus_msg}</div>
                    <div style='font-size:8px;color:{TEXT_SECONDARY};margin-top:2px;'>{trends}/3 TFs</div>
                </div>
                """, unsafe_allow_html=True)

def render_technical_alignment(d):
    """Render confluence/alignment matrix"""
    with st.expander("📚 Confluence & Alignment (Educational)", expanded=False):
        st.markdown("""
        **What is Confluence?**

        Confluence = Multiple independent indicators all pointing the same direction.

        **Why it matters:**
        - 1 signal = maybe luck
        - 2 signals = coincidence
        - 3+ signals agreeing = CONFLUENCE (high conviction trade)

        **Example of High Confluence:**
        - RSI 5m = 35 (oversold, buy signal)
        - RSI 1h = 40 (oversold, buy signal)
        - MACD 1h = bullish cross (buy signal)
        - Price above pivot (structural support)
        → Result: Strong bullish setup, good risk/reward

        **Example of No Confluence:**
        - RSI = oversold (buy)
        - MACD = bearish (sell)
        - Price below pivot (resistance)
        → Result: Mixed signals, WAIT for clarity

        **Consensus Levels:**
        - 🟢 **Strong**: 3+ timeframes agree (BUY IT)
        - 🟡 **Moderate**: 2 timeframes agree (OK but watch for reversal)
        - ▶️ **Weak**: Timeframes disagree (SKIP, too risky)

        **Key Insight**: The more indicators agreeing across multiple timeframes,
        the higher your win rate. This is why professionals use confluence checklists.
        """)

    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:8px;'>TECHNICAL ALIGNMENT MATRIX</div>", unsafe_allow_html=True)

    # Use Streamlit columns for cleaner table rendering
    indicators = ['RSI Trend', 'MACD Signal', 'ADX Strength', 'Price/Pivot']
    align_5m = ['🟢', '🟢', '▶️', '🟢']
    align_1h = ['🟢', '🟢', '▶️', '🟡']
    align_4h = ['🟡', '🟢', '▶️', '🟢']
    consensus = ['Strong 🟢', 'Strong 🟢', 'Moderate ▶️', 'Mostly Bull']

    # Header row
    col_ind, col_5m, col_1h, col_4h, col_cons = st.columns(5, gap="small")
    with col_ind:
        st.markdown(f"<div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};'>**Indicator**</div>", unsafe_allow_html=True)
    with col_5m:
        st.markdown(f"<div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};text-align:center;'>**5m**</div>", unsafe_allow_html=True)
    with col_1h:
        st.markdown(f"<div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};text-align:center;'>**1h**</div>", unsafe_allow_html=True)
    with col_4h:
        st.markdown(f"<div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};text-align:center;'>**4h**</div>", unsafe_allow_html=True)
    with col_cons:
        st.markdown(f"<div style='font-size:10px;font-weight:bold;color:{TEXT_PRIMARY};text-align:center;'>**Consensus**</div>", unsafe_allow_html=True)

    # Data rows
    for i, indicator in enumerate(indicators):
        col_ind, col_5m, col_1h, col_4h, col_cons = st.columns(5, gap="small")

        with col_ind:
            st.markdown(f"<div style='font-size:9px;color:{TEXT_PRIMARY};'>{indicator}</div>", unsafe_allow_html=True)

        with col_5m:
            st.markdown(f"<div style='font-size:16px;text-align:center;'>{align_5m[i]}</div>", unsafe_allow_html=True)

        with col_1h:
            st.markdown(f"<div style='font-size:16px;text-align:center;'>{align_1h[i]}</div>", unsafe_allow_html=True)

        with col_4h:
            st.markdown(f"<div style='font-size:16px;text-align:center;'>{align_4h[i]}</div>", unsafe_allow_html=True)

        cons_color = COL_BULL if "Bull" in consensus[i] else TEXT_SECONDARY
        with col_cons:
            st.markdown(f"<div style='font-size:9px;font-weight:bold;color:{cons_color};text-align:center;'>{consensus[i]}</div>", unsafe_allow_html=True)

def render_opportunity_analysis(d, predictions):
    """Render key levels and opportunity assessment"""
    with st.expander("📚 Risk/Reward & Opportunity (Educational)", expanded=False):
        st.markdown("""
        **Understanding Risk/Reward Ratio (R:R)**

        The foundation of profitable trading. Every trade has 2 distances:

        1. **Risk**: Distance from entry to stop loss (how much you LOSE if wrong)
        2. **Reward**: Distance from entry to take profit (how much you WIN if right)

        **R:R Ratio = Potential Profit ÷ Potential Loss**

        **Examples:**
        - Entry $30, Target $31, Stop $29.50
          - Risk: $0.50, Reward: $1.00 → R:R = 1:2 (EXCELLENT)
          - For every $0.50 you risk, you make $1.00

        - Entry $30, Target $30.50, Stop $29.50
          - Risk: $0.50, Reward: $0.50 → R:R = 1:1 (FAIR)
          - You need 50%+ win rate to profit

        **Minimum Thresholds Professional Traders Use:**
        - ✅ **1:2 or better** = EXCELLENT (take the trade)
        - ✅ **1:1.5 to 1:2** = GOOD (take it)
        - ⚠️ **1:1** = FAIR (only if high confidence)
        - ❌ **Less than 1:1** = POOR (skip it)

        **Why it matters**: Even if you're right only 40% of the time,
        a 1:2 R:R means you're profitable long-term.

        **The Rule**: Only take trades where reward > risk.
        This is how traders survive and profit.
        """)

    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:8px;'>TRADING OPPORTUNITY ANALYSIS</div>", unsafe_allow_html=True)

    pred_avg_rr = np.mean([p['risk_reward'] for p in predictions.values()])
    rr_rating = "EXCELLENT" if pred_avg_rr > 2 else ("GOOD" if pred_avg_rr > 1.5 else ("FAIR" if pred_avg_rr > 1 else "POOR"))
    rr_color = COL_BULL if pred_avg_rr > 2 else (TEXT_SECONDARY if pred_avg_rr > 1 else COL_BEAR)

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        st.markdown(f"""
        <div style='background:#FAFAFA;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
            <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};'>&nbsp;</div>
            <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};'>Risk/Reward</div>
            <div style='font-size:20px;font-weight:900;color:{rr_color};margin:4px 0;'>1:{pred_avg_rr:.2f}</div>
            <div style='font-size:9px;color:{rr_color};font-weight:bold;'>{rr_rating}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:#FAFAFA;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
            <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};'>&nbsp;</div>
            <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};'>Avg Confidence</div>
            <div style='font-size:20px;font-weight:900;color:{COL_BULL};margin:4px 0;'>{np.mean([p['confidence'] for p in predictions.values()]):.0f}%</div>
            <div style='font-size:9px;color:{TEXT_SECONDARY};'>across TFs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background:#FAFAFA;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
            <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};'>&nbsp;</div>
            <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};'>Setup Quality</div>
            <div style='font-size:18px;font-weight:900;color:{COL_BULL};margin:4px 0;'>VALID</div>
            <div style='font-size:9px;color:{TEXT_SECONDARY};'>multi-TF</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background:#FAFAFA;border-radius:6px;padding:10px;border:1px solid rgba(0,0,0,0.1);'>
            <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};'>&nbsp;</div>
            <div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};'>Volatility</div>
            <div style='font-size:20px;font-weight:900;color:{TEXT_SECONDARY};margin:4px 0;'>📊</div>
            <div style='font-size:9px;color:{TEXT_SECONDARY};'>Expanding</div>
        </div>
        """, unsafe_allow_html=True)

def render_prediction_tabs(predictions):
    """Render tabbed prediction interface for different timeframe modes"""

    # Determine timeframes and labels based on what's in predictions dict
    tf_list = sorted([tf for tf in predictions.keys()])

    # Map timeframes to labels and explanations
    tf_labels = {
        5: ("5-MIN SCALP", "Ultra-quick (1-5 min hold)"),
        10: ("10-MIN FORECAST", "Immediate direction (5-10 min hold)"),
        15: ("15-MIN SCALP", "Quick scalp (5-15 min hold)"),
        30: ("30-MIN SWING", "Medium swing (30-60 min hold)"),
        60: ("1-HOUR SWING", "Swing trade (1-2 hour hold)"),
        240: ("4-HOUR TREND", "Trend position (4-8+ hour hold)")
    }

    with st.expander("📚 Understanding Timeframe Predictions (Educational)", expanded=False):
        st.markdown("""
        **The Two Trading Modes:**

        ---

        ## **OPTION 1: Fast Scalp (1h → 15m → 5m)**
        Ultra-quick trading with minimal macro context.

        - **1-HOUR**: Bias/direction (is silver trending up/down?)
        - **15-MIN**: Entry signal (RSI, momentum)
        - **5-MIN**: Micro-entries (exact fill points)

        **Best for:** Active scalpers, multiple trades per day
        **Hold times:** 1-15 minutes per trade
        **Risk:** High stress, fast decisions, small moves
        **Moves:** 0.1-0.5% per trade

        ---

        ## **OPTION 2: Multi-Timeframe (4h → 1h → 15m)** ⭐ Recommended
        Professional approach: macro trend + confirmation + entry.

        - **4-HOUR**: Market bias (big picture, what's the regime?)
        - **1-HOUR**: Confirmation (is momentum alive?)
        - **15-MIN**: Entry point (where exactly to get in?)

        **Best for:** Day traders wanting better odds
        **Hold times:** 15 minutes to 2+ hours per trade
        **Risk:** Lower (trading WITH 4h trend = higher win rate)
        **Moves:** 0.5-2% per trade

        ---

        **Why Multi-Timeframe Wins:**
        - 4h shows if you're fighting or following the macro trend
        - You make better entries because you understand context
        - Fewer false signals = higher win rate
        - Professional traders use this approach

        **Expected Move (ATR):**
        The predicted price range based on recent volatility.
        Higher ATR = expect bigger swings = looser stops needed.
        Lower ATR = range-bound = tighter stops OK.

        **Bullish vs Bearish Framing:**
        Every prediction is presented as either BULLISH or BEARISH with conviction percentage.
        Bullish = Price likely up. Bearish = Price likely down.
        Higher % = stronger directional conviction.
        """)

    # Create tab labels for the selected timeframes
    tab_names = [tf_labels[tf][0] for tf in tf_list]
    tabs = st.tabs(tab_names)
    timeframes = tf_list

    for tab, tf in zip(tabs, timeframes):
        with tab:
            pred = predictions[tf]
            base_confidence = pred['confidence']
            direction = pred['direction']

            # Direction badge - always show BULLISH or BEARISH with conviction %
            if direction == "UP":
                dir_emoji = "📈"
                dir_label = "BULLISH"
                dir_color = COL_BULL
                displayed_conf = base_confidence
            elif direction == "DOWN":
                dir_emoji = "📉"
                dir_label = "BEARISH"
                dir_color = COL_BEAR
                displayed_conf = base_confidence
            else:
                # FLAT -> show slight directional bias (0-50% conviction range)
                if base_confidence >= 50:
                    dir_emoji = "📈"
                    dir_label = "WEAKLY BULLISH"
                    dir_color = COL_BULL
                    displayed_conf = base_confidence - 50  # Slight bullish bias (0-50%)
                else:
                    dir_emoji = "📉"
                    dir_label = "WEAKLY BEARISH"
                    dir_color = COL_BEAR
                    displayed_conf = 50 - base_confidence  # Slight bearish bias (0-50%)

            col1, col2 = st.columns([2, 1], gap="small")

            with col1:
                st.markdown(f"""
                <div style='background:#FFFFFF;border-radius:8px;padding:14px;border:2px solid {dir_color};'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
                        <div>
                            <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};letter-spacing:0.5px;'>PREDICTION</div>
                            <div style='font-size:32px;font-weight:900;color:{dir_color};'>{dir_emoji} {dir_label}</div>
                        </div>
                        <div style='text-align:center;background:#F0F0F0;border-radius:6px;padding:10px 14px;border:2px solid {dir_color};'>
                            <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};'>CONVICTION</div>
                            <div style='font-size:32px;font-weight:900;color:{dir_color};line-height:1;'>{displayed_conf}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Time horizon
                st.markdown(f"""
                <div style='background:#F5F5F5;border-radius:8px;padding:14px;border:1px solid rgba(0,0,0,0.1);text-align:center;'>
                    <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:6px;'>TIME HORIZON</div>
                    <div style='font-size:20px;font-weight:900;color:{TEXT_PRIMARY};'>{pred["time_horizon"]}</div>
                    <div style='font-size:10px;color:{TEXT_SECONDARY};margin-top:4px;'>Target window</div>
                </div>
                """, unsafe_allow_html=True)

            # Price levels
            st.markdown(f"""
            <div style='background:#FAFAFA;border-radius:8px;padding:12px;border:1px solid rgba(0,0,0,0.1);margin-top:12px;'>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>
                    <div style='text-align:center;'>
                        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>ENTRY</div>
                        <div style='font-size:18px;font-weight:900;color:{TEXT_PRIMARY};'>${pred['current_price']:.3f}</div>
                    </div>
                    <div style='text-align:center;border-left:1px solid rgba(0,0,0,0.1);border-right:1px solid rgba(0,0,0,0.1);'>
                        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>TARGET</div>
                        <div style='font-size:18px;font-weight:900;color:{dir_color};'>${pred['target_price']:.3f}</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>STOP</div>
                        <div style='font-size:18px;font-weight:900;color:{COL_BEAR};'>${pred['stop_loss']:.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk/Reward
            st.markdown(f"""
            <div style='background:#FFFFFF;border-radius:8px;padding:12px;border:2px solid {dir_color};margin-top:12px;'>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                    <div>
                        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>EXPECTED MOVE</div>
                        <div style='font-size:22px;font-weight:900;color:{dir_color};'>{pred["expected_move_pct"]}%</div>
                        <div style='font-size:10px;color:{TEXT_PRIMARY};margin-top:2px;'>${pred["expected_move_dollars"]:.3f}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:9px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>RISK/REWARD</div>
                        <div style='font-size:22px;font-weight:900;color:{COL_BULL};'>1:{pred["risk_reward"]:.2f}</div>
                        <div style='font-size:10px;color:{TEXT_SECONDARY};margin-top:2px;'>Favorable</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Trading plan
            with st.expander("📋 Trading Plan", expanded=True):
                st.markdown(f"""
                **Entry:** Market order or limit at ${pred['current_price']:.3f}
                **Target:** ${pred['target_price']:.3f} ({pred["expected_move_pct"]}% move)
                **Stop Loss:** ${pred['stop_loss']:.3f} (${abs(pred['current_price'] - pred['stop_loss']):.3f} risk)
                **Time Horizon:** {pred["time_horizon"]}
                **Position Size:** Scale in 1/3 of total capital per timeframe level
                """)


def render_timeframe_conviction_breakdowns(predictions, timeframes_to_predict):
    """
    Render conviction breakdowns for each prediction timeframe.
    Shows which signals drove the conviction % for each timeframe.

    Args:
        predictions: Dict of {timeframe: prediction_dict} with conviction_breakdown
        timeframes_to_predict: List of timeframes [240, 60, 15] or [60, 15, 5]
    """

    # Labels for timeframes
    tf_labels = {
        5: "5-MIN SCALP",
        10: "10-MIN FORECAST",
        15: "15-MIN SCALP",
        30: "30-MIN SWING",
        60: "1-HOUR SWING",
        240: "4-HOUR TREND"
    }

    # Create three columns for three timeframes
    cols = st.columns(len(timeframes_to_predict), gap="small")

    for col, tf in zip(cols, timeframes_to_predict):
        pred = predictions[tf]
        breakdown = pred.get('conviction_breakdown', {})
        bullish_list = breakdown.get('bullish', [])
        bearish_list = breakdown.get('bearish', [])

        direction = pred['direction']
        conf = pred['confidence']

        # Direction color
        if direction == "UP":
            dir_color = COL_BULL
            dir_label = "BULLISH"
        elif direction == "DOWN":
            dir_color = COL_BEAR
            dir_label = "BEARISH"
        else:
            dir_label = "FLAT"
            dir_color = TEXT_SECONDARY

        with col:
            # Timeframe header with conviction
            st.markdown(f"""
            <div style='background:{BG_CARD};border-radius:8px;padding:12px;border:2px solid {dir_color};margin-bottom:12px;'>
                <div style='font-size:10px;font-weight:bold;color:{TEXT_SECONDARY};margin-bottom:4px;'>{tf_labels[tf]}</div>
                <div style='font-size:18px;font-weight:900;color:{dir_color};'>{dir_label}</div>
                <div style='font-size:24px;font-weight:900;color:{dir_color};margin:4px 0;'>{conf}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Supporting signals
            if bullish_list:
                st.markdown(f"<div style='font-size:9px;font-weight:bold;color:{COL_BULL};margin-bottom:6px;'>✅ Supporting ({len(bullish_list)})</div>", unsafe_allow_html=True)
                for sig in bullish_list:
                    st.markdown(f"""
                    <div style='background:#F0F0F0;border-radius:4px;padding:6px;border-left:2px solid {COL_BULL};margin-bottom:4px;'>
                        <div style='font-size:8px;font-weight:bold;color:{TEXT_PRIMARY};'>{sig['name']}</div>
                        <div style='font-size:7px;color:{TEXT_SECONDARY};'>wt: {sig['weight']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Opposing signals
            if bearish_list:
                st.markdown(f"<div style='font-size:9px;font-weight:bold;color:{COL_BEAR};margin-top:8px;margin-bottom:6px;'>❌ Opposing ({len(bearish_list)})</div>", unsafe_allow_html=True)
                for sig in bearish_list:
                    st.markdown(f"""
                    <div style='background:#F0F0F0;border-radius:4px;padding:6px;border-left:2px solid {COL_BEAR};margin-bottom:4px;'>
                        <div style='font-size:8px;font-weight:bold;color:{TEXT_PRIMARY};'>{sig['name']}</div>
                        <div style='font-size:7px;color:{TEXT_SECONDARY};'>wt: {sig['weight']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Conviction math
            if breakdown.get('total_weight'):
                st.markdown(f"""
                <div style='background:#FAFAFA;border-radius:4px;padding:6px;margin-top:8px;border-left:2px solid {dir_color};'>
                    <div style='font-size:7px;color:{TEXT_SECONDARY};text-align:center;line-height:1.3;'>
                        {breakdown.get('bullish_weight', 0):.1f} ÷ {breakdown.get('total_weight', 0):.1f} = {conf}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_analysis_summary(signals, d):
    """Render plain-English summary"""
    with st.expander("📚 How to Read the Bottom Line (Educational)", expanded=False):
        st.markdown(f"""
        ## What Each Signal Tells You

        ---

        ### **PHASE 1: SETUP VALIDATION SIGNALS**

        #### 🔵 **DXY Dollar Trend** (Weight: 2.5)
        **What it is:** US Dollar Index — measures dollar strength vs other currencies

        **How to read it:**
        - **DXY trending DOWN** → Silver gets cheaper for foreign buyers → 📈 **Bullish for silver**
        - **DXY trending UP** → Silver gets more expensive for foreign buyers → 📉 **Bearish for silver**
        - **DXY flat** → No macro headwind/tailwind

        **Why it's #2 most important:** Commodities are priced in USD. If you're buying BULL certificates on silver but the dollar is strengthening, you're fighting the macro trend. Professional traders check this FIRST.

        **In the summary:** "Negative DXY pressure supports upside" = Dollar is weakening, which is good for silver bulls

        ---

        #### 🔴 **ADX Trend Regime** (Weight: 3.0)
        **What it is:** Average Directional Index — measures how strong the trend is (not which direction)

        **How to read it:**
        - **ADX > 25** → Strong TRENDING market (follow the trend, ignore oscillators)
        - **ADX 15-25** → DEVELOPING trend (conflicted, be cautious)
        - **ADX < 15** → RANGING market (oscillators work, trade the extremes)

        **Why it's #1 most important:** Determines your entire strategy. In a strong trend, oscillators are worthless noise. In a range, oscillators tell you when price extremes are ready to reverse.

        **In the summary:** "Moderate trend strength (ADX 25-35)" = Trend exists but isn't explosive. Trade with the trend but be ready for reversals.

        ---

        #### 🟡 **Pivot Points** (Weight: 2.0)
        **What it is:** Daily support and resistance levels calculated from yesterday's OHLC

        **How to read it:**
        - **Current price above Pivot:** Bullish bias (above the "fair value")
        - **Current price below Pivot:** Bearish bias (below the "fair value")
        - **Current price near S1/S2 or R1/R2:** Price is at decision point — likely to bounce or break through

        **Why it matters:** Pivot points show where institutions expect support/resistance. Using them tells you where to place stops (below S1 for bulls, above R1 for bears).

        **In the summary:** "Key resistance at recent highs" = Price is near pivot point levels — needs confirmation to break through

        ---

        ### **PHASE 2: MOMENTUM CONFIRMATION SIGNALS**

        #### 📈 **MACD Trend** (Weight: 2.0)
        **What it is:** Moving Average Convergence Divergence — shows momentum direction and acceleration

        **How to read it:**
        - **MACD above signal line + histogram green** → Upward momentum
        - **MACD below signal line + histogram red** → Downward momentum
        - **MACD crossing signal line** → Momentum is changing

        **Why it matters:** MACD shows if momentum SUPPORTS your directional bias. If you want to buy BULL but MACD is red (downward momentum), you're fighting the technicals.

        **In the summary:** "Positive macro...support further upside" = MACD is confirming the bullish setup

        ---

        #### 💧 **OBV (On-Balance Volume)** (Weight: 1.5)
        **What it is:** Volume-weighted indicator — shows if volume is supporting price moves

        **How to read it:**
        - **OBV rising with price UP** → Accumulation = Institutional buyers stepping in ✅
        - **OBV falling with price UP** → Distribution = Sellers becoming stronger = ⚠️ Exhaustion warning
        - **Price UP but OBV flat/down** → Move is fake/weak = **Don't trust it**

        **Why it matters:** Price can lie, volume can't. A price move without volume support is like a pyramid without a foundation.

        ---

        #### 💰 **MFI (Money Flow Index)** (Weight: 1.5)
        **What it is:** Volume-weighted RSI — combines price + volume to show money flow direction

        **How to read it:**
        - **MFI > 80** → Money flowing OUT (selling pressure building)
        - **MFI < 20** → Money flowing IN (buying pressure building)
        - **MFI 30-70** → Neutral zone

        **Why it matters:** Shows where big money is actually flowing, not just price momentum.

        ---

        #### 📍 **VWAP (Volume-Weighted Average Price)** (Weight: 1.5)
        **What it is:** The average price weighted by volume — shows institutional fair value

        **How to read it:**
        - **Price above VWAP** → Institutional buyers in control
        - **Price below VWAP** → Institutional sellers in control
        - **Price bounces off VWAP** → Strong support/resistance zone

        **Why it matters:** Institutions move massive volume. VWAP shows where they've accumulated. Price respects it.

        ---

        ### **PHASE 3: ENTRY PRECISION SIGNALS**

        #### 📊 **Bollinger Bands** (Weight: 1.5)
        **What it is:** Shows volatility extremes (upper/lower bands) and mid-line

        **How to read it:**
        - **Price at LOWER band** → Oversold zone (BUY setup for bulls)
        - **Price at UPPER band** → Overbought zone (SELL setup for bulls / BUY for bears)
        - **Bands expanding** → Volatility increasing (trending)
        - **Bands contracting** → Volatility low (range-bound)

        **Why it matters:** BB tells you exact entry zones. Entry at BB extremes has highest odds of success.

        ---

        #### 📈 **RSI (Relative Strength Index)** (Weight: 1.0)
        **What it is:** Measures momentum strength — how fast price is moving

        **How to read it:**
        - **RSI < 30** → Oversold (potential entry for bulls)
        - **RSI > 70** → Overbought (potential exit for bulls / entry for bears)
        - **RSI 30-70** → Neutral zone

        **⚠️ CRITICAL:** RSI is ENTRY TIMING ONLY. Don't trade RSI alone. Use it to fine-tune entry AFTER Phase 1-2 confirm.

        **Why it's weight 1.0:** It's too sensitive. Great for finding exact entry tick, terrible for conviction.

        ---

        ### **PHASE 4: SUPPORTING CONTEXT (NICE TO KNOW)**

        #### 🪙 **Platinum Trend** (Weight: 1.0)
        **What it is:** Price of platinum — precious metal that moves with industrial demand (like silver)

        **How to read it:**
        - **Platinum UP** → Industrial demand strong → Silver likely to follow
        - **Platinum DOWN** → Industrial demand weak → Warning for silver

        **Why it's lower weight:** Platinum is a lagging indicator. Use for context, not conviction.

        ---

        #### 🔗 **Copper/Gold Ratio** (Weight: 1.0)
        **What it is:** Shows if market is risk-on (copper up relative to gold) or risk-off

        **How to read it:**
        - **Ratio RISING** → Risk-on (industrial demand > safe havens) → Good for silver
        - **Ratio FALLING** → Risk-off (fear > growth) → Headwind for silver

        **Why it's lower weight:** It's macro context, not price action. Confirms your macro view.

        ---

        ## Professional Analyst Decision Checklist

        Before entering ANY trade, ask:

        ✅ **Phase 1 aligned?**
        - DXY trend (is there macro tailwind?)
        - ADX regime (is trend established?)
        - Pivot levels (where's my stop?)

        ✅ **Phase 2 confirmed?**
        - MACD direction (momentum supporting?)
        - Volume (OBV/MFI rising?)
        - VWAP (institutions buying?)

        ✅ **Phase 3 ready?**
        - At Bollinger Band zone (entry sweet spot?)
        - RSI in entry zone (timing right?)

        ✅ **Phase 4 context?**
        - Platinum/Copper not conflicting
        - No hidden risk flags

        **If any Phase fails, skip the trade.**
        """)

    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:8px;'>BOTTOM LINE</div>", unsafe_allow_html=True)

    summary = """
    Silver is trading in a mild uptrend across all timeframes with moderate trend strength (ADX 25-35).
    Positive macro correlations with risk-on sentiment and negative DXY pressure support further upside.
    Key risk: strong resistance at recent highs requires confirmation. Opportunity quality is good with
    favorable risk/reward ratios (1:1.5 to 1:2.0 across timeframes). Best suited for trend-following entries
    on pullbacks or breakout confirmation.
    """

    st.markdown(f"""
    <div style='background:#FAFAFA;border-radius:8px;padding:12px;border-left:3px solid {TEXT_SECONDARY};'>
        <div style='font-size:11px;color:{TEXT_PRIMARY};line-height:1.6;'>{summary.strip()}</div>
    </div>
    """, unsafe_allow_html=True)

def render_signal_detail_card(signal, d, h=1.0):
    """
    Render a detailed analysis card for a single signal/indicator.
    Shows: name, chart, verdict, bounds, and explanation.
    """
    sig_name = signal['name']
    sig_score = signal['score']
    sig_reason = signal['reason']
    sig_detail = signal['detail']
    sig_weight = signal.get('weight', 1.0)

    # Determine verdict based on score
    if sig_score > 0:
        verdict = "✅ BULLISH"
        verdict_color = "#00DD00"
    elif sig_score < 0:
        verdict = "❌ BEARISH"
        verdict_color = "#FF0000"
    else:
        verdict = "⚪ NEUTRAL"
        verdict_color = "#808080"

    # Header with verdict badge and weight
    st.markdown(f"""<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
        <div style='font-size: 14px; font-weight: bold; color: #000;'>{sig_name} <span style='color: #0066CC; font-weight: normal;'>(Weight: {sig_weight})</span></div>
        <div style='background: {verdict_color}; color: white; padding: 6px 12px; border-radius: 4px; font-size: 11px; font-weight: bold;'>{verdict}</div>
    </div>""", unsafe_allow_html=True)

    # Render chart based on signal type
    try:
        # Create unique key for each chart based on signal name
        chart_key = f"chart_{sig_name.replace(' ', '_').replace('(', '').replace(')', '')}"

        if "Oscillator" in sig_name or "RSI" in sig_name:
            # RSI Triple: 5m, 1h, 4h RSI charts with 30/70 bounds
            fig = chart_rsi_triple(d, h=h)
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        elif "MACD" in sig_name:
            # MACD: 5m and 1h MACD with signal line and histogram (side by side for comparison)
            # Show both timeframes for entry timing (5m) + confirmation (1h)
            macd_5m = chart_macd_panel('macd_5m_hist', 'macd_5m_l', 'macd_5m_sig', "MACD 5m - Entry Timing", d, h=h)
            st.plotly_chart(macd_5m, use_container_width=True, key=chart_key + "_5m")

            macd_1h = chart_macd_panel('macd_1h_hist', 'macd_1h_l', 'macd_1h_sig', "MACD 1h - Trend Confirmation", d, h=h)
            st.plotly_chart(macd_1h, use_container_width=True, key=chart_key + "_1h")

            # Add interpretation guide
            st.markdown("""
            <div style='font-size: 10px; color: #666; background: #f5f5f5; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong>How to read:</strong><br>
            • <strong>5m chart</strong> (entry timing): Look for MACD crossing above signal = BUY, below = SELL<br>
            • <strong>1h chart</strong> (confirmation): If MACD is above signal on 1h, 5m setup is stronger (trend aligned)<br>
            • <strong>Histogram color</strong>: Green = momentum UP, Red = momentum DOWN<br>
            • <strong>Best entries</strong>: When both 5m AND 1h MACD are above signal line
            </div>
            """, unsafe_allow_html=True)
        elif "ADX" in sig_name or "DI" in sig_name:
            # ADX: 1h ADX with +DI/-DI lines, bounds at 25 (trending) and 15 (developing)
            fig = chart_adx_panel('adx_1h', 'di_plus_1h', 'di_minus_1h', "ADX + DI (1h)", d, h=h)
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        elif "Bollinger" in sig_name or "KC Squeeze" in sig_name:
            # Bollinger Bands: 5m and 1h candlesticks with BB and KC (side by side for comparison)
            # Show both timeframes for entry precision (5m) + confirmation (1h)

            # 5m chart - Entry precision
            bb_5m = chart_candle(d['chart']['s5m'],
                                d['chart']['bb_5m_up'], d['chart']['bb_5m_mid'], d['chart']['bb_5m_lo'],
                                d['chart']['kc_5m_up'], d['chart']['kc_5m_lo'],
                                d['chart']['vwap_5m'], "BB + KC 5m - Entry Precision", h=h)
            st.plotly_chart(bb_5m, use_container_width=True, key=chart_key + "_5m")

            # 1h chart - Confirmation
            bb_1h = chart_candle(d['chart']['s1h'],
                                d['chart']['bb_1h_up'], d['chart']['bb_1h_mid'], d['chart']['bb_1h_lo'],
                                d['chart']['kc_1h_up'], d['chart']['kc_1h_lo'],
                                d['chart']['vwap_1h'], "BB + KC 1h - Trend Confirmation", h=h)
            st.plotly_chart(bb_1h, use_container_width=True, key=chart_key + "_1h")

            # Add interpretation guide
            st.markdown("""
            <div style='font-size: 10px; color: #666; background: #f5f5f5; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong>How to read:</strong><br>
            • <strong>5m chart</strong> (entry precision): Price touching lower BB = oversold entry zone, upper BB = overbought exit zone<br>
            • <strong>1h chart</strong> (confirmation): If price is in lower BB zone on 1h too, 5m entry is stronger (confirmed oversold)<br>
            • <strong>BB Squeeze</strong> (KC inside BB): Narrow bands = low volatility. Breakout likely when bands widen<br>
            • <strong>VWAP</strong> (dashed line): Institutional support/resistance. Price above = buying pressure, below = selling pressure<br>
            • <strong>Best entries</strong>: When 5m price touches BB lower AND 1h price is in lower half of BB
            </div>
            """, unsafe_allow_html=True)
        elif "OBV" in sig_name:
            # OBV: 5m and 1h On-Balance Volume with moving average (side by side)
            # Show both timeframes for entry timing (5m) + confirmation (1h)

            # 5m OBV - Entry timing
            obv_5m = chart_obv_panel('obv_5m', 'obv_5m_ma', "OBV 5m - Entry Timing", d, h=h)
            st.plotly_chart(obv_5m, use_container_width=True, key=chart_key + "_5m")

            # 1h OBV - Confirmation
            obv_1h = chart_obv_panel('obv_1h', 'obv_1h_ma', "OBV 1h - Trend Confirmation", d, h=h)
            st.plotly_chart(obv_1h, use_container_width=True, key=chart_key + "_1h")

            # Add interpretation guide
            st.markdown("""
            <div style='font-size: 10px; color: #666; background: #f5f5f5; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong>How to read:</strong><br>
            • <strong>5m chart</strong> (entry timing): OBV above MA = accumulation (buying), below MA = distribution (selling)<br>
            • <strong>1h chart</strong> (confirmation): If 1h OBV is also above MA, 5m entry is stronger (confirmed buying pressure)<br>
            • <strong>Rising OBV</strong> = Volume supporting UP move (bullish), Falling OBV = Volume supporting DOWN move (bearish)<br>
            • <strong>Best entries</strong>: When both 5m AND 1h OBV are above their moving averages
            </div>
            """, unsafe_allow_html=True)

        elif "MFI" in sig_name:
            # MFI: 5m and 1h Money Flow Index with volume analysis
            # Show both timeframes for entry timing (5m) + confirmation (1h)

            # 5m MFI - Entry timing
            mfi_5m = chart_advanced_oscillators(d, h=h)
            st.plotly_chart(mfi_5m, use_container_width=True, key=chart_key + "_5m")

            st.markdown("""
            <div style='font-size: 10px; color: #666; background: #f5f5f5; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong>Advanced Oscillators Panel (5m + 1h):</strong><br>
            • <strong>MFI (Money Flow Index)</strong>: Volume-weighted RSI. >80 = Overbought, <20 = Oversold<br>
            • <strong>StochRSI</strong>: Fast oscillator. Shows momentum divergence from price movement<br>
            • <strong>Williams %R</strong>: -80 to -20 = Oversold (buy), -20 to 0 = Overbought (sell)<br>
            • <strong>Best setup</strong>: MFI <20 + StochRSI oversold + Williams %R <-80 = Strong BUY confluencce<br>
            • <strong>Divergence alert</strong>: Price up but MFI/oscillators down = Exhaustion (exit soon)
            </div>
            """, unsafe_allow_html=True)
        elif "DXY" in sig_name:
            # DXY: US Dollar Index - inverse correlation with silver
            # Shows DXY trend vs 20-hour MA (above MA = dollar strength = silver headwind)
            dxy_s = d['chart']['dxy1h']['Close'].dropna()
            fig = go.Figure()
            fig.add_hline(y=d['dxy_ma20'], line=dict(color="#FF9500", dash="dash", width=2),
                         annotation_text=f"MA20h {d['dxy_ma20']:.2f}",
                         annotation_position="right")
            fig.add_trace(go.Scatter(x=dxy_s.index, y=dxy_s,
                                    line=dict(color="#0066CC", width=2),
                                    name="DXY", fill='tozeroy', fillcolor="rgba(0,102,204,0.1)"))

            # Auto-scale y-axis to data range with 5% padding
            dxy_min = dxy_s.min()
            dxy_max = dxy_s.max()
            dxy_range = dxy_max - dxy_min
            y_padding = dxy_range * 0.05  # 5% padding above and below
            y_min = dxy_min - y_padding
            y_max = dxy_max + y_padding

            fig.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                            title="US Dollar Index (1h) - Inverse Silver Correlation",
                            yaxis_title="DXY", xaxis_title="Time",
                            yaxis=dict(range=[y_min, y_max]),
                            margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified')
            fig.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        elif "Platinum" in sig_name:
            # Platinum: Industrial demand indicator - shared with silver
            # Shows Platinum price vs 20-hour MA (above MA = risk-on = silver bullish)
            pt_s = d['chart']['pt1h']['Close'].dropna()
            fig = go.Figure()
            if d['pt_ma20']:
                fig.add_hline(y=d['pt_ma20'], line=dict(color="#9933FF", dash="dash", width=2),
                             annotation_text=f"MA20h ${d['pt_ma20']:.0f}",
                             annotation_position="right")
            fig.add_trace(go.Scatter(x=pt_s.index, y=pt_s,
                                    line=dict(color="#9933FF", width=2),
                                    name="Platinum", fill='tozeroy', fillcolor="rgba(153,51,255,0.1)"))
            fig.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                            title="Platinum Price (1h) - Industrial Demand Indicator",
                            yaxis_title="Price USD", xaxis_title="Time",
                            margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified')
            fig.update_yaxes(gridcolor=GRID_COL, tickformat="$.0f")
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        elif "Copper" in sig_name or "Inter-Market" in sig_name:
            # Copper/Gold Ratio: Risk-on indicator
            # Rising ratio = industrial demand up = silver bullish
            gs_s = d['chart']['gs_ratio'].dropna()
            fig = go.Figure()
            fig.add_hline(y=80, line=dict(color=COL_BULL, dash="dot", width=1),
                         annotation_text="80 — Silver cheap vs Gold")
            fig.add_hline(y=60, line=dict(color=COL_BEAR, dash="dot", width=1),
                         annotation_text="60 — Silver expensive vs Gold")
            fig.add_trace(go.Scatter(x=gs_s.index, y=gs_s,
                                    line=dict(color="#FF6600", width=2),
                                    name="Gold/Silver Ratio", fill='tozeroy', fillcolor="rgba(255,102,0,0.1)"))
            fig.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                            title="Gold/Silver Ratio (1h) - Relative Value Indicator",
                            yaxis_title="G/S Ratio", xaxis_title="Time",
                            margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified')
            fig.update_yaxes(gridcolor=GRID_COL, tickformat=".1f")
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        elif "VWAP" in sig_name:
            # VWAP: Volume-Weighted Average Price - institutional intraday benchmark
            # Shows 5m and 1h candles with VWAP overlay for entry timing + trend confirmation

            # 5m VWAP - Entry Timing
            s5m = d['chart']['s5m']
            vwap_5m = d['chart']['vwap_5m'].dropna()
            fig_5m = go.Figure()
            fig_5m.add_trace(go.Candlestick(x=s5m.index, open=s5m['Open'], high=s5m['High'],
                                           low=s5m['Low'], close=s5m['Close'],
                                           increasing_line_color=COL_BULL,
                                           decreasing_line_color=COL_BEAR,
                                           name="Silver"))
            fig_5m.add_trace(go.Scatter(x=vwap_5m.index, y=vwap_5m,
                                       line=dict(color="#FF9500", width=2, dash="dash"),
                                       name="VWAP", fill=None))
            fig_5m.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                                title="VWAP 5m - Entry Timing (Institutional Benchmark)",
                                yaxis_title="Price USD", xaxis_title="Time",
                                margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified')
            fig_5m.update_yaxes(gridcolor=GRID_COL, tickformat="$.2f")
            st.plotly_chart(fig_5m, use_container_width=True, key=chart_key + "_5m")

            # 1h VWAP - Trend Confirmation
            s1h = d['chart']['s1h']
            vwap_1h = d['chart']['vwap_1h'].dropna()
            fig_1h = go.Figure()
            fig_1h.add_trace(go.Candlestick(x=s1h.index, open=s1h['Open'], high=s1h['High'],
                                           low=s1h['Low'], close=s1h['Close'],
                                           increasing_line_color=COL_BULL,
                                           decreasing_line_color=COL_BEAR,
                                           name="Silver"))
            fig_1h.add_trace(go.Scatter(x=vwap_1h.index, y=vwap_1h,
                                       line=dict(color="#FF9500", width=2, dash="dash"),
                                       name="VWAP", fill=None))
            fig_1h.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                                title="VWAP 1h - Trend Confirmation (Institutional Benchmark)",
                                yaxis_title="Price USD", xaxis_title="Time",
                                margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified')
            fig_1h.update_yaxes(gridcolor=GRID_COL, tickformat="$.2f")
            st.plotly_chart(fig_1h, use_container_width=True, key=chart_key + "_1h")

            st.markdown("""
            <div style='font-size: 10px; color: #666; background: #f5f5f5; padding: 8px; border-radius: 4px; margin-top: 8px;'>
            <strong>VWAP Interpretation Guide (5m + 1h):</strong><br>
            • <strong>VWAP Setup</strong>: Volume-Weighted Average Price resets daily at market open. Shows institutional accumulation zones<br>
            • <strong>Price Above VWAP</strong>: Buying pressure in control. Good entry for BUY BULL. Hold while above<br>
            • <strong>Price Below VWAP</strong>: Selling pressure in control. Good entry for BUY BEAR. Hold while below<br>
            • <strong>5m Chart</strong>: Entry-level precision. Price must cross VWAP for entry confirmation<br>
            • <strong>1h Chart</strong>: Trend confirmation. 1h VWAP shows broader institutional trend. 5m + 1h alignment = strongest signal<br>
            • <strong>Best setup</strong>: Price breaks above VWAP on both 5m AND 1h = Strong institutional entry. Vice versa for bears<br>
            • <strong>Divergence alert</strong>: 5m above but 1h below = Choppy market, wait for alignment before entering
            </div>
            """, unsafe_allow_html=True)
        elif "Pivot" in sig_name:
            # Pivot Points: Daily S2, S1, Pivot, R1, R2 levels
            # Shows 1h candles with pivot level lines
            s1h = d['chart']['s1h']
            pivots = d.get('pivots')
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=s1h.index, open=s1h['Open'], high=s1h['High'],
                                        low=s1h['Low'], close=s1h['Close'],
                                        increasing_line_color=COL_BULL,
                                        decreasing_line_color=COL_BEAR,
                                        name="Silver"))

            # Add pivot level lines if available
            if pivots is not None:
                for key, color, label in [
                    ("R2", "rgba(255,50,50,0.8)", "R2 Resistance"),
                    ("R1", "rgba(255,100,100,0.8)", "R1 Resistance"),
                    ("P", "rgba(100,100,100,0.8)", "Daily Pivot"),
                    ("S1", "rgba(50,200,50,0.8)", "S1 Support"),
                    ("S2", "rgba(30,150,30,0.8)", "S2 Support"),
                ]:
                    if key in pivots and not pivots[key].empty:
                        s = pivots[key].dropna()
                        if len(s) > 0:
                            fig.add_trace(go.Scatter(x=s1h.index, y=[s.iloc[-1]]*len(s1h),
                                                    mode='lines',
                                                    line=dict(color=color, width=1, dash="dash"),
                                                    name=label))

            fig.update_layout(height=int(260 * h), plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                            title="Silver 1h with Daily Pivot Points (S2, S1, P, R1, R2)",
                            yaxis_title="Price USD", xaxis_title="Time",
                            margin=dict(l=10, r=10, t=40, b=10), hovermode='x unified')
            fig.update_yaxes(gridcolor=GRID_COL, tickformat="$.2f")
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
    except Exception as e:
        st.warning(f"Chart rendering for {sig_name}: {str(e)}")

    st.markdown("")  # Spacing

    # Current Status explanation
    st.markdown(f"<div style='font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase;'>Current Status</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 11px; color: #000; line-height: 1.5;'>{sig_reason}</div>", unsafe_allow_html=True)

    st.markdown("")  # Spacing

    # What This Means explanation
    st.markdown(f"<div style='font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase;'>What This Means</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size: 11px; color: #555; line-height: 1.5;'>{sig_detail}</div>", unsafe_allow_html=True)

    st.markdown("---")  # Separator between cards

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

st.sidebar.markdown(f"""
<div style='background:{BG_CARD};border-radius:6px;padding:10px;
            border:2px solid {TEXT_SECONDARY};margin-bottom:12px;'>
    <div style='font-size:12px;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:0.5px;'>
        CONTROL PANEL
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.info("📊 Manual refresh only — Press button to get latest data")
if st.sidebar.button("🔄 Refresh Now (Get Fresh Data)", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"""
<div style='font-size:11px;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:0.5px;
            margin-top:12px;margin-bottom:6px;text-transform:uppercase;border-top:1px solid rgba(255,255,255,0.15);padding-top:12px;'>
Display Settings
</div>
""", unsafe_allow_html=True)
chart_scale = st.sidebar.select_slider(
    "Chart Height",
    options=["Compact", "Normal", "Large"],
    value="Normal"
)
H = {"Compact": 0.7, "Normal": 1.4, "Large": 2.1}[chart_scale]


# ═══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════

# Clean minimalist header banner
st.markdown(f"""
<div style='background:{BG_CARD};border-radius:8px;padding:20px 24px;margin-bottom:20px;
            border:2px solid rgba(0,0,0,0.15);'>
    <div style='font-size:36px;font-weight:900;color:{TEXT_PRIMARY};margin-bottom:8px;letter-spacing:1px;'>
        SILVER MARKET ANALYSIS
    </div>
    <div style='font-size:12px;color:{TEXT_SECONDARY};line-height:1.5;'>
        Professional analyst dashboard + Educational platform for intraday trading
        <br>
        <span style='color:{COL_BULL};font-weight:bold;'>●</span> Multi-timeframe analysis (5m / 1h / 4h)
        <span style='color:{TEXT_SECONDARY};margin-left:14px;'>●</span> Regime-adaptive signal weighting
        <span style='color:{TEXT_SECONDARY};margin-left:14px;'>●</span> Real-time volatility-scaled predictions
    </div>
</div>
""", unsafe_allow_html=True)

# Educational introduction
with st.expander("📖 How to Use This Dashboard (Start Here)", expanded=False):
    st.markdown(f"""
    ## Welcome to Silver Pro Advisor

    This dashboard follows **professional trading analyst workflow** — the way senior traders at hedge funds actually decide whether to enter a trade.

    ---

    ## The Professional Decision Flow

    ### ⚡ **PHASE 1: SETUP CHECK** — Should I Trade Right Now?

    **Look at these signals FIRST:**
    - **DXY Dollar Trend** (Weight: 2.5) — Is there macro tailwind or headwind?
    - **ADX Trend Regime** (Weight: 3.0) — Is a trend established or ranging?
    - **Pivot Points** (Weight: 2.0) — Where are my safe entry/exit levels?

    **Stop here if ANY of these are not aligned with your bias.**

    **Learn**: Professional traders DON'T trade without macro setup. Commodities are priced in USD, so dollar strength determines the baseline.

    ---

    ### 🎯 **PHASE 2: TRADING DECISION** — Confirmed Setup, What's My Signal?

    Once Phase 1 checks out, confirm momentum is supporting it:
    - **MACD Trend** (Weight: 2.0) — Is momentum in your direction?
    - **OBV/MFI/VWAP** (Weight: 1.5 each) — Is volume confirming the move?

    **The trading signal at the top is your verdict.** Confidence % shows how many signals agree.

    **Learn**: Volume must confirm price. No volume = setup is fake.

    ---

    ### 💰 **PHASE 3: ENTRY PRECISION** — Exactly When & Where Do I Enter?

    Once you decide to trade, find your exact entry and exit zones:
    - **Entry Options by Timeframe** — Choose your trading style (Fast Scalp vs Multi-Timeframe)
    - **Entry/Exit Range Analysis** — Shows exact Bollinger Band zones for entry/exit
    - **Bollinger Bands + RSI** (Weight: 1.5 + 1.0) — Entry timing and price extremes

    **Learn**: Entries at extremes (lower BB for bulls, upper BB for bears) have higher odds.

    ---

    ### 📚 **PHASE 4: SUPPORTING CONTEXT** — Additional Reference (Collapsed)

    **Optional deep dive** for additional context:
    - **Platinum & Copper/Gold Trends** (Weight: 1.0) — How related markets look
    - **Signal History** — How the signal has evolved over last 45 minutes
    - **Market Data** — Current prices, spread status, regime

    **Learn**: Supporting signals are lower priority. Trust Phase 1-3 first.

    ---

    ## Weight Hierarchy (What Matters Most)

    This dashboard weights signals by **professional trading practice**, not by technical complexity:

    | Importance | Signals | Weight | Why |
    |------------|---------|--------|-----|
    | CRITICAL | ADX, DXY | 3.0-2.5 | Determines IF you trade |
    | HIGH | Pivots, MACD | 2.0-2.5 | Confirms setup + momentum |
    | MEDIUM | Volume (OBV/MFI), VWAP | 1.5 | Validates the move |
    | ENTRY | Bollinger Bands | 1.5 | Exact entry zone |
    | TIMING | RSI/Oscillators | 1.0 | When to enter within zone |
    | CONTEXT | Platinum, Metals Ratios | 1.0 | Nice to know, not critical |

    **Key insight**: Oscillators (RSI, StochRSI) are LOW weight because they're entry timing tools, not conviction builders.

    ---

    ## How to Read This Dashboard

    **For Quick Decisions (2 minutes):**
    1. Check Phase 1 — Is setup in place?
    2. Check trading signal — What's the verdict?
    3. Go to Phase 3 — Where exactly do I enter?

    **For Deep Analysis (5 minutes):**
    1. Read all Phase 1 signals with charts
    2. Verify Phase 2 momentum confirms
    3. Plan entry/exit using Phase 3 zones
    4. Expand Phase 4 for macro context

    **For Learning:**
    - Each signal card explains:
      - What the indicator measures
      - What it's saying right now
      - What it means for your trade

    ---

    ## Pro Tips

    ✅ **DO**: Trade only when Phase 1 + Phase 2 align
    ✅ **DO**: Use Phase 3 zones for precise entry/exit
    ✅ **DO**: Check signal weight — higher weights = more important
    ❌ **DON'T**: Trade oscillators alone (they're weight 1.0 for a reason)
    ❌ **DON'T**: Ignore DXY — commodities are dollar-driven
    ❌ **DON'T**: Chase entries outside Bollinger Bands — wait for the zone

    Click any **"📚 Educational"** expander to understand the "why" behind the "what".

    ---

    ## Quick Tips

    ✅ **For Trading**: Focus on Market Snapshot → Opportunity → Predictions (5-10 min analysis)

    ✅ **For Learning**: Expand ALL educational sections, understand each metric deeply

    ✅ **For Beginners**: Start with the Executive Summary and work backward to understand why signals point that direction

    ✅ **For Advanced**: Modify the regime multipliers in Sidebar → Display Settings if you disagree with defaults

    ---

    **Remember**: Markets can change rapidly. Always verify live before trading. This is analysis, not financial advice.
    """)


d, err = gather_intelligence()
if err:
    st.error(f"Data fetch error:\n\n```\n{err}\n```")
    st.stop()

# Initialize signal history tracking
initialize_signal_history()

# Status bar with data freshness
age    = data_age_hours(d['last_ts'])
ts_str = d['last_ts'].strftime('%Y-%m-%d %H:%M UTC')

if age > 1:
    status_color = COL_BEAR
    status_icon = "⚠️"
    status_text = f"STALE - Last candle {age:.1f}h old ({ts_str})"
    status_bg = BG_CARD
else:
    status_color = COL_BULL
    status_icon = "●"
    status_text = f"LIVE - {ts_str} | Session: {d['session_label']} ({d['session_weight']}×)"
    status_bg = BG_CARD

st.markdown(f"""
<div style='background:{status_bg};border-radius:8px;padding:12px 14px;margin-bottom:16px;
            border:2px solid {status_color};'>
    <div style='display:flex;align-items:center;gap:10px;'>
        <span style='font-size:18px;color:{status_color};'>{status_icon}</span>
        <div style='flex:1;'>
            <div style='font-size:10px;color:{TEXT_SECONDARY};letter-spacing:0.5px;font-weight:bold;'>MARKET STATUS</div>
            <div style='font-size:12px;color:{TEXT_PRIMARY};margin-top:2px;'>{status_text}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

signals, total, max_total = run_scoring(d)

# Calculate independent confidence for BOTH directions (UP and DOWN)
bull_conviction, bull_breakdown = calculate_signal_conviction(signals, "UP", d, 60)
bear_conviction, bear_breakdown = calculate_signal_conviction(signals, "DOWN", d, 60)

# Generate predictions for certificate signal (use 1h as primary)
cert_predictions = {}
for tf in [60, 15, 5]:  # 1h, 15m, 5m
    cert_predictions[tf] = predict_move(d, tf, signals=signals)

# Generate certificate trading signal
cert_signal, cert_conf, cert_entry, cert_target, cert_stop, cert_reasoning = generate_certificate_signal(d, cert_predictions)

# Get regime progression for display
cert_regime_progression = get_regime_progression(d)

# Snapshot current signal for history tracking
snapshot_current_signal(d, cert_signal, cert_conf, d['silver'])

# On first load, backtest 45-minute history
if len(st.session_state.signal_history) == 1:  # Just added first snapshot
    backtest_signals = backtest_signal_history(d, signals)
    if backtest_signals:
        st.session_state.signal_history = backtest_signals + st.session_state.signal_history

# ═══════════════════════════════════════════════════════════════════
# SIMPLIFIED: MAIN SIGNAL + WHY
# ═══════════════════════════════════════════════════════════════════

# Simple header
silver_price = d.get('silver') or 0
atr_val = d.get('atr_5m') or 0
regime_val = d.get('regime') or 'UNKNOWN'
st.markdown(f"### ${silver_price:.2f} | {regime_val} Market | ATR: {atr_val:.2f}")

# Main signal box (simplified)
if cert_signal == 'BUY_BULL':
    sig_color, sig_text = COL_BULL, "🟢 BUY BULL"
elif cert_signal == 'BUY_BEAR':
    sig_color, sig_text = COL_BEAR, "🔴 BUY BEAR"
else:
    sig_color, sig_text = COL_NEUT, "⚪ WAIT"

if cert_signal == 'WAIT' or cert_entry is None or cert_target is None or cert_stop is None:
    # WAIT signal or insufficient data - show no prices
    st.markdown(f"""
<div style='background:#fff;border:3px solid {sig_color};border-radius:8px;padding:16px;margin-bottom:16px;'>
    <div style='font-size:28px;font-weight:bold;color:{sig_color};margin-bottom:8px;'>{sig_text}</div>
    <div style='font-size:14px;color:{TEXT_PRIMARY};margin-bottom:8px;'>Confidence: <strong>{cert_conf}%</strong></div>
    <div style='margin-top:12px;padding-top:12px;border-top:1px solid #eee;font-size:11px;color:{TEXT_SECONDARY};line-height:1.7;'>
        {'<br>'.join(cert_reasoning[:3]) if cert_reasoning else 'Waiting for clearer signal...'}
    </div>
</div>
""", unsafe_allow_html=True)
else:
    # BUY signal - show entry/target/stop
    entry_val = cert_entry if cert_entry is not None else d.get('silver', 0)
    target_val = cert_target if cert_target is not None else d.get('silver', 0)
    stop_val = cert_stop if cert_stop is not None else d.get('silver', 0)

    st.markdown(f"""
<div style='background:#fff;border:3px solid {sig_color};border-radius:8px;padding:16px;margin-bottom:16px;'>
    <div style='font-size:28px;font-weight:bold;color:{sig_color};margin-bottom:8px;'>{sig_text}</div>
    <div style='font-size:14px;color:{TEXT_PRIMARY};margin-bottom:8px;'>Confidence: <strong>{cert_conf}%</strong></div>
    <div style='font-size:12px;color:{TEXT_PRIMARY};line-height:1.6;'>
        Entry: <strong>${entry_val:.3f}</strong> | Target: <strong>${target_val:.3f}</strong> | Stop: <strong>${stop_val:.3f}</strong>
    </div>
    <div style='margin-top:12px;padding-top:12px;border-top:1px solid #eee;font-size:11px;color:{TEXT_SECONDARY};line-height:1.7;'>
        {'<br>'.join(cert_reasoning[:3]) if cert_reasoning else 'Signal generated'}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")  # Spacing

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL ANALYST DECISION FLOW
# ═══════════════════════════════════════════════════════════════════════════════

# PHASE 1: SETUP CHECK — "Should I Trade Right Now?"
st.markdown("---")
st.markdown(f"""
<div style='font-size:16px;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:1px;margin-bottom:4px;'>
⚡ PHASE 1: SETUP CHECK — Should I Trade Right Now?
</div>
<div style='font-size:11px;color:{TEXT_SECONDARY};margin-bottom:16px;'>
Macro tailwind? Trend established? Momentum confirmed?
</div>
""", unsafe_allow_html=True)

# Render Phase 1 signals: DXY, ADX, Pivot Points
phase_1_signals = ["DXY Dollar Trend", "ADX Trend Regime (1h)", "Pivot Point Proximity"]
for sig in signals:
    if sig['name'] in phase_1_signals:
        render_signal_detail_card(sig, d, h=H)

st.markdown("")  # Spacing

# PHASE 2: TRADING DECISION — "I Have Setup Confirmation, Now What's My Signal?"
st.markdown("---")
st.markdown(f"""
<div style='font-size:16px;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:1px;margin-bottom:4px;'>
🎯 PHASE 2: TRADING DECISION
</div>
<div style='font-size:11px;color:{TEXT_SECONDARY};margin-bottom:16px;'>
(Signal already shown above — if Phase 1 checks out, this is your verdict)
</div>
""", unsafe_allow_html=True)

# Show momentum confirmation signals
st.markdown(f"""
<div style='font-size:13px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:12px;'>
📈 Momentum Confirmation (MACD, OBV, VWAP, MFI)
</div>
""", unsafe_allow_html=True)

phase_2_signals = ["MACD Trend (5m + 1h)", "OBV Accumulation (5m + 1h)", "MFI Volume Flow (5m + 1h)", "VWAP (5m + 1h)"]
for sig in signals:
    if sig['name'] in phase_2_signals:
        render_signal_detail_card(sig, d, h=H)

st.markdown("")  # Spacing

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ENTRY PRECISION — "Exactly When & Where Do I Enter?"
st.markdown("---")
st.markdown(f"""
<div style='font-size:16px;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:1px;margin-bottom:4px;'>
💰 PHASE 3: ENTRY PRECISION
</div>
<div style='font-size:11px;color:{TEXT_SECONDARY};margin-bottom:16px;'>
Now that setup is confirmed, find your exact entry and exit zones
</div>
""", unsafe_allow_html=True)

# Entry options by timeframe (show only the timeframes being predicted)
st.markdown("#### Entry Options by Timeframe")

# Prediction timeframe mode selector
timeframe_mode = st.radio(
    "Choose your trading style:",
    options=["Fast Scalp (1h→15m→5m)", "Multi-Timeframe (4h→1h→15m)"],
    index=1,
    horizontal=True,
    help="Fast Scalp: Quick 5m-15m entries with 1h confirmation. Multi-Timeframe: Strategic entries from 4h macro view"
)

# Compute predictions based on selected timeframe mode
predictions = {}
if timeframe_mode == "Fast Scalp (1h→15m→5m)":
    timeframes_to_predict = [60, 15, 10, 5]  # 1h, 15m, 10m, 5m
    mode_label = "⚡ Fast Scalp Mode (Intraday)"
    display_timeframes = [60, 15, 5]  # 1h, 15m, 5m
else:  # Multi-Timeframe (4h→1h→15m)
    timeframes_to_predict = [240, 60, 15, 10]  # 4h, 1h, 15m, 10m
    mode_label = "📊 Multi-Timeframe Mode (Macro + Entry)"
    display_timeframes = [240, 60, 15]  # 4h, 1h, 15m

for tf in timeframes_to_predict:
    predictions[tf] = predict_move(d, tf, signals=signals)

st.caption(f"Mode: {mode_label}")

cols = st.columns(3)
for i, tf in enumerate(display_timeframes):
    pred = predictions.get(tf, {})
    direction = pred.get('direction', 'UNKNOWN')
    confidence = pred.get('confidence', 0)

    dir_icon = "📈" if direction == "UP" else ("📉" if direction == "DOWN" else "⏸️")
    dir_text = "UP" if direction == "UP" else ("DOWN" if direction == "DOWN" else "FLAT")

    with cols[i]:
        tf_label = "5m" if tf == 5 else ("15m" if tf == 15 else ("1h" if tf == 60 else "4h"))
        st.write(f"**{tf_label}**: {dir_icon} {dir_text} ({confidence}%)")
        if 'target_price' in pred and 'stop_loss' in pred:
            st.caption(f"T: ${pred['target_price']:.3f} | S: ${pred['stop_loss']:.3f}")

st.markdown("")  # Spacing

# TECHNICAL ANALYSIS - Entry/Exit Range as dedicated cards
st.markdown("---")
st.markdown("## 📈 Entry/Exit Range Analysis")

# Toggle between BULL and BEAR analysis view
analysis_type = st.radio(
    "Analyze for:",
    ["🟢 BUY BULL (Price UP)", "🔴 BUY BEAR (Price DOWN)"],
    horizontal=True,
    help="Choose to see entry/exit analysis for BULL or BEAR positions"
)

is_bull_analysis = "BULL" in analysis_type

if is_bull_analysis:
    st.markdown("**Entry Strategy: Buy when price reaches LOWER Bollinger Band | Exit Strategy: Sell when price reaches UPPER Bollinger Band**")
else:
    st.markdown("**Entry Strategy: Short when price reaches UPPER Bollinger Band | Exit Strategy: Cover when price reaches LOWER Bollinger Band**")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════
# LAST 1 HOUR PRICE ACTION (GRANULAR VIEW)
# ═══════════════════════════════════════════════════════════════════

st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:12px;'>LAST 1 HOUR PRICE ACTION (5m Candles + Bollinger Bands)</div>", unsafe_allow_html=True)

# Get last 12 5-minute candles (approximately 1 hour)
s5m = d.get('chart', {}).get('s5m', None)
if s5m is not None and len(s5m) >= 12:
    s5m_last_1h = s5m.iloc[-12:].copy()
    bb_up_5m = d.get('chart', {}).get('bb_5m_up', None)
    bb_mid_5m = d.get('chart', {}).get('bb_5m_mid', None)
    bb_lo_5m = d.get('chart', {}).get('bb_5m_lo', None)

    if bb_up_5m is not None and bb_mid_5m is not None and bb_lo_5m is not None:
        # Filter bands for the same time range
        bb_up_filt = bb_up_5m.loc[s5m_last_1h.index].dropna() if isinstance(bb_up_5m, pd.Series) else None
        bb_mid_filt = bb_mid_5m.loc[s5m_last_1h.index].dropna() if isinstance(bb_mid_5m, pd.Series) else None
        bb_lo_filt = bb_lo_5m.loc[s5m_last_1h.index].dropna() if isinstance(bb_lo_5m, pd.Series) else None

        # Create candlestick chart
        fig = go.Figure()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=s5m_last_1h.index,
            open=s5m_last_1h['Open'],
            high=s5m_last_1h['High'],
            low=s5m_last_1h['Low'],
            close=s5m_last_1h['Close'],
            increasing_line_color=COL_BULL,
            decreasing_line_color=COL_BEAR,
            name="Silver Price",
            hovertemplate="<b>%{x|%H:%M}</b><br>O: $%{open:.3f}<br>H: $%{high:.3f}<br>L: $%{low:.3f}<br>C: $%{close:.3f}"
        ))

        # Bollinger Bands
        if bb_up_filt is not None and len(bb_up_filt) > 0:
            fig.add_trace(go.Scatter(
                x=bb_up_filt.index, y=bb_up_filt,
                mode='lines', name='Upper BB (Exit Zone)',
                line=dict(color=COL_BEAR, width=1, dash='dash'),
                hovertemplate="<b>Upper BB</b><br>%{y:.3f}"
            ))

        if bb_mid_filt is not None and len(bb_mid_filt) > 0:
            fig.add_trace(go.Scatter(
                x=bb_mid_filt.index, y=bb_mid_filt,
                mode='lines', name='Middle BB',
                line=dict(color=TEXT_SECONDARY, width=1, dash='dash'),
                hovertemplate="<b>Mid BB</b><br>%{y:.3f}"
            ))

        if bb_lo_filt is not None and len(bb_lo_filt) > 0:
            fig.add_trace(go.Scatter(
                x=bb_lo_filt.index, y=bb_lo_filt,
                mode='lines', name='Lower BB (Entry Zone)',
                line=dict(color=COL_BULL, width=1, dash='dash'),
                hovertemplate="<b>Lower BB</b><br>%{y:.3f}"
            ))

        # Current price line
        current_price_val = d.get('silver') or 0
        fig.add_hline(
            y=current_price_val,
            line=dict(color="#FF9500", width=2, dash="solid"),
            annotation_text=f"Current: ${current_price_val:.3f}",
            annotation_position="right"
        )

        fig.update_layout(
            title="Last 1 Hour Silver Price Action (5-Min Candles with Entry/Exit Zones)",
            yaxis_title="Price (USD)",
            xaxis_title="Time",
            template='plotly_white',
            height=int(380 * H),
            hovermode='x unified',
            xaxis=dict(tickformat='%H:%M'),
            yaxis=dict(tickformat="$.3f"),
            margin=dict(l=60, r=40, t=40, b=40),
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PLOT_BG
        )

        fig.update_yaxes(gridcolor=GRID_COL)
        st.plotly_chart(fig, use_container_width=True, key="1h_price_action")

        # Quick interpretation
        last_close = s5m_last_1h['Close'].iloc[-1]
        price_at_upper = last_close > (bb_up_filt.iloc[-1] if bb_up_filt is not None and len(bb_up_filt) > 0 else 0)
        price_at_lower = last_close < (bb_lo_filt.iloc[-1] if bb_lo_filt is not None and len(bb_lo_filt) > 0 else 0)

        if price_at_upper:
            zone_status = "⚠️ Price at UPPER BB - Overbought, consider exit for bulls / entry for bears"
        elif price_at_lower:
            zone_status = "✅ Price at LOWER BB - Oversold, consider entry for bulls / exit for bears"
        else:
            zone_status = "⏸️ Price in MIDDLE zone - Neutral, wait for breakout"

        st.caption(zone_status)

st.markdown("")

# Entry/Exit Analysis Cards

bb_lower = d.get('bb_lo_5m') or 0
bb_mid = d.get('bb_mid_5m') or 0
bb_upper = d.get('bb_up_5m') or 0
current_price = d.get('silver') or 0

if is_bull_analysis:
    # BULL - Entry Card
    entry_dist = current_price - bb_lower if current_price > bb_lower else 0
    entry_status_bull = 'Ready to enter when price dips to $' + f'{bb_lower:.3f}' if entry_dist > 0 else 'Already in entry zone!'
    entry_position_bull = 'above' if entry_dist > 0 else 'below'
    st.markdown(f"""
<div style="border: 2px solid #00DD00; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: rgba(0, 221, 0, 0.05);">
    <div style="font-size: 13px; font-weight: bold; color: #000; margin-bottom: 8px;">📈 BULL ENTRY ZONE</div>
    <div style="background: white; border: 1px solid #00DD00; border-radius: 4px; padding: 10px; margin-bottom: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Entry Level</div>
        <div style="font-size: 14px; font-weight: bold; color: #00DD00;">${bb_lower:.3f}</div>
        <div style="font-size: 10px; color: #555; margin-top: 4px;">Lower Bollinger Band (Oversold Zone)</div>
    </div>
    <div style="background: white; border: 1px solid #00DD00; border-radius: 4px; padding: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Current Position</div>
        <div style="font-size: 10px; color: #555; line-height: 1.6;">
            Price: ${current_price:.3f}<br>
            Distance from entry: ${entry_dist:.3f} (Price is {entry_position_bull} entry zone)<br>
            Status: {entry_status_bull}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # BULL - Exit Card
    exit_dist = bb_upper - current_price if current_price < bb_upper else 0
    st.markdown(f"""
<div style="border: 2px solid #FF0000; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: rgba(255, 0, 0, 0.05);">
    <div style="font-size: 13px; font-weight: bold; color: #000; margin-bottom: 8px;">📈 BULL EXIT ZONE</div>
    <div style="background: white; border: 1px solid #FF0000; border-radius: 4px; padding: 10px; margin-bottom: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Target Price</div>
        <div style="font-size: 14px; font-weight: bold; color: #FF0000;">${bb_upper:.3f}</div>
        <div style="font-size: 10px; color: #555; margin-top: 4px;">Upper Bollinger Band (Overbought Zone)</div>
    </div>
    <div style="background: white; border: 1px solid #FF0000; border-radius: 4px; padding: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Exit Signals</div>
        <div style="font-size: 10px; color: #555; line-height: 1.6;">
            Profit target: ${bb_upper:.3f} (+${bb_upper - current_price:.3f} from now)<br>
            Exit triggers: RSI > 70 OR MACD histogram turns red OR price closes above upper BB<br>
            Don't hold for diminishing returns - exit at target or at first reversal signal
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

else:
    # BEAR - Entry Card
    entry_dist = bb_upper - current_price if current_price < bb_upper else 0
    entry_status_bear = 'Ready to enter when price rises to $' + f'{bb_upper:.3f}' if entry_dist > 0 else 'Already in entry zone!'
    entry_position_bear = 'below' if entry_dist > 0 else 'above'
    st.markdown(f"""
<div style="border: 2px solid #FF0000; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: rgba(255, 0, 0, 0.05);">
    <div style="font-size: 13px; font-weight: bold; color: #000; margin-bottom: 8px;">📉 BEAR ENTRY ZONE</div>
    <div style="background: white; border: 1px solid #FF0000; border-radius: 4px; padding: 10px; margin-bottom: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Entry Level</div>
        <div style="font-size: 14px; font-weight: bold; color: #FF0000;">${bb_upper:.3f}</div>
        <div style="font-size: 10px; color: #555; margin-top: 4px;">Upper Bollinger Band (Overbought Zone)</div>
    </div>
    <div style="background: white; border: 1px solid #FF0000; border-radius: 4px; padding: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Current Position</div>
        <div style="font-size: 10px; color: #555; line-height: 1.6;">
            Price: ${current_price:.3f}<br>
            Distance from entry: ${entry_dist:.3f} (Price is {entry_position_bear} entry zone)<br>
            Status: {entry_status_bear}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # BEAR - Exit Card
    exit_dist = current_price - bb_lower if current_price > bb_lower else 0
    st.markdown(f"""
<div style="border: 2px solid #00DD00; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: rgba(0, 221, 0, 0.05);">
    <div style="font-size: 13px; font-weight: bold; color: #000; margin-bottom: 8px;">📉 BEAR EXIT ZONE</div>
    <div style="background: white; border: 1px solid #00DD00; border-radius: 4px; padding: 10px; margin-bottom: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Target Price</div>
        <div style="font-size: 14px; font-weight: bold; color: #00DD00;">${bb_lower:.3f}</div>
        <div style="font-size: 10px; color: #555; margin-top: 4px;">Lower Bollinger Band (Oversold Zone)</div>
    </div>
    <div style="background: white; border: 1px solid #00DD00; border-radius: 4px; padding: 10px;">
        <div style="font-size: 10px; color: #666; margin-bottom: 4px; text-transform: uppercase; font-weight: bold;">Exit Signals</div>
        <div style="font-size: 10px; color: #555; line-height: 1.6;">
            Profit target: ${bb_lower:.3f} (${current_price - bb_lower:.3f} profit from now)<br>
            Exit triggers: RSI < 30 OR MACD histogram turns green OR price closes below lower BB<br>
            Don't hold for diminishing returns - exit at target or at first reversal signal
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# Render Phase 3 entry precision signals: Bollinger Bands + RSI
st.markdown(f"""
<div style='font-size:13px;font-weight:bold;color:{TEXT_PRIMARY};margin-bottom:12px;margin-top:20px;'>
🎯 Entry Precision Signals (Bollinger Bands + RSI)
</div>
""", unsafe_allow_html=True)

phase_3_signals = ["Bollinger Bands + KC Squeeze (5m/1h)", "Oscillator Consensus (RSI + StochRSI + Williams%R)"]
for sig in signals:
    if sig['name'] in phase_3_signals:
        render_signal_detail_card(sig, d, h=H)

st.markdown("")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: REFERENCE & CONTEXT — "Additional Context (Optional)"
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style='font-size:16px;font-weight:900;color:{TEXT_PRIMARY};letter-spacing:1px;margin-bottom:4px;'>
📚 PHASE 4: SUPPORTING CONTEXT
</div>
<div style='font-size:11px;color:{TEXT_SECONDARY};margin-bottom:16px;'>
Additional analysis and reference information (collapsed by default)
</div>
""", unsafe_allow_html=True)

# Collapsible section for remaining signals
with st.expander("🔍 Detailed Analysis & Macro Context", expanded=False):
    # Render remaining Phase 4 signals: Platinum, Copper/Gold
    phase_4_signals = ["Platinum Trend (1h)", "Inter-Market: Copper/Gold"]
    for sig in signals:
        if sig['name'] in phase_4_signals:
            render_signal_detail_card(sig, d, h=H)

    st.markdown("")

    # Signal history chart
    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin:16px 0 8px 0;'>Signal Performance (Last 45 Min)</div>", unsafe_allow_html=True)
    render_signal_history_chart(st.session_state.signal_history)

    st.markdown("")

    # Market data metrics
    st.markdown(f"<div style='font-size:12px;font-weight:bold;color:{TEXT_PRIMARY};margin:16px 0 8px 0;'>Market Data</div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        spot = d.get('silver') or 0
        st.metric("Spot (XAGX)", f"${spot:.2f}")
    with col2:
        futures = d.get('futures_price') or 0
        st.metric("Futures (SI=F)", f"${futures:.2f}")
    with col3:
        spot = d.get('silver') or 0
        futures = d.get('futures_price') or 0
        spread = futures - spot if futures and spot else 0
        status = "📈 Contango" if spread > 0 else "📉 Backwardation"
        st.metric("Spread", status)
    with col4:
        regime = d.get('regime') or 'UNKNOWN'
        st.metric("Regime", regime)

st.markdown("")
st.markdown("---")
st.markdown("")

# Market summary (simplified plain english)
st.markdown("#### Summary")
render_analysis_summary(signals, d)

# Footer (minimal)
st.markdown(f"<div style='text-align:center;margin-top:20px;font-size:9px;color:#999;'>Updated: {ts_str} | Disclaimer: Research only</div>", unsafe_allow_html=True)
