# Silver Advisor - Professional Trading Dashboard

A professional-grade Streamlit dashboard for intraday silver certificate trading with real-time technical analysis, signal generation, and decision support.

## Features

### 🎯 Core Trading Signal
- **BUY BULL / BUY BEAR / EXIT** verdict with confidence levels
- Entry, target, and stop-loss prices based on 1-hour predictions
- Real-time signal history tracking (45-minute window)

### 📊 Professional Analyst Decision Flow

**Phase 1: Setup Validation**
- DXY Dollar Trend (macro tailwind/headwind)
- ADX Trend Regime (established vs ranging)
- Pivot Points (daily support/resistance)

**Phase 2: Momentum Confirmation**
- MACD Trend (5m + 1h dual timeframes)
- Volume Flow Analysis (OBV, MFI)
- VWAP Institutional Flow

**Phase 3: Entry Precision**
- Bollinger Bands + Keltner Channels (entry/exit zones)
- RSI Entry Zones (oversold/overbought)
- Last 1-Hour Granular Price Action (5-minute candlesticks)

**Phase 4: Supporting Context**
- StochRSI, Williams%R, Platinum Trend
- Copper/Gold Inter-Market Ratio
- Market Snapshot with session weights

### 📈 Technical Indicators (11 Signals)

| Signal | Weight | Purpose |
|--------|--------|---------|
| ADX Trend Regime | 3.0 | Trend strength classification |
| DXY Dollar Trend | 3.5 | Macro USD strength (inverse silver) |
| Pivot Point Proximity | 2.5 | Daily support/resistance levels |
| MACD Trend | 2.5 | Momentum confirmation |
| OBV Accumulation | 2.0 | Volume flow analysis |
| MFI Volume Flow | 2.0 | Volume-weighted momentum |
| VWAP | 2.0 | Institutional buying/selling |
| Bollinger Bands + KC | 1.5 | Entry/exit zones |
| Oscillator Consensus | 0.8 | RSI, StochRSI, Williams%R |
| Platinum Trend | 1.0 | Precious metals co-movement |
| Copper/Gold Ratio | 1.0 | Industrial demand proxy |

### 🔄 Prediction Modes
- **Fast Scalp (1h→15m→5m)**: Ultra-quick entries for intraday scalping
- **Multi-Timeframe (4h→1h→15m)**: Strategic entries with macro context

### 📚 Educational Features
- "How to Use This Dashboard" guide with interactive examples
- "How to Read the Bottom Line" with per-signal explanations
- Professional analyst decision checklist
- Entry/exit strategy guidance

## Data Sources

- **Silver Spot (XAGX-USD)**: 24/5 market data from yfinance
- **Silver Futures (SI=F)**: CME futures for trend confirmation
- **Macro Data**: DXY (USD Index), Gold, Platinum, Copper
- **Technical**: 5m/1h/4h candlesticks via yfinance

## Installation

```bash
pip install streamlit pandas numpy plotly yfinance scipy
```

## Usage

```bash
streamlit run silver_advisor.py
```

Then open your browser to `http://localhost:8501`

## Configuration

### Display Settings (Sidebar)
- **Chart Height**: Compact (0.7×), Normal (1.4×), Large (2.1×)
- **Prediction Timeframes**: Fast Scalp vs Multi-Timeframe mode
- **Auto-refresh**: Manual refresh only (press button)

### Data Freshness
- Status bar shows: LIVE (< 1 hour old) or STALE (older data)
- Last update timestamp displayed
- Session weight multiplier for current trading window

## Trading Signals Explained

### Signal Conviction Calculation
- Weighted consensus from 11 technical indicators
- Regime-adaptive multipliers (TRENDING favors trend signals, RANGING favors oscillators)
- Session-weighted scoring (London/NY overlap highest, off-hours lowest)
- Timeframe-adjusted for 5m, 15m, 30m, 60m predictions

### Entry/Exit Strategy

**For BUY BULL Certificates:**
- Entry: Price touches LOWER Bollinger Band (oversold)
- Exit: Price reaches UPPER Bollinger Band (overbought)
- Stop Loss: 1 ATR below entry (volatility-scaled)

**For BUY BEAR Certificates:**
- Entry: Price touches UPPER Bollinger Band (overbought)
- Exit: Price reaches LOWER Bollinger Band (oversold)
- Stop Loss: 1 ATR above entry

## Limitations

- yfinance 5m data: 5-day maximum history, gaps possible on weekends
- No COT reports (requires paid data feed)
- Spot price unavailable during US market hours (24/5 market)
- This is analysis only, not financial advice

## Support

For issues or feature requests, please open an issue on GitHub.

---

**Last Updated:** May 2026  
**Python Version:** 3.9+  
**License:** MIT
