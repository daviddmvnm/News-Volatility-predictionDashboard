# 📈 News & Volatility Dashboard

Interactive dashboard exploring how news sentiment predicts extreme market volatility.

**[Live Demo](https://your-app.streamlit.app)** ← Update this after deployment

## Overview

This dashboard implements a **hybrid residual model** that combines:
- **Baseline**: Logistic regression on market features (captures volatility persistence)
- **Residual**: Random forest on news features (captures nonlinear news effects)

The key insight from the underlying research: **news is most informative for extreme volatility events**, not routine fluctuations. The effect varies significantly across sectors.

## Features

- Enter any valid Yahoo Finance ticker
- Adjustable volatility percentile threshold (80th-95th)
- Interactive visualizations:
  - Alpha curve (news weight vs. model performance)
  - Feature importance rankings
  - Time series of predictions vs. actuals
- Cached results for fast repeated queries

## Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/yourusername/news-volatility-dashboard.git
cd news-volatility-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and `app.py`
5. Deploy!

## Data

The dashboard uses two data sources:

1. **Market Data**: Pulled live from Yahoo Finance via `yfinance`
2. **News Features**: Pre-computed daily theme-level sentiment features (`daily_features.csv`)

The news features are derived from ~2 million financial news articles (2016-2024), clustered into 7 themes:
- Markets, Macro, Earnings, Energy, Tech, Trade, Geopolitics

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Ticker + Date Range              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Market Features          │  News Features                  │
│  - Lagged volatility      │  - Sentiment intensity          │
│  - Lagged returns         │  - Primary burst                │
│                           │  - Regime indicators            │
└─────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Logistic Regression│      │   Random Forest     │
│  (Baseline)         │      │   (Residual)        │
└─────────────────────┘      └─────────────────────┘
          │                              │
          │      P = baseline + α × residual
          └──────────────┬───────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Classification: Extreme Volatility?            │
└─────────────────────────────────────────────────────────────┘
```

## Results Summary

From the underlying research (BEE3066 Term Project):

| Sector | Baseline F1 | Hybrid F1 | News Uplift |
|--------|-------------|-----------|-------------|
| Energy (XLE) | 0.488 | 0.576 | +18% |
| Consumer Staples (XLP) | 0.745 | 0.864 | +16% |
| Healthcare (XLV) | 0.698 | 0.776 | +11% |
| Financials (XLF) | 0.738 | 0.800 | +8% |
| Utilities (XLU) | 0.656 | 0.656 | 0% |

**Key finding**: News sensitivity varies by sector. Assets with event-driven volatility (Energy, Consumer Staples) benefit most from news features. Utility-like sectors with stable, regulated cash flows show minimal benefit.

## Project Structure

```
news-volatility-dashboard/
├── app.py                  # Main Streamlit application
├── daily_features.csv      # Pre-computed news features
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Based On

This dashboard is based on the BEE3066 Term Project: **Information Arrival and Volatility Structures**

The full research project includes:
- Large-scale news processing (57M articles → 2M relevant)
- Two-stage semantic clustering
- Extensive EDA and feature engineering
- Comparative modelling framework

[Link to full project repo](#) ← Add your main project repo link

## License

MIT

## Author

David Clements  
University of Exeter, Economics  
BEE3066 Financial Data Science
