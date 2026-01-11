# 📈 News & Volatility Dashboard

Interactive dashboard exploring how news sentiment predicts extreme market volatility.

**[Live Demo](https://news-volatility-predictiondashboard-zs3egth3qvtkrfzp5cmczj.streamlit.app/)**

## What This Does

Implements a **hybrid residual model** that combines:
- **Baseline**: Logistic regression on market features (captures volatility persistence)
- **Residual**: Random forest on news features (captures nonlinear news effects)

The key insight: **news is most informative for extreme volatility events**, not routine fluctuations. The effect varies significantly across sectors.

## Features

- Enter any valid Yahoo Finance ticker
- Adjustable volatility percentile threshold (80th-95th)
- Interactive alpha curve, feature importance, and prediction time series
- Cached results for fast repeated queries

## Quick Start
```bash
git clone https://github.com/yourusername/news-volatility-dashboard.git
cd news-volatility-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Data

- **Market Data**: Live from Yahoo Finance
- **News Features**: Pre-computed daily sentiment features from ~2M financial articles (2016-2024), clustered into 7 themes (Markets, Macro, Earnings, Energy, Tech, Trade, Geopolitics)

## Model Architecture
```
Market Features ──► Logistic Regression (Baseline) ──┐
                                                     ├──► P = baseline + α × residual ──► Extreme Vol?
News Features ────► Random Forest (Residual) ────────┘
```

## Sample Results

| Sector | Baseline F1 | Hybrid F1 | Uplift |
|--------|-------------|-----------|--------|
| Energy (XLE) | 0.488 | 0.576 | +18% |
| Consumer Staples (XLP) | 0.745 | 0.864 | +16% |
| Healthcare (XLV) | 0.698 | 0.776 | +11% |
| Utilities (XLU) | 0.656 | 0.656 | 0% |

News sensitivity varies by sector. Event-driven sectors benefit most; stable regulated sectors show minimal uplift.

## Links

- [Full Project Report](https://docs.google.com/document/d/1d13AOZHMbSHORa-LI81_pXSDOy3eiwBfxVOXyzW3Y7w/edit?tab=t.cujvg6q004tp#heading=h.9hqcxrglxo85)
- [Project Data](https://drive.google.com/drive/folders/1FFkeFcLwR7XuvW-mU_Um4EJlmSUq3AQr?usp=drive_link)
