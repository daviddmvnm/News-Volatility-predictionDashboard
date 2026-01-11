"""
News & Volatility Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict

# ============================================================================
# CONFIG
# ============================================================================

TRADING_DAYS = 252
WINDOW = 5

BG = "#0f1117"
SURFACE = "#1a1d24"
ACCENT = "#00d4aa"
MUTED = "#6b7280"
TEXT = "#e5e7eb"
RED = "#ef4444"

st.set_page_config(
    page_title="vol × news",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}
    
    .stMarkdown, .stText, p, span, label {{
        color: {TEXT} !important;
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    h1 {{
        color: {TEXT} !important;
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.05em;
        margin-bottom: 2rem !important;
    }}
    
    h2, h3 {{
        color: {MUTED} !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}
    
    .block-container {{
        padding: 3rem 4rem !important;
        max-width: 1400px;
    }}
    
    .stTextInput > div > div > input {{
        background-color: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid #2d3139 !important;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: none !important;
    }}
    
    .stButton > button {{
        background-color: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid #2d3139 !important;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        padding: 0.5rem 1.5rem;
        transition: border-color 0.2s;
    }}
    
    .stButton > button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    
    .stSelectSlider > div {{
        background-color: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: transparent;
        border-bottom: 1px solid #2d3139;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {MUTED};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.5rem 0;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 1px solid {ACCENT};
    }}
    
    .stExpander {{
        background-color: {SURFACE};
        border: 1px solid #2d3139;
        border-radius: 4px;
    }}
    
    .stExpander p {{
        font-size: 0.8rem !important;
        color: {MUTED} !important;
    }}
    
    .stAlert {{
        background-color: {SURFACE} !important;
        border: none !important;
    }}
    
    hr {{
        border-color: #2d3139 !important;
        opacity: 0.3;
    }}
    
    .metric-box {{
        background-color: {SURFACE};
        border: 1px solid #2d3139;
        border-radius: 4px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        color: {MUTED};
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }}
    
    .metric-value {{
        color: {ACCENT};
        font-size: 1.8rem;
        font-weight: 300;
        font-family: 'IBM Plex Mono', monospace;
    }}
    
    .metric-delta {{
        color: {MUTED};
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }}
    
    div[data-testid="stDataFrame"] {{
        background-color: {SURFACE};
    }}
    
    .stSpinner > div {{
        border-color: {ACCENT} !important;
    }}
</style>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResidualModelResult:
    ticker: str
    best_alpha: float
    metrics: Dict[str, Dict[str, float]]
    series: pd.DataFrame
    feature_importance: pd.DataFrame
    residual_stats: Dict[str, float]
    alpha_curve: pd.DataFrame


# ============================================================================
# DATA
# ============================================================================

@st.cache_data
def load_news_features() -> pd.DataFrame:
    df = pd.read_csv("daily_features.csv")
    df["date_key"] = pd.to_datetime(df["date_key"])
    
    themes = ["earnings", "markets", "macro", "energy", "tech", "trade", "geopol"]
    
    cols = (
        ["date_key"]
        + [f"{t}_activity_share" for t in themes]
        + [f"{t}_sentiment_intensity" for t in themes]
        + [f"{t}_uncertainty_ratio" for t in themes]
    )
    
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("date_key").reset_index(drop=True)
    df["is_covid"] = df["date_key"].between("2020-03-01", "2020-12-31").astype(int)
    
    return df


def load_market_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    df = (
        df.reset_index()
          .rename(columns={"Date": "date_key"})
          .assign(date_key=lambda x: pd.to_datetime(x["date_key"]))
    )
    
    return df[["date_key", "Open", "High", "Low", "Close", "Volume"]]


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    df["vol"] = df["ret"].rolling(WINDOW).std() * np.sqrt(TRADING_DAYS)
    df["log_vol"] = np.log1p(df["vol"])
    return df


def merge_news_market(news: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = (
        market.merge(news, on="date_key", how="left")
              .sort_values("date_key")
              .reset_index(drop=True)
    )
    news_cols = news.columns.drop("date_key")
    df[news_cols] = df[news_cols].fillna(0)
    return df.dropna(subset=["vol"])


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    base_cols = ["ret", "vol", "log_vol"]
    intensity_cols = [c for c in df.columns if "sentiment_intensity" in c and "lag" not in c]
    
    for col in base_cols + intensity_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)
    
    return df.dropna().reset_index(drop=True)


def add_rv_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_abs"] = df["ret"].abs()
    df["rv1"] = df["ret_abs"].shift(-1)
    return df.dropna(subset=["rv1"]).reset_index(drop=True)


def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    primary_intensity_cols = ["markets_sentiment_intensity", "macro_sentiment_intensity"]
    primary_activity_cols = ["markets_activity_share", "macro_activity_share"]
    
    existing_intensity = [c for c in primary_intensity_cols if c in df.columns]
    existing_activity = [c for c in primary_activity_cols if c in df.columns]
    
    if existing_intensity and existing_activity:
        primary = df[existing_intensity].max(axis=1)
        activity = df[existing_activity].max(axis=1)
        df["primary_burst"] = primary * activity
        
        q_low, q_high = primary.quantile([0.2, 0.8])
        df["u_shape_regime"] = np.select(
            [primary < q_low, primary > q_high],
            [0, 2],
            default=1
        )
    else:
        df["primary_burst"] = 0
        df["u_shape_regime"] = 1
    
    dominance_cols = ["geopol_sentiment_intensity", "trade_sentiment_intensity"]
    existing_dom = [c for c in dominance_cols if c in df.columns]
    
    if existing_dom:
        df["dominance_burst"] = df[existing_dom].max(axis=1)
    else:
        df["dominance_burst"] = 0
    
    return df.dropna().reset_index(drop=True)


def build_dataset(ticker: str, news: pd.DataFrame) -> pd.DataFrame:
    market = load_market_data(ticker, news["date_key"].min(), news["date_key"].max())
    market = add_market_features(market)
    
    df = merge_news_market(news, market)
    df = add_lags(df)
    df = add_rv_targets(df)
    df = engineer_interactions(df)
    
    return df


# ============================================================================
# MODEL
# ============================================================================

def run_residual_model(
    df_final: pd.DataFrame,
    ticker: str,
    q_extreme: float = 0.9,
    alpha_grid: np.ndarray = np.linspace(0, 1, 21)
) -> ResidualModelResult:
    
    df = df_final.copy()
    
    df["ret_abs"] = df["ret"].abs()
    df["rv1_target"] = df["ret_abs"].shift(-1)
    df = df.dropna(subset=["rv1_target"]).reset_index(drop=True)
    
    def expanding_binary_extreme(series, q):
        thresholds = [series.iloc[:i+1].quantile(q) for i in range(len(series))]
        return (series >= thresholds).astype(int)
    
    df["rv1_extreme"] = expanding_binary_extreme(df["rv1_target"], q_extreme)
    
    market_only = [
        "vol", "vol_lag1", "vol_lag3",
        "ret", "ret_lag1", "ret_lag3"
    ]
    
    news_only = [
        "markets_sentiment_intensity", "markets_sentiment_intensity_lag3",
        "primary_burst",
        "macro_sentiment_intensity", "macro_sentiment_intensity_lag1", "macro_sentiment_intensity_lag3",
        "energy_sentiment_intensity", "energy_sentiment_intensity_lag1", "energy_sentiment_intensity_lag3",
        "tech_sentiment_intensity", "tech_sentiment_intensity_lag1", "tech_sentiment_intensity_lag3",
        "u_shape_regime", "dominance_burst"
    ]
    
    market_only = [c for c in market_only if c in df.columns]
    news_only = [c for c in news_only if c in df.columns]
    
    cut = int(len(df) * 0.9)
    
    X_m_tr = df.iloc[:cut][market_only]
    X_m_te = df.iloc[cut:][market_only]
    X_n_tr = df.iloc[:cut][news_only]
    X_n_te = df.iloc[cut:][news_only]
    
    y_tr = df.iloc[:cut]["rv1_extreme"].values
    y_te = df.iloc[cut:]["rv1_extreme"].values
    
    tscv = TimeSeriesSplit(n_splits=5)
    oof_baseline = np.zeros(len(y_tr))
    
    scaler_m = StandardScaler()
    scaler_n = StandardScaler()
    
    for tr_idx, va_idx in tscv.split(X_m_tr):
        X_tr = scaler_m.fit_transform(X_m_tr.iloc[tr_idx])
        X_va = scaler_m.transform(X_m_tr.iloc[va_idx])
        
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X_tr, y_tr[tr_idx])
        
        oof_baseline[va_idx] = lr.predict_proba(X_va)[:, 1]
    
    true_residuals = y_tr - oof_baseline
    residual_mean = true_residuals.mean()
    residual_std = true_residuals.std()
    true_residuals_centered = true_residuals - residual_mean
    
    X_m_tr_s = scaler_m.fit_transform(X_m_tr)
    X_m_te_s = scaler_m.transform(X_m_te)
    
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_m_tr_s, y_tr)
    baseline_probs = baseline.predict_proba(X_m_te_s)[:, 1]
    
    X_n_tr_s = scaler_n.fit_transform(X_n_tr)
    X_n_te_s = scaler_n.transform(X_n_te)
    
    tree = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=30,
        min_samples_split=60,
        max_features="sqrt",
        random_state=42
    )
    tree.fit(X_n_tr_s, true_residuals_centered)
    residual_pred = tree.predict(X_n_te_s)
    
    from sklearn.metrics import precision_score, recall_score
    
    alpha_rows = []
    for a in alpha_grid:
        combined = np.clip(baseline_probs + a * residual_pred, 0, 1)
        pred = (combined > 0.5).astype(int)
        
        alpha_rows.append({
            "alpha": float(a),
            "f1": f1_score(y_te, pred, zero_division=0),
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0)
        })
    
    alpha_df = pd.DataFrame(alpha_rows)
    best_row = alpha_df.loc[alpha_df["f1"].idxmax()]
    best_alpha = float(best_row["alpha"])
    
    final_combined = np.clip(baseline_probs + best_alpha * residual_pred, 0, 1)
    
    series = pd.DataFrame({
        "date": df["date_key"].iloc[cut:].values,
        "baseline_prob": baseline_probs,
        "residual": residual_pred,
        "combined_prob": final_combined,
        "y_true": y_te
    })
    
    importance = (
        pd.DataFrame({
            "feature": news_only,
            "importance": tree.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    
    return ResidualModelResult(
        ticker=ticker,
        best_alpha=best_alpha,
        metrics={
            "baseline": {
                "f1": f1_score(y_te, (baseline_probs > 0.5).astype(int), zero_division=0),
                "accuracy": accuracy_score(y_te, (baseline_probs > 0.5).astype(int)),
                "precision": precision_score(y_te, (baseline_probs > 0.5).astype(int), zero_division=0),
                "recall": recall_score(y_te, (baseline_probs > 0.5).astype(int), zero_division=0)
            },
            "hybrid": {
                "f1": float(best_row["f1"]),
                "accuracy": float(best_row["accuracy"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"])
            }
        },
        series=series,
        feature_importance=importance,
        residual_stats={
            "mean": float(residual_pred.mean()),
            "std": float(residual_pred.std()),
            "min": float(residual_pred.min()),
            "max": float(residual_pred.max()),
            "raw_train_mean": float(residual_mean),
            "raw_train_std": float(residual_std)
        },
        alpha_curve=alpha_df
    )


@st.cache_data(show_spinner=False)
def train_model_cached(ticker: str, q_extreme: float) -> dict:
    news = load_news_features()
    df = build_dataset(ticker, news)
    result = run_residual_model(df, ticker, q_extreme=q_extreme)
    
    return {
        "ticker": result.ticker,
        "best_alpha": result.best_alpha,
        "metrics": result.metrics,
        "series": result.series.to_dict(orient="records"),
        "feature_importance": result.feature_importance.to_dict(orient="records"),
        "residual_stats": result.residual_stats,
        "alpha_curve": result.alpha_curve.to_dict(orient="records")
    }


# ============================================================================
# CHARTS
# ============================================================================

def plot_alpha_curve(alpha_curve: pd.DataFrame, best_alpha: float, show_metrics: list = None) -> go.Figure:
    if show_metrics is None:
        show_metrics = ["f1"]
    
    fig = go.Figure()
    
    colors = {
        "f1": ACCENT,
        "accuracy": "#a78bfa",  # purple
        "precision": "#f472b6",  # pink
        "recall": "#fbbf24"  # amber
    }
    
    for metric in show_metrics:
        if metric in alpha_curve.columns:
            fig.add_trace(go.Scatter(
                x=alpha_curve["alpha"],
                y=alpha_curve[metric],
                mode="lines",
                name=metric,
                line=dict(color=colors.get(metric, MUTED), width=1.5),
                hovertemplate=f"α=%{{x:.2f}}<br>{metric}=%{{y:.3f}}<extra></extra>"
            ))
    
    # Mark best alpha point on F1 curve
    if "f1" in show_metrics:
        best_f1 = alpha_curve.loc[alpha_curve["alpha"].round(2) == round(best_alpha, 2), "f1"]
        if len(best_f1) > 0:
            fig.add_trace(go.Scatter(
                x=[best_alpha],
                y=[best_f1.values[0]],
                mode="markers",
                marker=dict(color=ACCENT, size=8),
                hoverinfo="skip",
                showlegend=False
            ))
    
    # Baseline reference line for F1
    if "f1" in show_metrics:
        baseline_f1 = alpha_curve.loc[alpha_curve["alpha"] == 0, "f1"].values[0]
        fig.add_hline(
            y=baseline_f1,
            line_dash="dot",
            line_color=MUTED,
            line_width=1,
            annotation_text=f"baseline f1 {baseline_f1:.3f}",
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color=MUTED
        )
    
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono", color=MUTED, size=11),
        margin=dict(l=50, r=30, t=40, b=40),
        title=dict(text="alpha curve", font=dict(size=12, color=MUTED)),
        xaxis_title="α",
        yaxis_title="score",
        height=350,
        showlegend=len(show_metrics) > 1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),
        xaxis=dict(gridcolor="#1f2329", zerolinecolor="#1f2329", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1f2329", zerolinecolor="#1f2329", tickfont=dict(size=10))
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    df = importance_df.head(8)
    
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color=ACCENT,
        marker_line_width=0,
        hovertemplate="%{y}<br>%{x:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono", color=MUTED, size=11),
        margin=dict(l=50, r=30, t=40, b=40),
        title=dict(text="feature importance", font=dict(size=12, color=MUTED)),
        xaxis_title="",
        yaxis_title="",
        height=350,
        showlegend=False,
        xaxis=dict(gridcolor="#1f2329", zerolinecolor="#1f2329", tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", gridcolor="#1f2329", tickfont=dict(size=9))
    )
    
    return fig


def plot_prediction_series(series_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35]
    )
    
    fig.add_trace(go.Scatter(
        x=series_df["date"],
        y=series_df["baseline_prob"],
        name="baseline",
        line=dict(color=MUTED, width=1),
        hovertemplate="%{y:.3f}<extra>baseline</extra>"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=series_df["date"],
        y=series_df["combined_prob"],
        name="hybrid",
        line=dict(color=ACCENT, width=1),
        hovertemplate="%{y:.3f}<extra>hybrid</extra>"
    ), row=1, col=1)
    
    extreme_dates = series_df[series_df["y_true"] == 1]["date"]
    if len(extreme_dates) > 0:
        fig.add_trace(go.Scatter(
            x=extreme_dates,
            y=[1.02] * len(extreme_dates),
            mode="markers",
            marker=dict(color=RED, size=4, symbol="triangle-down"),
            hovertemplate="%{x}<extra>extreme</extra>"
        ), row=1, col=1)
    
    colors = [ACCENT if r > 0 else MUTED for r in series_df["residual"]]
    fig.add_trace(go.Bar(
        x=series_df["date"],
        y=series_df["residual"],
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="%{y:.3f}<extra>residual</extra>"
    ), row=2, col=1)
    
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono", color=MUTED, size=11),
        margin=dict(l=50, r=30, t=40, b=40),
        height=420,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor="#1f2329", zerolinecolor="#1f2329", tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#1f2329", zerolinecolor="#1f2329", tickfont=dict(size=9))
    
    return fig


# ============================================================================
# COMPONENTS
# ============================================================================

def metric_card(label: str, value: str, delta: str = None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown("# vol × news")
    st.markdown(f"""
    <p style="color: {MUTED}; font-size: 0.85rem; margin-bottom: 0.5rem;">
    Predicting extreme volatility using news sentiment features. This dashboard accompanies a capstone project exploring whether semantic information from financial news improves volatility forecasting beyond market persistence.
    <a href="https://docs.google.com/document/d/1d13AOZHMbSHORa-LI81_pXSDOy3eiwBfxVOXyzW3Y7w/edit?tab=t.cujvg6q004tp" style="color: {ACCENT};">Read the full report →</a>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        ticker = st.text_input("ticker", value="SPY", label_visibility="collapsed", placeholder="ticker").upper().strip()
    
    with col2:
        percentile = st.select_slider(
            "percentile",
            options=[0.80, 0.85, 0.90, 0.95],
            value=0.90,
            format_func=lambda x: f"p{int(x*100)}",
            label_visibility="collapsed"
        )
    
    with col3:
        run_button = st.button("run", use_container_width=True)
    
    with col4:
        pass
    
    # Quick picks
    st.markdown(f'<p style="color: {MUTED}; font-size: 0.75rem; margin-top: 0.5rem;">quick picks — or use any <a href="https://finance.yahoo.com/" style="color: {ACCENT};">yfinance</a> ticker</p>', unsafe_allow_html=True)
    
    qp_cols = st.columns(5)
    quick_picks = [
        ("XLF", "Financials"),
        ("XLE", "Energy"),
        ("XLK", "Tech"),
        ("XLV", "Healthcare"),
        ("XLU", "Utilities"),
    ]
    
    for i, (sym, label) in enumerate(quick_picks):
        with qp_cols[i]:
            if st.button(f"{sym} ({label})", use_container_width=True, key=f"qp_{sym}"):
                st.session_state["selected_ticker"] = sym
                st.rerun()
    
    # Check for quick pick selection
    if "selected_ticker" in st.session_state:
        ticker = st.session_state.pop("selected_ticker")
        run_button = True
    
    # Explain controls
    with st.expander("what do these controls mean?"):
        st.markdown(f"""
        **Ticker**: Any valid Yahoo Finance symbol (e.g., SPY, AAPL, XLE, TSLA, SHEL.L)
        
        **Percentile (p80-p95)**: Defines what counts as "extreme" volatility. 
        - p90 means we're predicting whether tomorrow's volatility will be in the top 10% of historical values
        - Higher percentiles = rarer, more extreme events
        - Lower percentiles = more frequent "elevated" volatility days
        
        The model uses an expanding window, so the threshold updates as new data arrives (no look-ahead bias).
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run model or use cached result
    if run_button and ticker:
        with st.spinner(""):
            try:
                result = train_model_cached(ticker, percentile)
                st.session_state["last_result"] = result
                st.session_state["last_ticker"] = ticker
                st.session_state["last_percentile"] = percentile
            except ValueError as e:
                st.error(f"invalid ticker: {ticker}")
                return
            except Exception as e:
                st.error(f"error: {str(e)}")
                return
    
    # Display results if we have them
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        ticker = st.session_state["last_ticker"]
        percentile = st.session_state["last_percentile"]
        
        series_df = pd.DataFrame(result["series"])
        importance_df = pd.DataFrame(result["feature_importance"])
        alpha_curve_df = pd.DataFrame(result["alpha_curve"])
        
        baseline_f1 = result["metrics"]["baseline"]["f1"]
        hybrid_f1 = result["metrics"]["hybrid"]["f1"]
        delta_f1 = hybrid_f1 - baseline_f1
        pct_uplift = (delta_f1 / baseline_f1 * 100) if baseline_f1 > 0 else 0
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card("alpha", f"{result['best_alpha']:.2f}")
        with col2:
            metric_card("baseline f1", f"{baseline_f1:.3f}")
        with col3:
            metric_card("hybrid f1", f"{hybrid_f1:.3f}", f"+{delta_f1:.3f}" if delta_f1 > 0 else f"{delta_f1:.3f}")
        with col4:
            metric_card("samples", f"{len(series_df)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["alpha", "features", "series"])
        
        with tab1:
            # Metric toggles
            st.markdown(f'<p style="color: {MUTED}; font-size: 0.75rem;">show metrics:</p>', unsafe_allow_html=True)
            tog_cols = st.columns(4)
            with tog_cols[0]:
                show_f1 = st.checkbox("f1", value=True, key="show_f1")
            with tog_cols[1]:
                show_acc = st.checkbox("accuracy", value=False, key="show_acc")
            with tog_cols[2]:
                show_prec = st.checkbox("precision", value=False, key="show_prec")
            with tog_cols[3]:
                show_rec = st.checkbox("recall", value=False, key="show_rec")
            
            show_metrics = []
            if show_f1:
                show_metrics.append("f1")
            if show_acc:
                show_metrics.append("accuracy")
            if show_prec:
                show_metrics.append("precision")
            if show_rec:
                show_metrics.append("recall")
            
            if not show_metrics:
                show_metrics = ["f1"]
            
            st.plotly_chart(plot_alpha_curve(alpha_curve_df, result["best_alpha"], show_metrics), use_container_width=True)
            with st.expander("info"):
                st.markdown(f"""
                **What is the alpha curve?**
                
                This shows how model performance changes as we vary the weight on news-derived signals.
                
                - **α = 0**: Pure baseline — only uses lagged volatility and returns (market features)
                - **α = 1**: Full news weight — baseline prediction plus the full news residual adjustment
                - **Optimal α = {result['best_alpha']:.2f}**: The weight that maximises F1 on the out-of-sample test set
                
                **Metrics explained:**
                - **F1**: Harmonic mean of precision and recall — balances both
                - **Accuracy**: % of days correctly classified (can be misleading with imbalanced classes)
                - **Precision**: When we predict extreme vol, how often are we right?
                - **Recall**: Of all actual extreme vol days, how many did we catch?
                
                The dotted line shows baseline F1 performance (α=0) for reference.
                """)
        
        with tab2:
            st.plotly_chart(plot_feature_importance(importance_df), use_container_width=True)
            with st.expander("info"):
                st.markdown("""
                **What are these features?**
                
                These are the news-derived features used by the Random Forest to predict the residual (the part the baseline misses).
                
                - **sentiment_intensity**: How strongly positive/negative the news coverage is for a given theme (markets, macro, energy, tech, etc.)
                - **primary_burst**: Interaction of markets/macro sentiment intensity with attention share — captures "big news days"
                - **dominance_burst**: Same concept for geopolitics/trade themes
                - **u_shape_regime**: Indicator for whether sentiment is in an extreme state (very high or very low)
                - **_lag1, _lag3**: Lagged versions (1-day and 3-day) to capture delayed effects
                
                **Importance**: Higher values mean the feature contributes more to the residual model's predictions. This tells you which news themes matter most for this asset's volatility.
                """)
        
        with tab3:
            st.plotly_chart(plot_prediction_series(series_df), use_container_width=True)
            with st.expander("info"):
                st.markdown("""
                **Top panel: Probability predictions**
                - **Grey line**: Baseline probability of extreme volatility (market features only)
                - **Teal line**: Hybrid probability (baseline + α × news residual)
                - **Red triangles**: Days where extreme volatility actually occurred
                
                **Bottom panel: News residual contribution**
                - **Teal bars (positive)**: News signals suggest *higher* volatility than the baseline expects
                - **Grey bars (negative)**: News signals suggest *lower* volatility than the baseline expects
                
                **Why are residuals mostly negative?**
                
                The baseline model uses lagged volatility, which captures volatility clustering (high vol tends to follow high vol). However, when there's active news coverage, information gets incorporated into prices faster, meaning yesterday's volatility becomes less predictive of today's.
                
                The negative residuals represent the news model learning: "when news flow is active, the autoregressive baseline's persistence assumption is too strong — volatility mean-reverts faster than the baseline expects."
                
                The *variation* in residual magnitude is the signal. Days with less negative residuals (or positive ones) are when news suggests the baseline might be *underestimating* risk.
                """)
        
        # Interpretation section - after charts but before residual stats
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("interpretation", expanded=True):
            # Determine sensitivity level
            if result['best_alpha'] > 0.7:
                sensitivity = "highly"
                sensitivity_detail = "News features provide substantial predictive information beyond what market persistence alone captures."
            elif result['best_alpha'] > 0.3:
                sensitivity = "moderately"
                sensitivity_detail = "News adds meaningful signal, but market dynamics remain the primary driver."
            else:
                sensitivity = "minimally"
                sensitivity_detail = "Volatility for this asset is primarily driven by autoregressive persistence; news adds limited incremental value."
            
            # Determine uplift interpretation
            if pct_uplift > 10:
                uplift_interp = "substantial improvement"
            elif pct_uplift > 0:
                uplift_interp = "modest improvement"
            else:
                uplift_interp = "no improvement"
            
            # Precision/recall interpretation
            baseline_prec = result["metrics"]["baseline"]["precision"]
            hybrid_prec = result["metrics"]["hybrid"]["precision"]
            baseline_rec = result["metrics"]["baseline"]["recall"]
            hybrid_rec = result["metrics"]["hybrid"]["recall"]
            
            prec_delta = hybrid_prec - baseline_prec
            rec_delta = hybrid_rec - baseline_rec
            
            st.markdown(f"""
            ### {ticker} at p{int(percentile*100)}
            
            **News sensitivity**
            
            Optimal α = **{result['best_alpha']:.2f}** — this asset is **{sensitivity}** sensitive to news signals. {sensitivity_detail}
            
            **Performance comparison**
            
            |  | Baseline | Hybrid | Δ |
            |--|----------|--------|---|
            | F1 | {baseline_f1:.3f} | {hybrid_f1:.3f} | {delta_f1:+.3f} |
            | Precision | {baseline_prec:.3f} | {hybrid_prec:.3f} | {prec_delta:+.3f} |
            | Recall | {baseline_rec:.3f} | {hybrid_rec:.3f} | {rec_delta:+.3f} |
            
            Adding news features provides **{uplift_interp}** ({pct_uplift:+.1f}% F1 uplift) in identifying extreme volatility days.
            
            **What this means**
            
            The hybrid model combines a baseline that captures volatility persistence (yesterday's vol predicts today's) with a news-based residual that adjusts this prediction. The α parameter controls how much weight to give the news signal — higher α means news is more informative for this asset.
            """)
        
        with st.expander("residual stats"):
            st.code(f"""test mean:  {result['residual_stats']['mean']:.4f}
test std:   {result['residual_stats']['std']:.4f}
train bias: {result['residual_stats']['raw_train_mean']:.4f}""")
    
    else:
        # Default state
        st.markdown(f"""
        <div style="color: {MUTED}; font-size: 0.8rem; margin-top: 2rem;">
        Enter a ticker symbol and click run to analyse.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("how does this work?"):
            st.markdown("""
            **The problem**: Volatility is highly persistent — yesterday's volatility strongly predicts today's. Simple autoregressive models capture this well, but they miss information contained in news.
            
            **The approach**: A two-stage hybrid model:
            
            1. **Baseline model** (Logistic Regression): Predicts extreme volatility using only market features — lagged volatility, lagged returns. This captures the persistent, autoregressive structure.
            
            2. **Residual model** (Random Forest): Trained on what the baseline *misses* — the residuals. Uses news features: sentiment intensity across themes (markets, macro, energy, tech, geopolitics, trade), attention bursts, and regime indicators.
            
            3. **Combined prediction**: `P(extreme) = baseline + α × residual`
            
            The α parameter is tuned on a validation set to find the optimal weight for news information.
            
            **Key finding**: News doesn't predict volatility directly. Instead, it modulates how much to trust the baseline's persistence-based forecast. When news is active, information gets priced in faster, and the baseline's "yesterday predicts today" assumption becomes less reliable.
            
            **Data**: News features are derived from ~2 million financial news articles (2016-2024), clustered into 7 thematic categories using sentence embeddings.
            """)


if __name__ == "__main__":
    main()
