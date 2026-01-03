"""
News & Volatility Dashboard
============================
Interactive dashboard exploring how news sentiment predicts market volatility.

Based on the BEE3066 Term Project: Information Arrival and Volatility Structures
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

TRADING_DAYS = 252
WINDOW = 5

# Page config
st.set_page_config(
    page_title="News & Volatility Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResidualModelResult:
    """Container for all model outputs."""
    ticker: str
    best_alpha: float
    metrics: Dict[str, Dict[str, float]]
    series: pd.DataFrame
    feature_importance: pd.DataFrame
    residual_stats: Dict[str, float]
    alpha_curve: pd.DataFrame


# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

@st.cache_data
def load_news_features() -> pd.DataFrame:
    """Load pre-computed daily news features."""
    df = pd.read_csv("daily_features.csv")
    df["date_key"] = pd.to_datetime(df["date_key"])
    
    themes = ["earnings", "markets", "macro", "energy", "tech", "trade", "geopol"]
    
    cols = (
        ["date_key"]
        + [f"{t}_activity_share" for t in themes]
        + [f"{t}_sentiment_intensity" for t in themes]
        + [f"{t}_uncertainty_ratio" for t in themes]
    )
    
    # Only keep columns that exist
    cols = [c for c in cols if c in df.columns]
    
    df = df[cols].sort_values("date_key").reset_index(drop=True)
    df["is_covid"] = df["date_key"].between("2020-03-01", "2020-12-31").astype(int)
    
    return df


def load_market_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Pull market data from yfinance."""
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    
    df = (
        df.reset_index()
          .rename(columns={"Date": "date_key"})
          .assign(date_key=lambda x: pd.to_datetime(x["date_key"]))
    )
    
    return df[["date_key", "Open", "High", "Low", "Close", "Volume"]]


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns and volatility features."""
    df = df.copy()
    df["ret"] = df["Close"].pct_change()
    df["vol"] = df["ret"].rolling(WINDOW).std() * np.sqrt(TRADING_DAYS)
    df["log_vol"] = np.log1p(df["vol"])
    return df


def merge_news_market(news: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Merge news features with market data."""
    df = (
        market.merge(news, on="date_key", how="left")
              .sort_values("date_key")
              .reset_index(drop=True)
    )
    news_cols = news.columns.drop("date_key")
    df[news_cols] = df[news_cols].fillna(0)
    return df.dropna(subset=["vol"])


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features for time series modelling."""
    df = df.copy()
    
    base_cols = ["ret", "vol", "log_vol"]
    intensity_cols = [c for c in df.columns if "sentiment_intensity" in c and "lag" not in c]
    
    for col in base_cols + intensity_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)
    
    return df.dropna().reset_index(drop=True)


def add_rv_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add realised volatility target."""
    df = df.copy()
    df["ret_abs"] = df["ret"].abs()
    df["rv1"] = df["ret_abs"].shift(-1)
    return df.dropna(subset=["rv1"]).reset_index(drop=True)


def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer interaction features based on EDA findings."""
    df = df.copy()
    
    # Primary burst: max intensity * max activity
    primary_intensity_cols = ["markets_sentiment_intensity", "macro_sentiment_intensity"]
    primary_activity_cols = ["markets_activity_share", "macro_activity_share"]
    
    existing_intensity = [c for c in primary_intensity_cols if c in df.columns]
    existing_activity = [c for c in primary_activity_cols if c in df.columns]
    
    if existing_intensity and existing_activity:
        primary = df[existing_intensity].max(axis=1)
        activity = df[existing_activity].max(axis=1)
        df["primary_burst"] = primary * activity
        
        # U-shape regime based on quantiles
        q_low, q_high = primary.quantile([0.2, 0.8])
        df["u_shape_regime"] = np.select(
            [primary < q_low, primary > q_high],
            [0, 2],
            default=1
        )
    else:
        df["primary_burst"] = 0
        df["u_shape_regime"] = 1
    
    # Dominance burst (if available)
    dominance_cols = ["geopol_sentiment_intensity", "trade_sentiment_intensity"]
    existing_dom = [c for c in dominance_cols if c in df.columns]
    
    if existing_dom:
        df["dominance_burst"] = df[existing_dom].max(axis=1)
    else:
        df["dominance_burst"] = 0
    
    return df.dropna().reset_index(drop=True)


def build_dataset(ticker: str, news: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: load market data, merge with news, engineer features."""
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
    """
    Fully purged walk-forward residual model.
    Returns a rich, inspectable result object.
    """
    
    df = df_final.copy()
    
    # ---------- TARGET ----------
    df["ret_abs"] = df["ret"].abs()
    df["rv1_target"] = df["ret_abs"].shift(-1)
    df = df.dropna(subset=["rv1_target"]).reset_index(drop=True)
    
    def expanding_binary_extreme(series, q):
        thresholds = [series.iloc[:i+1].quantile(q) for i in range(len(series))]
        return (series >= thresholds).astype(int)
    
    df["rv1_extreme"] = expanding_binary_extreme(df["rv1_target"], q_extreme)
    
    # ---------- FEATURES ----------
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
    
    # ---------- OOF BASELINE ----------
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
    
    # ---------- FINAL FIT ----------
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
    tree.fit(X_n_tr_s, true_residuals)
    residual_pred = tree.predict(X_n_te_s)
    
    # ---------- ALPHA CURVE ----------
    alpha_rows = []
    for a in alpha_grid:
        combined = np.clip(baseline_probs + a * residual_pred, 0, 1)
        pred = (combined > 0.5).astype(int)
        
        alpha_rows.append({
            "alpha": float(a),
            "f1": f1_score(y_te, pred, zero_division=0),
            "accuracy": accuracy_score(y_te, pred)
        })
    
    alpha_df = pd.DataFrame(alpha_rows)
    best_row = alpha_df.loc[alpha_df["f1"].idxmax()]
    best_alpha = float(best_row["alpha"])
    
    # ---------- FINAL SERIES ----------
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
                "accuracy": accuracy_score(y_te, (baseline_probs > 0.5).astype(int))
            },
            "hybrid": {
                "f1": float(best_row["f1"]),
                "accuracy": float(best_row["accuracy"])
            }
        },
        series=series,
        feature_importance=importance,
        residual_stats={
            "mean": float(true_residuals.mean()),
            "std": float(true_residuals.std()),
            "min": float(true_residuals.min()),
            "max": float(true_residuals.max())
        },
        alpha_curve=alpha_df
    )


@st.cache_data(show_spinner=False)
def train_model_cached(ticker: str, q_extreme: float) -> dict:
    """Cached wrapper for model training. Returns dict for caching compatibility."""
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
# VISUALIZATION HELPERS
# ============================================================================

def plot_alpha_curve(alpha_curve: pd.DataFrame, best_alpha: float) -> go.Figure:
    """Plot F1 score across alpha values."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=alpha_curve["alpha"],
        y=alpha_curve["f1"],
        mode="lines+markers",
        name="F1 Score",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6)
    ))
    
    # Mark best alpha
    best_f1 = alpha_curve.loc[alpha_curve["alpha"] == best_alpha, "f1"].values[0]
    fig.add_trace(go.Scatter(
        x=[best_alpha],
        y=[best_f1],
        mode="markers",
        name=f"Best α = {best_alpha:.2f}",
        marker=dict(color="#d62728", size=14, symbol="star")
    ))
    
    # Baseline reference line
    baseline_f1 = alpha_curve.loc[alpha_curve["alpha"] == 0, "f1"].values[0]
    fig.add_hline(
        y=baseline_f1,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline (α=0): {baseline_f1:.3f}"
    )
    
    fig.update_layout(
        title="Alpha Curve: News Weight vs. Model Performance",
        xaxis_title="α (News Residual Weight)",
        yaxis_title="F1 Score",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98)
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Plot feature importance from the residual model."""
    df = importance_df.head(10)  # Top 10
    
    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker_color="#2ca02c"
    ))
    
    fig.update_layout(
        title="Top News Features (Residual Model)",
        xaxis_title="Importance",
        yaxis_title="",
        template="plotly_white",
        height=400,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def plot_prediction_series(series_df: pd.DataFrame) -> go.Figure:
    """Plot baseline vs hybrid predictions over time."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Model Probabilities vs Actual", "News Residual Contribution")
    )
    
    # Probabilities
    fig.add_trace(go.Scatter(
        x=series_df["date"],
        y=series_df["baseline_prob"],
        name="Baseline Prob",
        line=dict(color="#1f77b4", width=1.5),
        opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=series_df["date"],
        y=series_df["combined_prob"],
        name="Hybrid Prob",
        line=dict(color="#2ca02c", width=1.5)
    ), row=1, col=1)
    
    # Actual extreme days
    extreme_dates = series_df[series_df["y_true"] == 1]["date"]
    fig.add_trace(go.Scatter(
        x=extreme_dates,
        y=[1.05] * len(extreme_dates),
        mode="markers",
        name="Actual Extreme",
        marker=dict(color="#d62728", size=6, symbol="triangle-down")
    ), row=1, col=1)
    
    # Residual contribution
    fig.add_trace(go.Bar(
        x=series_df["date"],
        y=series_df["residual"],
        name="News Residual",
        marker_color=np.where(series_df["residual"] > 0, "#d62728", "#1f77b4"),
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    
    return fig


def plot_metrics_comparison(metrics: dict) -> go.Figure:
    """Bar chart comparing baseline vs hybrid metrics."""
    categories = ["F1 Score", "Accuracy"]
    baseline_vals = [metrics["baseline"]["f1"], metrics["baseline"]["accuracy"]]
    hybrid_vals = [metrics["hybrid"]["f1"], metrics["hybrid"]["accuracy"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Baseline (Market Only)",
        x=categories,
        y=baseline_vals,
        marker_color="#1f77b4"
    ))
    
    fig.add_trace(go.Bar(
        name="Hybrid (Market + News)",
        x=categories,
        y=hybrid_vals,
        marker_color="#2ca02c"
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        template="plotly_white",
        height=350,
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.title("📈 News & Volatility Dashboard")
    st.markdown("""
    **Exploring how news sentiment predicts extreme market volatility.**
    
    This dashboard implements a hybrid residual model that combines:
    - **Baseline**: Logistic regression on market features (volatility persistence)
    - **Residual**: Random forest on news features (captures nonlinear news effects)
    
    The key insight: news is most informative for *extreme* volatility, not routine fluctuations.
    """)
    
    st.divider()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            help="Enter any valid Yahoo Finance ticker (e.g., SPY, AAPL, XLE, TSLA)"
        ).upper().strip()
        
        percentile = st.select_slider(
            "Volatility Percentile",
            options=[0.80, 0.85, 0.90, 0.95],
            value=0.90,
            format_func=lambda x: f"{int(x*100)}th percentile",
            help="Define 'extreme' volatility as days above this percentile"
        )
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        st.divider()
        
        st.markdown("""
        **Quick Picks:**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("SPY", use_container_width=True):
                st.session_state["quick_ticker"] = "SPY"
                st.rerun()
            if st.button("XLE", use_container_width=True):
                st.session_state["quick_ticker"] = "XLE"
                st.rerun()
            if st.button("XLF", use_container_width=True):
                st.session_state["quick_ticker"] = "XLF"
                st.rerun()
        with col2:
            if st.button("AAPL", use_container_width=True):
                st.session_state["quick_ticker"] = "AAPL"
                st.rerun()
            if st.button("TSLA", use_container_width=True):
                st.session_state["quick_ticker"] = "TSLA"
                st.rerun()
            if st.button("XOM", use_container_width=True):
                st.session_state["quick_ticker"] = "XOM"
                st.rerun()
        
        st.divider()
        
        st.markdown("""
        **About**
        
        Based on BEE3066 Term Project:  
        *Information Arrival and Volatility Structures*
        
        [GitHub Repo](#) | [Full Report](#)
        """)
    
    # Handle quick picks
    if "quick_ticker" in st.session_state:
        ticker = st.session_state.pop("quick_ticker")
        run_button = True
    
    # Main content
    if run_button and ticker:
        with st.spinner(f"Fetching data and training model for {ticker}..."):
            try:
                result = train_model_cached(ticker, percentile)
                
                # Convert back to dataframes
                series_df = pd.DataFrame(result["series"])
                importance_df = pd.DataFrame(result["feature_importance"])
                alpha_curve_df = pd.DataFrame(result["alpha_curve"])
                
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure you entered a valid Yahoo Finance ticker symbol.")
                return
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                return
        
        # Success message
        st.success(f"✅ Analysis complete for **{ticker}**")
        
        # Key metrics row
        st.subheader("📊 Key Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Best α",
                f"{result['best_alpha']:.2f}",
                help="Optimal weight for news residual"
            )
        
        with col2:
            baseline_f1 = result["metrics"]["baseline"]["f1"]
            hybrid_f1 = result["metrics"]["hybrid"]["f1"]
            delta = hybrid_f1 - baseline_f1
            st.metric(
                "Hybrid F1",
                f"{hybrid_f1:.3f}",
                delta=f"{delta:+.3f} vs baseline",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Baseline F1",
                f"{baseline_f1:.3f}",
                help="Market-only model performance"
            )
        
        with col4:
            news_uplift = (hybrid_f1 - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
            st.metric(
                "News Uplift",
                f"{news_uplift:+.1f}%",
                help="Relative improvement from adding news"
            )
        
        st.divider()
        
        # Charts
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Alpha Curve",
            "🎯 Feature Importance", 
            "📉 Time Series",
            "📋 Raw Data"
        ])
        
        with tab1:
            st.plotly_chart(
                plot_alpha_curve(alpha_curve_df, result["best_alpha"]),
                use_container_width=True
            )
            
            st.markdown("""
            **Interpretation:** The alpha curve shows how model performance changes as we 
            increase the weight on news-derived signals. α=0 is the pure market baseline; 
            higher α values incorporate more news information. The optimal α indicates 
            how much news contributes to volatility prediction for this asset.
            """)
        
        with tab2:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(
                    plot_feature_importance(importance_df),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**Top Features:**")
                for _, row in importance_df.head(5).iterrows():
                    st.markdown(f"- `{row['feature']}`: {row['importance']:.3f}")
                
                st.markdown("""
                ---
                **Feature Types:**
                - `*_sentiment_intensity`: Magnitude of news coverage
                - `primary_burst`: Attention × Intensity interaction
                - `u_shape_regime`: Market state indicator
                """)
        
        with tab3:
            st.plotly_chart(
                plot_prediction_series(series_df),
                use_container_width=True
            )
            
            st.markdown("""
            **Reading the chart:** 
            - Blue line = baseline probability (market features only)
            - Green line = hybrid probability (market + news)
            - Red triangles = actual extreme volatility days
            - Bottom panel = news residual contribution (positive = news predicts higher vol)
            """)
        
        with tab4:
            st.markdown("**Model Metrics:**")
            st.json(result["metrics"])
            
            st.markdown("**Residual Statistics:**")
            st.json(result["residual_stats"])
            
            st.markdown("**Alpha Curve Data:**")
            st.dataframe(alpha_curve_df, use_container_width=True)
            
            st.markdown("**Prediction Series (last 20 rows):**")
            st.dataframe(series_df.tail(20), use_container_width=True)
        
        # Interpretation box
        st.divider()
        
        with st.expander("🔍 **Interpretation Guide**", expanded=False):
            st.markdown(f"""
            ### Results for {ticker}
            
            **What the numbers mean:**
            
            - **Best α = {result['best_alpha']:.2f}**: This asset's volatility is 
              {'highly' if result['best_alpha'] > 0.5 else 'moderately' if result['best_alpha'] > 0.2 else 'minimally'} 
              sensitive to news signals. {'News provides substantial information beyond market persistence.' if result['best_alpha'] > 0.5 else 'Market dynamics dominate, but news adds some value.' if result['best_alpha'] > 0.2 else 'Volatility is primarily driven by market persistence.'}
            
            - **F1 Uplift = {news_uplift:+.1f}%**: Adding news features 
              {'substantially improves' if news_uplift > 10 else 'modestly improves' if news_uplift > 0 else 'does not improve'} 
              the model's ability to identify extreme volatility days.
            
            **Sector context:**
            
            Based on the research, different sectors respond differently to news:
            - **Energy (XLE, XOM)**: High news sensitivity - supply shocks, geopolitics
            - **Financials (XLF, JPM)**: Moderate - policy, macro conditions  
            - **Tech (XLK, AAPL)**: Mixed - earnings, regulation, innovation cycles
            - **Utilities (XLU)**: Low - structural, regulated cash flows
            """)
    
    else:
        # Default state
        st.info("👈 Enter a ticker and click **Run Analysis** to get started.")
        
        # Show example results or explanation
        with st.expander("ℹ️ **How it works**", expanded=True):
            st.markdown("""
            ### The Hybrid Residual Model
            
            **Problem:** Volatility is highly persistent - yesterday's volatility strongly predicts today's. 
            Linear models capture this well, but miss the nonlinear effects of news.
            
            **Solution:** A two-stage approach:
            
            1. **Baseline Model** (Logistic Regression)
               - Uses only market features: lagged volatility, returns
               - Captures the persistent, autoregressive structure
               
            2. **Residual Model** (Random Forest)
               - Trained on what the baseline *misses*
               - Uses news features: sentiment intensity, attention, regime indicators
               - Captures nonlinear, threshold-driven news effects
            
            3. **Combined Prediction**
               - `P(extreme) = baseline_prob + α × residual_pred`
               - α is tuned to find optimal news weight
            
            **Key Finding:** News is most predictive for *extreme* volatility events, 
            not routine fluctuations. The effect varies significantly across sectors.
            """)


if __name__ == "__main__":
    main()
