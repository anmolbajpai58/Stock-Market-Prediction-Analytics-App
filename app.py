
import os
from pathlib import Path
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Stock Prediction App", layout="wide", page_icon="📈")
BASE_DIR = Path.cwd()
CSV_PATH = Path(r"C:\Users\Lenovostore\Downloads\Final-50-stocks.csv")  # modify if needed
MODELS_DIR = BASE_DIR / "models"
TRAIN_SUMMARY = MODELS_DIR / "training_summary_for_app.csv"

# -----------------------
# Helpers & features (same logic you used)
# -----------------------
def safe_name(stock: str) -> str:
    return stock.replace(" ", "_").replace("/", "_").upper()

def add_features_price_series(df_price: pd.DataFrame, stock_col: str) -> pd.DataFrame:
    data = df_price[["DATE", stock_col]].copy().rename(columns={stock_col: "Close"})
    data["Return"] = data["Close"].pct_change()
    data["lag1"] = data["Return"].shift(1)
    data["lag2"] = data["Return"].shift(2)
    data["lag3"] = data["Return"].shift(3)
    data["SMA_5"] = data["Close"].rolling(5).mean()
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["EMA_5"] = data["Close"].ewm(span=5, adjust=False).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["Volatility_5"] = data["Return"].rolling(5).std()
    data["Volatility_10"] = data["Return"].rolling(10).std()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["STD_20"] = data["Close"].rolling(20).std()
    data["Upper_Band"] = data["SMA_20"] + 2 * data["STD_20"]
    data["Lower_Band"] = data["SMA_20"] - 2 * data["STD_20"]
    data["Target_Return_Next"] = data["Return"].shift(-1)
    data["Target_Dir_Next"] = (data["Target_Return_Next"] > 0).astype(int)
    data = data.dropna().reset_index(drop=True)
    return data

# -----------------------
# Plotly helpers
# -----------------------
PLOT_BG = "#071019"

def create_price_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['Close'], mode='lines', name='Close',
                             line=dict(color='#00E3FF', width=2)))
    if 'SMA_5' in df.columns:
        fig.add_trace(go.Scatter(x=df['DATE'], y=df['SMA_5'], mode='lines', name='SMA 5',
                                 line=dict(color='#8A2BE2', width=1.6, dash='dash')))
    if 'SMA_10' in df.columns:
        fig.add_trace(go.Scatter(x=df['DATE'], y=df['SMA_10'], mode='lines', name='SMA 10',
                                 line=dict(color='#6EE7B7', width=1.4, dash='dot')))
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='rgba(236,236,236,0.95)',
        margin=dict(l=40, r=12, t=10, b=36),
        legend=dict(orientation="v", y=0.98, x=0.99),
        xaxis=dict(showgrid=False, color='rgba(236,236,236,0.5)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)', color='rgba(236,236,236,0.6)')
    )
    fig.update_traces(hoverinfo='x+y')
    return fig

def create_rsi_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['RSI'], mode='lines', line=dict(color='#00E3FF')))
    fig.add_hline(y=70, line_dash='dash', line_color='#FF4B4B')
    fig.add_hline(y=30, line_dash='dash', line_color='#00FF7F')
    fig.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=30))
    return fig

def create_vol_fig(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['DATE'], y=df['Volatility_10'], fill='tozeroy', line=dict(color='#00FF7F')))
    fig.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, b=30))
    return fig

def gauge_donut(percent: float, label: str):
    v = float(max(0.0, min(100.0, percent)))
    color = '#00FF7F' if v >= 55 else ('#FF4B4B' if v <= 45 else '#00E3FF')
    fig = go.Figure(go.Pie(values=[v, 100-v], hole=0.62, marker=dict(colors=[color, 'rgba(255,255,255,0.05)']), textinfo='none'))
    fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), showlegend=False,
                      annotations=[dict(text=f"<b>{int(v)}%</b><br>{label}", x=0.5, y=0.5, showarrow=False,
                                        font=dict(color='rgba(236,236,236,0.95)', size=14))],
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

# -----------------------
# CSS styling (only visuals changed)
# -----------------------
st.markdown("""
    <style>
            
            /* Fix heading visibility */
.pred-label {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: rgba(236,236,236,0.92) !important;
}

/* Improve small gray texts */
.small {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: rgba(236,236,236,0.85) !important;
}

/* Fix title clipping */
.block-container {
    padding-top: 50px !important;
}

/* Direction label */
.direction-label {
    font-size: 26px;
    font-weight: 800;
}

    :root{
      --bg: #0D1116;
      --card: rgba(22,26,31,0.94);
      --accent: #00E3FF;
      --accent2: #8A2BE2;
      --success: #00FF7F;
      --danger: #FF4B4B;
      --text: rgba(236,236,236,0.95);
      --muted: rgba(236,236,236,0.55);
    }
    .block-container { padding-top: 36px; padding-left:36px; padding-right:36px; }
    .stApp { background: var(--bg); color:var(--text); }
    .app-header { font-size:34px; font-weight:800; color:var(--text); margin-bottom:6px; }
    .app-sub { color:var(--muted); font-size:14px; margin-bottom:12px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.006)); border-radius:14px; padding:16px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.03); }
    .label-strong { font-size:15px; font-weight:800; color:var(--text); }
    .label-muted { font-size:13px; color:var(--muted); }
    .value-large { font-size:40px; font-weight:900; color:var(--text); }
    .value-medium { font-size:28px; font-weight:800; color:var(--text); }
    .small { font-size:13px; color:var(--muted); }
    .fi-item { display:flex; align-items:center; gap:12px; margin:10px 0; }
    .fi-name { width:96px; color:var(--muted); font-size:14px; }
    .fi-bar { height:12px; border-radius:10px; background:rgba(255,255,255,0.03); overflow:hidden; flex:1; }
    .fi-bar .fill { height:100%; background: linear-gradient(90deg,var(--accent),var(--accent2)); box-shadow: 0 8px 20px rgba(10,20,60,0.25); border-radius:10px; }
    .predict-btn { background: linear-gradient(90deg,#073b6b,#00a7ff); color:white; padding:10px 16px; border-radius:10px; border:none; font-weight:700; }
    .donut-wrap { display:flex; align-items:center; justify-content:center; padding-top:10px; }
    .stat-title { font-size:15px; color:var(--muted); font-weight:700; }
    .muted-strong { color: rgba(236,236,236,0.88); font-size:14px; }
    .footer { text-align:center; color:rgba(236,236,236,0.32); margin-top:28px; padding-bottom:20px; }
            
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# Load CSV
# -----------------------
if not CSV_PATH.exists():
    st.error(f"CSV not found at {CSV_PATH}. Place Final-50-stocks.csv at this path or update CSV_PATH.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_csv(path):
    df = pd.read_csv(path)
    if "DATE" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    return df

df_all = load_csv(CSV_PATH)
stock_cols = [c for c in df_all.columns if c.upper() != "DATE"]
if len(stock_cols) == 0:
    st.error("No stock columns found in CSV.")
    st.stop()

# -----------------------
# Header
# -----------------------
col_h1, col_h2 = st.columns([8,2])
with col_h1:
    st.markdown('<div class="app-header">STOCK PREDICTION APP</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-sub">AI-Driven Forecasting • Market Insights • Smart Decisions</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<div style="text-align:right;"><button class="predict-btn">Download Results</button></div>', unsafe_allow_html=True)

st.write("")

# -----------------------
# Layout columns (left / center / right)
# -----------------------
col_left, col_center, col_right = st.columns([1,2.4,1])

# LEFT: Inputs (unchanged behaviour)
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label-strong">Select Stock</div>', unsafe_allow_html=True)
    default_idx = 0
    if "RELIANCE" in stock_cols:
        default_idx = stock_cols.index("RELIANCE")
    stock = st.selectbox("", options=stock_cols, index=default_idx)
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict", key="predict_btn_left")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.03)'>", unsafe_allow_html=True)
    st.markdown('<div class="small" style="font-weight:700; margin-bottom:6px;">Selected Stock</div>', unsafe_allow_html=True)
    last_close_val = df_all[stock].dropna().iloc[-1] if not df_all[stock].dropna().empty else float("nan")
    st.markdown(f"<div class='small'>Last Close <strong style='float:right'>{last_close_val:.2f}</strong></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small' style='margin-top:6px;'>Data Range: {df_all['DATE'].min().date()} → {df_all['DATE'].max().date()}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# CENTER: Prediction (kept in center exactly) and Charts. Market Sentiment donut will be placed in the RIGHT column next to these.
with col_center:
    # feature generation for stock
    try:
        df_stock_full = df_all[["DATE", stock]].rename(columns={stock: stock}).dropna().reset_index(drop=True)
        fe_all = add_features_price_series(df_stock_full, stock)
    except Exception:
        st.error("Feature generation failed for selected stock.")
        st.stop()

    max_date = fe_all["DATE"].max()
    one_year_ago = max_date - pd.DateOffset(years=1)
    last_1y = fe_all[fe_all["DATE"] >= one_year_ago].copy()

    SAFE = safe_name(stock)
    reg_path = MODELS_DIR / f"{SAFE}_reg.pkl"
    clf_path = MODELS_DIR / f"{SAFE}_clf.pkl"
    scaler_path = MODELS_DIR / f"{SAFE}_scaler.pkl"
    feat_path = MODELS_DIR / f"{SAFE}_features.pkl"

    # load artifacts (quiet)
    reg_model = clf_model = scaler = feature_cols = None
    try:
        if reg_path.exists(): reg_model = joblib.load(reg_path)
    except Exception:
        reg_model = None
    try:
        if clf_path.exists(): clf_model = joblib.load(clf_path)
    except Exception:
        clf_model = None
    try:
        if scaler_path.exists(): scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None
    try:
        if feat_path.exists(): feature_cols = joblib.load(feat_path)
    except Exception:
        feature_cols = None

    if feature_cols is None:
        feature_cols = [
            'Return', 'lag1', 'lag2', 'lag3',
            'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10',
            'Volatility_5', 'Volatility_10',
            'RSI', 'SMA_20', 'STD_20', 'Upper_Band', 'Lower_Band'
        ]

    X_new = last_1y.iloc[-1:][feature_cols].copy()
    X_new = X_new.replace([np.inf, -np.inf], np.nan).fillna(0)

    reg_pred = None
    clf_pred = None
    confidence = 0.0
    MODEL_METRICS = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "Accuracy": float("nan")}

    # regression pred
    if reg_model is not None:
        try:
            reg_pred = float(reg_model.predict(X_new)[0])
        except Exception:
            try:
                reg_pred = float(reg_model.predict(X_new.values)[0])
            except Exception:
                reg_pred = None

    # classification pred + confidence
    if clf_model is not None:
        try:
            Xs = scaler.transform(X_new) if scaler is not None else X_new.values
            clf_pred = int(clf_model.predict(Xs)[0])
            if hasattr(clf_model, "predict_proba"):
                proba = clf_model.predict_proba(Xs)[0]
                confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                confidence = 0.5
        except Exception:
            clf_pred = None
            confidence = 0.0

    display_return = f"{reg_pred*100:+.2f}%" if reg_pred is not None else "N/A"
    direction = "UP" if clf_pred == 1 else ("DOWN" if clf_pred == 0 else "N/A")
    conf_pct = int(round(confidence * 100))
    if conf_pct >= 75:
        signal_strength = "Strong Buy" if direction == "UP" else ("Strong Sell" if direction == "DOWN" else "Hold")
    elif conf_pct >= 55:
        signal_strength = "Buy" if direction == "UP" else ("Sell" if direction == "DOWN" else "Hold")
    else:
        signal_strength = "Hold"

    # try to read training summary metrics (if present)
    if TRAIN_SUMMARY.exists():
        try:
            s = pd.read_csv(TRAIN_SUMMARY)
            row = s[s['stock'].str.upper() == stock.upper()]
            if not row.empty:
                MODEL_METRICS["RMSE"] = float(row.iloc[0].get('rf_rmse', math.nan))
                MODEL_METRICS["MAE"] = float(row.iloc[0].get('rf_mae', math.nan))
                MODEL_METRICS["R2"]  = float(row.iloc[0].get('rf_r2', math.nan))
                MODEL_METRICS["Accuracy"] = float(row.iloc[0].get('log_acc', math.nan))
        except Exception:
            pass

    # --- PREDICTION CARD (kept in center column exactly) ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # arrange stats vertically (as requested — same center position)
    st.markdown('<div style="display:flex;gap:18px;align-items:flex-start;">', unsafe_allow_html=True)
    # left area (labels + big values stacked)
    st.markdown('<div style="flex:1;">', unsafe_allow_html=True)
    st.markdown('<div class="label-strong">Predicted Return</div>', unsafe_allow_html=True)
    ret_color = "#00FF7F" if (isinstance(reg_pred, (int, float)) and reg_pred >= 0) else "#FF4B4B"
    st.markdown(f'<div class="value-large" style="color:{ret_color}; margin-top:8px;">{display_return}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="label-strong">Confidence</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="value-medium" style="color:var(--accent); margin-top:8px;">{conf_pct}%</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="label-strong">Signal Strength</div>', unsafe_allow_html=True)
    sig_col = "#00FF7F" if "Buy" in signal_strength else ("#FF4B4B" if "Sell" in signal_strength else "rgba(236,236,236,0.95)")
    st.markdown(f'<div class="value-medium" style="color:{sig_col}; margin-top:8px;">{signal_strength}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # right area remains blank visually (we will NOT disturb it) - but under the card we'll show model line & confidence bar
    st.markdown('</div>', unsafe_allow_html=True)

    # model line & gradient bar
    model_line = f"Model: {reg_model.__class__.__name__ if reg_model is not None else 'N/A'} (reg) + {clf_model.__class__.__name__ if clf_model is not None else 'N/A'} (clf)"
    st.markdown(f"<div style='margin-top:10px;color:var(--muted); font-size:13px;'>{model_line}</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="margin-top:10px;">
          <div style='height:12px;background:rgba(255,255,255,0.04);border-radius:8px;'>
            <div style='width:{conf_pct}%;height:100%;background:linear-gradient(90deg,#00E3FF,#8A2BE2);border-radius:8px;'></div>
          </div>
          <div style='margin-top:8px;color:var(--muted);font-size:13px;'>{conf_pct}% for {direction} &nbsp; | &nbsp; Last updated: {fe_all['DATE'].iloc[-1].date()}</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts card (unchanged)
    st.markdown('<div style="margin-top:12px;" class="card">', unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700; margin-bottom:6px;">Charts (1 year)</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Price", "RSI", "Volatility"])
    with tabs[0]:
        try:
            fig_price = create_price_fig(last_1y)
            st.plotly_chart(fig_price, use_container_width=True, config={'displayModeBar': False})
        except Exception:
            st.error("Could not render price chart.")
    with tabs[1]:
        try:
            fig_rsi = create_rsi_fig(last_1y)
            st.plotly_chart(fig_rsi, use_container_width=True, config={'displayModeBar': False})
        except Exception:
            pass
    with tabs[2]:
        try:
            fig_vol = create_vol_fig(last_1y)
            st.plotly_chart(fig_vol, use_container_width=True, config={'displayModeBar': False})
        except Exception:
            pass
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT: place Market Sentiment donut here (in the blank to the right of the prediction card) and Model Performance + Feature Importance
with col_right:
    # Market Sentiment donut in the top empty area (this is exactly the blank space next to predictions)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label-strong">Market Sentiment</div>', unsafe_allow_html=True)
    # Simple sentiment calculation: recent returns mean -> map to 0..100
    recent_mean = float(last_1y["Return"].tail(10).mean()) if not last_1y["Return"].tail(10).empty else 0.0
    sentiment_pct = int(max(0, min(100, 50 + recent_mean * 1000)))
    sentiment_label = "Bullish" if sentiment_pct >= 55 else ("Bearish" if sentiment_pct <= 45 else "Neutral")
    fig_donut = gauge_donut(sentiment_pct, sentiment_label)
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Model Performance card (more visible numbers)
    st.markdown('<div style="margin-top:12px;" class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label-strong">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;justify-content:space-between; margin-top:8px;'><div class='small'>RMSE</div><div class='muted-strong'>{MODEL_METRICS['RMSE'] if not math.isnan(MODEL_METRICS['RMSE']) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;justify-content:space-between; margin-top:6px;'><div class='small'>MAE</div><div class='muted-strong'>{MODEL_METRICS['MAE'] if not math.isnan(MODEL_METRICS['MAE']) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;justify-content:space-between; margin-top:6px;'><div class='small'>R²</div><div class='muted-strong'>{MODEL_METRICS['R2'] if not math.isnan(MODEL_METRICS['R2']) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;justify-content:space-between; margin-top:6px;'><div class='small'>Accuracy</div><div class='muted-strong'>{MODEL_METRICS['Accuracy'] if not math.isnan(MODEL_METRICS['Accuracy']) else 'N/A'}</div></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color: rgba(255,255,255,0.03); margin-top:10px;'>", unsafe_allow_html=True)

    # Feature importance (keeps gradient bars)
    st.markdown('<div class="label-strong" style="margin-top:6px;">Feature Importance</div>', unsafe_allow_html=True)
    try:
        if reg_model is not None and hasattr(reg_model, "feature_importances_"):
            importances = reg_model.feature_importances_
            fi = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
        else:
            fi = pd.DataFrame({"feature": feature_cols, "importance": np.linspace(1, 0.1, len(feature_cols))})
        top = fi.head(6)
        max_val = top["importance"].max() if not top.empty else 1.0
        for _, r in top.iterrows():
            feat = r["feature"]
            val = float(r["importance"])
            pct = int((val / max_val) * 100) if max_val > 0 else 0
            st.markdown(f"""
                <div class="fi-item">
                  <div class="fi-name">{feat}</div>
                  <div class="fi-bar"><div class="fill" style="width:{pct}%;"></div></div>
                </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.write("No feature importance available.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Stock Prediction App • 2025</div>', unsafe_allow_html=True)
