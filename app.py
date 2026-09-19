
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="UAC Predictive Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    padding: 14px;
    border-radius: 12px;
    background: rgba(128,128,128,.05);
}
h1, h2, h3 {letter-spacing: -0.02em;}
.small-note {font-size: 0.85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
EXPECTED = [
    "Date",
    "Apprehended",
    "CBP_Custody",
    "Transfers",
    "HHS_Care",
    "Discharged",
]

ALIASES = {
    "Date": ["Date", "Reporting Date"],
    "Apprehended": [
        "Children apprehended and placed in CBP custody",
        "Children apprehended",
        "Apprehended",
    ],
    "CBP_Custody": [
        "Children in CBP custody",
        "CBP custody",
        "CBP_Custody",
    ],
    "Transfers": [
        "Children transferred out of CBP custody",
        "Children transferred out of CBP custody ",
        "Transfers",
        "Transferred",
    ],
    "HHS_Care": [
        "Children in HHS Care",
        "Children in HHS care",
        "HHS Care",
        "HHS_Care",
    ],
    "Discharged": [
        "Children discharged from HHS Care",
        "Children discharged from HHS care",
        "Discharged",
    ],
}

def standardize_columns(df):
    rename = {}
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for target, aliases in ALIASES.items():
        for a in aliases:
            key = a.strip().lower()
            if key in normalized:
                rename[normalized[key]] = target
                break
    out = df.rename(columns=rename).copy()
    missing = [c for c in EXPECTED if c not in out.columns]
    return out, missing

def load_data(uploaded):
    if uploaded is None:
        candidates = list(Path(".").glob("*.xlsx")) + list(Path(".").glob("*.csv"))
        if not candidates:
            return None, ["Upload the actual Excel/CSV dataset."]
        path = candidates[0]
        df = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    else:
        df = (
            pd.read_excel(uploaded)
            if uploaded.name.lower().endswith((".xlsx", ".xls"))
            else pd.read_csv(uploaded)
        )

    df, missing = standardize_columns(df)
    if missing:
        return None, [f"Missing columns: {', '.join(missing)}"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    for c in EXPECTED[1:]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    # One row per date; duplicates are averaged rather than silently duplicated.
    df = df.groupby("Date", as_index=False)[EXPECTED[1:]].mean()
    df = df.sort_values("Date").set_index("Date")

    # Preserve original observation availability. Missing days are not assumed to be zero.
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_dates = full_idx.difference(df.index)

    # Interpolate only numeric observations across short gaps, while retaining a flag.
    df = df.reindex(full_idx)
    df.index.name = "Date"
    df["Was_Missing_Date"] = df["HHS_Care"].isna().astype(int)

    numeric = EXPECTED[1:]
    df[numeric] = df[numeric].interpolate(limit_direction="both")
    return df, missing_dates

def add_features(df):
    x = df.copy()
    x["Net_Pressure"] = x["Transfers"] - x["Discharged"]

    for lag in [1, 2, 7, 14]:
        x[f"HHS_Lag_{lag}"] = x["HHS_Care"].shift(lag)

    for w in [7, 14]:
        x[f"HHS_Roll_Mean_{w}"] = x["HHS_Care"].rolling(w).mean()
        x[f"HHS_Roll_Std_{w}"] = x["HHS_Care"].rolling(w).std()

    for lag in [1, 7]:
        x[f"Transfers_Lag_{lag}"] = x["Transfers"].shift(lag)
        x[f"Discharged_Lag_{lag}"] = x["Discharged"].shift(lag)

    x["DayOfWeek"] = x.index.dayofweek
    x["Month"] = x.index.month
    x["DayOfYear"] = x.index.dayofyear
    x["WeekOfYear"] = x.index.isocalendar().week.astype(int)
    x["IsWeekend"] = (x["DayOfWeek"] >= 5).astype(int)

    return x

FEATURES = [
    "HHS_Lag_1", "HHS_Lag_2", "HHS_Lag_7", "HHS_Lag_14",
    "HHS_Roll_Mean_7", "HHS_Roll_Mean_14",
    "HHS_Roll_Std_7", "HHS_Roll_Std_14",
    "Transfers", "Discharged", "Net_Pressure",
    "Transfers_Lag_1", "Transfers_Lag_7",
    "Discharged_Lag_1", "Discharged_Lag_7",
    "DayOfWeek", "Month", "DayOfYear", "WeekOfYear", "IsWeekend"
]

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))

def mape(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)

def evaluate_models(feat_df, test_days):
    clean = feat_df.dropna(subset=FEATURES + ["HHS_Care"]).copy()
    if len(clean) < 40:
        return None, None, "Not enough complete observations for reliable model evaluation."

    test_days = min(test_days, max(7, len(clean) // 5))
    split = len(clean) - test_days
    train, test = clean.iloc[:split], clean.iloc[split:]

    X_train, y_train = train[FEATURES], train["HHS_Care"]
    X_test, y_test = test[FEATURES], test["HHS_Care"]

    # Naive
    naive = test["HHS_Lag_1"].values

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    # SARIMA
    sarima_pred = None
    try:
        sar = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        sarima_pred = np.asarray(sar.forecast(steps=len(test)))
    except Exception:
        sarima_pred = np.full(len(test), np.nan)

    rows = [
        ["Naive", mean_absolute_error(y_test, naive), rmse(y_test, naive), mape(y_test, naive)],
        ["Gradient Boosting", mean_absolute_error(y_test, gb_pred), rmse(y_test, gb_pred), mape(y_test, gb_pred)],
    ]
    if np.isfinite(sarima_pred).all():
        rows.append(["SARIMA", mean_absolute_error(y_test, sarima_pred), rmse(y_test, sarima_pred), mape(y_test, sarima_pred)])

    results = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE (%)"]).sort_values("MAE")
    preds = {
        "test_index": test.index,
        "actual": y_test.values,
        "Naive": naive,
        "Gradient Boosting": gb_pred,
        "SARIMA": sarima_pred,
    }
    return results, preds, None

def make_future_features(history, next_date, transfer_assumption, discharge_assumption):
    h = history.copy()

    def val_lag(n):
        return float(h["HHS_Care"].iloc[-n]) if len(h) >= n else float(h["HHS_Care"].mean())

    row = {
        "HHS_Lag_1": val_lag(1),
        "HHS_Lag_2": val_lag(2),
        "HHS_Lag_7": val_lag(7),
        "HHS_Lag_14": val_lag(14),
        "HHS_Roll_Mean_7": float(h["HHS_Care"].tail(7).mean()),
        "HHS_Roll_Mean_14": float(h["HHS_Care"].tail(14).mean()),
        "HHS_Roll_Std_7": float(h["HHS_Care"].tail(7).std() or 0),
        "HHS_Roll_Std_14": float(h["HHS_Care"].tail(14).std() or 0),
        "Transfers": transfer_assumption,
        "Discharged": discharge_assumption,
        "Net_Pressure": transfer_assumption - discharge_assumption,
        "Transfers_Lag_1": float(h["Transfers"].iloc[-1]),
        "Transfers_Lag_7": float(h["Transfers"].iloc[-7]) if len(h) >= 7 else float(h["Transfers"].mean()),
        "Discharged_Lag_1": float(h["Discharged"].iloc[-1]),
        "Discharged_Lag_7": float(h["Discharged"].iloc[-7]) if len(h) >= 7 else float(h["Discharged"].mean()),
        "DayOfWeek": next_date.dayofweek,
        "Month": next_date.month,
        "DayOfYear": next_date.dayofyear,
        "WeekOfYear": int(next_date.isocalendar().week),
        "IsWeekend": int(next_date.dayofweek >= 5),
    }
    return pd.DataFrame([row], index=[next_date])[FEATURES]

def recursive_gb_forecast(base_df, model, days, scenario_mult=1.0):
    history = base_df[["HHS_Care", "Transfers", "Discharged"]].copy()
    out = []

    base_transfer = float(history["Transfers"].tail(7).mean())
    base_discharge = float(history["Discharged"].tail(7).mean())

    for _ in range(days):
        next_date = history.index.max() + pd.Timedelta(days=1)
        tr = base_transfer * scenario_mult
        dc = base_discharge
        Xf = make_future_features(history, next_date, tr, dc)
        pred = float(model.predict(Xf)[0])
        pred = max(0.0, pred)

        out.append([next_date, pred, tr, dc, tr - dc])
        history.loc[next_date, "HHS_Care"] = pred
        history.loc[next_date, "Transfers"] = tr
        history.loc[next_date, "Discharged"] = dc

    return pd.DataFrame(out, columns=["Date", "Forecast", "Transfers_Assumed", "Discharges_Assumed", "Net_Pressure"]).set_index("Date")

def quantile_intervals(feat_df, days, scenario_mult=1.0):
    clean = feat_df.dropna(subset=FEATURES + ["HHS_Care"]).copy()
    models = {}
    for q in [0.1, 0.5, 0.9]:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        )
        model.fit(clean[FEATURES], clean["HHS_Care"])
        models[q] = model

    history = clean[["HHS_Care", "Transfers", "Discharged"]].copy()
    rows = []
    base_transfer = float(history["Transfers"].tail(7).mean()) * scenario_mult
    base_discharge = float(history["Discharged"].tail(7).mean())

    for _ in range(days):
        d = history.index.max() + pd.Timedelta(days=1)
        Xf = make_future_features(history, d, base_transfer, base_discharge)
        lo = max(0, float(models[0.1].predict(Xf)[0]))
        mid = max(0, float(models[0.5].predict(Xf)[0]))
        hi = max(lo, float(models[0.9].predict(Xf)[0]))
        rows.append([d, lo, mid, hi])
        history.loc[d, "HHS_Care"] = mid
        history.loc[d, "Transfers"] = base_transfer
        history.loc[d, "Discharged"] = base_discharge

    return pd.DataFrame(rows, columns=["Date", "Lower", "Forecast", "Upper"]).set_index("Date")

# ---------- UI ----------
st.title("UAC Predictive Intelligence Dashboard")
st.caption("Predictive Forecasting of Care Load & Placement Demand")

uploaded = st.sidebar.file_uploader(
    "Upload UAC dataset", type=["xlsx", "xls", "csv"]
)

df, errors = load_data(uploaded)
if df is None:
    st.warning("Upload the actual Excel/CSV dataset to activate the dashboard.")
    for e in errors:
        st.info(e)
    st.stop()

feat = add_features(df)

# Sidebar controls
st.sidebar.header("Forecast Controls")
horizon = st.sidebar.selectbox("Forecast horizon", [7, 14, 30], index=0)
scenario = st.sidebar.selectbox(
    "Incoming transfer scenario",
    ["Normal", "High Intake (+20%)", "Stress (+40%)"]
)
scenario_mult = {"Normal": 1.0, "High Intake (+20%)": 1.2, "Stress (+40%)": 1.4}[scenario]

capacity = st.sidebar.number_input(
    "Planning capacity", min_value=1, value=int(max(2500, df["HHS_Care"].max() * 1.05))
)

# KPI row
current = float(df["HHS_Care"].iloc[-1])
latest_transfer = float(df["Transfers"].iloc[-1])
latest_discharge = float(df["Discharged"].iloc[-1])
net = latest_transfer - latest_discharge

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current HHS Care", f"{current:,.0f}")
c2.metric("Latest Transfers", f"{latest_transfer:,.0f}")
c3.metric("Latest Discharges", f"{latest_discharge:,.0f}")
c4.metric("Net Pressure", f"{net:+,.0f}")

if len(errors) > 0:
    st.info(f"Data quality note: {len(errors)} issue(s) detected.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Care Load Forecast",
    "🏠 Placement / Discharge Demand",
    "🧠 Model Comparison",
    "🔎 Data Quality & Insights",
])

with tab1:
    st.subheader("Future HHS Care Load")
    st.caption(f"Scenario: {scenario} | Horizon: {horizon} days")

    clean = feat.dropna(subset=FEATURES + ["HHS_Care"]).copy()
    model = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    model.fit(clean[FEATURES], clean["HHS_Care"])

    forecast = recursive_gb_forecast(clean, model, horizon, scenario_mult)
    interval = quantile_intervals(feat, horizon, scenario_mult)

    fig = go.Figure()
    hist = df.tail(min(120, len(df)))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["HHS_Care"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Forecast"], mode="lines+markers", name="Forecast"))
    fig.add_trace(go.Scatter(
        x=list(interval.index) + list(interval.index[::-1]),
        y=list(interval["Upper"]) + list(interval["Lower"][::-1]),
        fill="toself", line=dict(color="rgba(0,0,0,0)"),
        name="Forecast uncertainty", hoverinfo="skip"
    ))
    fig.add_hline(y=capacity, line_dash="dash", annotation_text="Planning capacity")
    fig.update_layout(
        height=500, margin=dict(l=20,r=20,t=30,b=20),
        xaxis_title="Date", yaxis_title="Children",
        hovermode="x unified", legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

    peak = float(forecast["Forecast"].max())
    end_value = float(forecast["Forecast"].iloc[-1])
    breach_days = int((forecast["Forecast"] > capacity).sum())
    util = peak / capacity * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("End-of-horizon forecast", f"{end_value:,.0f}")
    k2.metric("Peak forecast", f"{peak:,.0f}")
    k3.metric("Peak capacity utilization", f"{util:.1f}%")

    if breach_days:
        st.error(f"Capacity warning: forecast exceeds planning capacity on {breach_days} day(s).")
    elif util >= 90:
        st.warning("Capacity watch: forecast reaches at least 90% of planning capacity.")
    else:
        st.success("Capacity outlook: forecast remains below the 90% planning threshold.")

    st.dataframe(forecast.reset_index().rename(columns={
        "Date":"Date",
        "Forecast":"Forecast HHS Care",
        "Transfers_Assumed":"Assumed Transfers",
        "Discharges_Assumed":"Assumed Discharges",
        "Net_Pressure":"Assumed Net Pressure",
    }), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Discharge / Placement Demand")
    st.caption("Short-term demand is estimated from historical discharge patterns and recent flow conditions.")

    discharge_roll = df["Discharged"].rolling(7).mean().iloc[-1]
    expected_total = float(df["Discharged"].tail(7).mean() * horizon)

    a, b, c = st.columns(3)
    a.metric("Recent 7-day avg. discharges", f"{discharge_roll:,.1f}/day")
    b.metric(f"Estimated {horizon}-day demand", f"{expected_total:,.0f}")
    c.metric("Current net pressure", f"{net:+,.0f}")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df.tail(min(60, len(df))).index,
        y=df.tail(min(60, len(df)))["Discharged"],
        name="Daily discharges"
    ))
    fig2.add_trace(go.Scatter(
        x=df.tail(min(60, len(df))).index,
        y=df.tail(min(60, len(df)))["Discharged"].rolling(7).mean(),
        mode="lines", name="7-day average"
    ))
    fig2.update_layout(height=430, margin=dict(l=20,r=20,t=30,b=20),
                       xaxis_title="Date", yaxis_title="Children")
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "Planning interpretation: sustained transfers above discharges increase care-load pressure; "
        "sustained discharges above transfers reduce pressure. This panel is a planning estimate, not a guarantee."
    )

with tab3:
    st.subheader("Statistical vs Machine-Learning Forecasting")
    test_days = st.slider("Hold-out evaluation window (days)", 7, 30, 14)
    results, preds, err = evaluate_models(feat, test_days)

    if err:
        st.warning(err)
    else:
        st.dataframe(results.round(3), use_container_width=True, hide_index=True)
        best = results.iloc[0]["Model"]
        st.success(f"Best model on the selected hold-out window by MAE: {best}")

        fig3 = go.Figure()
        idx = preds["test_index"]
        fig3.add_trace(go.Scatter(x=idx, y=preds["actual"], mode="lines", name="Actual"))
        for name in ["Naive", "SARIMA", "Gradient Boosting"]:
            p = preds.get(name)
            if p is not None and np.isfinite(p).any():
                fig3.add_trace(go.Scatter(x=idx, y=p, mode="lines", name=name))
        fig3.update_layout(height=480, margin=dict(l=20,r=20,t=30,b=20),
                           xaxis_title="Date", yaxis_title="Children", hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("Data Quality & Operational Insights")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_dates = full_idx.difference(df.index[df["Was_Missing_Date"] == 0])
    missing_cells = int(df[EXPECTED[1:]].isna().sum().sum())

    q1, q2, q3 = st.columns(3)
    q1.metric("Records after daily reindexing", f"{len(df):,}")
    q2.metric("Missing calendar dates", f"{len(missing_dates):,}")
    q3.metric("Missing numeric cells after interpolation", f"{missing_cells:,}")

    st.markdown("### Recent operating picture")
    recent = df.tail(14)[["Apprehended","CBP_Custody","Transfers","HHS_Care","Discharged"]].copy()
    recent["Net_Pressure"] = recent["Transfers"] - recent["Discharged"]
    st.dataframe(recent.round(1), use_container_width=True)

    st.markdown("### Interpretation")
    avg_net = float(df["Transfers"].tail(14).mean() - df["Discharged"].tail(14).mean())
    if avg_net > 0:
        st.warning(f"Average net pressure over the last 14 observations is +{avg_net:.1f} children/day.")
    else:
        st.success(f"Average net pressure over the last 14 observations is {avg_net:.1f} children/day.")

st.markdown("---")
st.markdown(
    '<div class="small-note">Forecasts are decision-support estimates. Future transfers/discharges are uncertain; '
    'scenario assumptions should be reviewed by operational planners.</div>',
    unsafe_allow_html=True
)
