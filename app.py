import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import re
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

# ---------- Config ----------
EXPECTED = ["Date", "Apprehended", "CBP_Custody", "Transfers", "HHS_Care", "Discharged"]

ALIASES = {
    "Date": ["Date", "Reporting Date"],
    "Apprehended": ["Children apprehended and placed in CBP custody", "Children apprehended", "Apprehended"],
    "CBP_Custody": ["Children in CBP custody", "CBP custody", "CBP_Custody"],
    "Transfers": ["Children transferred out of CBP custody", "Children transferred out of CBP custody ", "Transfers", "Transferred"],
    "HHS_Care": ["Children in HHS Care", "Children in HHS care", "HHS Care", "HHS_Care"],
    "Discharged": ["Children discharged from HHS Care", "Children discharged from HHS care", "Discharged"],
}

FEATURES = [
    "HHS_Lag_1", "HHS_Lag_2", "HHS_Lag_7", "HHS_Lag_14",
    "HHS_Roll_Mean_7", "HHS_Roll_Mean_14",
    "HHS_Roll_Std_7", "HHS_Roll_Std_14",
    "Transfers", "Discharged", "Net_Pressure",
    "Transfers_Lag_1", "Transfers_Lag_7",
    "Discharged_Lag_1", "Discharged_Lag_7",
    "DayOfWeek", "Month", "DayOfYear", "WeekOfYear", "IsWeekend",
]

GB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)


# ---------- Data loading ----------
def _clean_header(text):
    """Lowercase, trim, and strip trailing footnote marks (*, †, numbers) and extra spaces."""
    text = str(text).strip().lower()
    text = re.sub(r"[\*\u2020\u2021#\d\s]+$", "", text)  # strip trailing *, †, ‡, #, digits, spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def standardize_columns(df):
    normalized = {_clean_header(c): c for c in df.columns}
    rename = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            key = _clean_header(alias)
            if key in normalized:
                rename[normalized[key]] = target
                break
        else:
            # Fallback: substring match in case of extra wording/footnotes
            for norm_key, orig_col in normalized.items():
                if any(_clean_header(a) in norm_key or norm_key in _clean_header(a) for a in aliases):
                    rename[orig_col] = target
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
        is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))
        df = pd.read_excel(uploaded) if is_excel else pd.read_csv(uploaded)

    df, missing = standardize_columns(df)
    if missing:
        return None, [
            f"Missing columns: {', '.join(missing)}",
            f"Columns detected in your file: {', '.join(str(c) for c in df.columns)}",
        ]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    numeric_cols = EXPECTED[1:]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    # One row per date; duplicates are averaged rather than silently duplicated.
    df = df.groupby("Date", as_index=False)[numeric_cols].mean()
    df = df.sort_values("Date").set_index("Date")

    # Preserve original observation availability. Missing days are not assumed to be zero.
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing_dates = full_idx.difference(df.index)

    df = df.reindex(full_idx)
    df.index.name = "Date"
    df["Was_Missing_Date"] = df["HHS_Care"].isna().astype(int)
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
    return df, missing_dates


# ---------- Feature engineering ----------
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


# ---------- Metrics ----------
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mask = np.abs(y) > 1e-9
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def score_row(name, y_true, y_pred):
    return [name, mean_absolute_error(y_true, y_pred), rmse(y_true, y_pred), mape(y_true, y_pred)]


def evaluate_models(feat_df, test_days):
    clean = feat_df.dropna(subset=FEATURES + ["HHS_Care"]).copy()
    if len(clean) < 40:
        return None, None, "Not enough complete observations for reliable model evaluation."

    test_days = min(test_days, max(7, len(clean) // 5))
    split = len(clean) - test_days
    train, test = clean.iloc[:split], clean.iloc[split:]
    X_train, y_train = train[FEATURES], train["HHS_Care"]
    X_test, y_test = test[FEATURES], test["HHS_Care"]

    naive = test["HHS_Lag_1"].values

    gb = GradientBoostingRegressor(**GB_PARAMS)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    try:
        sar = SARIMAX(
            y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        sarima_pred = np.asarray(sar.forecast(steps=len(test)))
    except Exception:
        sarima_pred = np.full(len(test), np.nan)

    rows = [score_row("Naive", y_test, naive), score_row("Gradient Boosting", y_test, gb_pred)]
    if np.isfinite(sarima_pred).all():
        rows.append(score_row("SARIMA", y_test, sarima_pred))

    results = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE (%)"]).sort_values("MAE")
    preds = {
        "test_index": test.index, "actual": y_test.values,
        "Naive": naive, "Gradient Boosting": gb_pred, "SARIMA": sarima_pred,
    }
    return results, preds, None


# ---------- Forecasting ----------
def make_future_features(history, next_date, transfer_assumption, discharge_assumption):
    def lag(n):
        return float(history["HHS_Care"].iloc[-n]) if len(history) >= n else float(history["HHS_Care"].mean())

    row = {
        "HHS_Lag_1": lag(1), "HHS_Lag_2": lag(2), "HHS_Lag_7": lag(7), "HHS_Lag_14": lag(14),
        "HHS_Roll_Mean_7": float(history["HHS_Care"].tail(7).mean()),
        "HHS_Roll_Mean_14": float(history["HHS_Care"].tail(14).mean()),
        "HHS_Roll_Std_7": float(history["HHS_Care"].tail(7).std() or 0),
        "HHS_Roll_Std_14": float(history["HHS_Care"].tail(14).std() or 0),
        "Transfers": transfer_assumption,
        "Discharged": discharge_assumption,
        "Net_Pressure": transfer_assumption - discharge_assumption,
        "Transfers_Lag_1": float(history["Transfers"].iloc[-1]),
        "Transfers_Lag_7": float(history["Transfers"].iloc[-7]) if len(history) >= 7 else float(history["Transfers"].mean()),
        "Discharged_Lag_1": float(history["Discharged"].iloc[-1]),
        "Discharged_Lag_7": float(history["Discharged"].iloc[-7]) if len(history) >= 7 else float(history["Discharged"].mean()),
        "DayOfWeek": next_date.dayofweek,
        "Month": next_date.month,
        "DayOfYear": next_date.dayofyear,
        "WeekOfYear": int(next_date.isocalendar().week),
        "IsWeekend": int(next_date.dayofweek >= 5),
    }
    return pd.DataFrame([row], index=[next_date])[FEATURES]


def _step_forward(history, transfer, discharge, predict_fn):
    """Advance history by one day using predict_fn(features_row) -> value."""
    next_date = history.index.max() + pd.Timedelta(days=1)
    Xf = make_future_features(history, next_date, transfer, discharge)
    value = max(0.0, predict_fn(Xf))
    history.loc[next_date] = {"HHS_Care": value, "Transfers": transfer, "Discharged": discharge}
    return next_date, value


def recursive_gb_forecast(base_df, model, days, scenario_mult=1.0):
    history = base_df[["HHS_Care", "Transfers", "Discharged"]].copy()
    base_transfer = float(history["Transfers"].tail(7).mean())
    base_discharge = float(history["Discharged"].tail(7).mean())
    tr = base_transfer * scenario_mult

    rows = []
    for _ in range(days):
        next_date, pred = _step_forward(history, tr, base_discharge, lambda X: float(model.predict(X)[0]))
        rows.append([next_date, pred, tr, base_discharge, tr - base_discharge])

    return pd.DataFrame(
        rows, columns=["Date", "Forecast", "Transfers_Assumed", "Discharges_Assumed", "Net_Pressure"]
    ).set_index("Date")


def quantile_intervals(feat_df, days, scenario_mult=1.0):
    clean = feat_df.dropna(subset=FEATURES + ["HHS_Care"]).copy()
    models = {}
    for q in [0.1, 0.5, 0.9]:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **GB_PARAMS)
        model.fit(clean[FEATURES], clean["HHS_Care"])
        models[q] = model

    history = clean[["HHS_Care", "Transfers", "Discharged"]].copy()
    base_transfer = float(history["Transfers"].tail(7).mean()) * scenario_mult
    base_discharge = float(history["Discharged"].tail(7).mean())

    rows = []
    for _ in range(days):
        next_date = history.index.max() + pd.Timedelta(days=1)
        Xf = make_future_features(history, next_date, base_transfer, base_discharge)
        lo = max(0, float(models[0.1].predict(Xf)[0]))
        mid = max(0, float(models[0.5].predict(Xf)[0]))
        hi = max(lo, float(models[0.9].predict(Xf)[0]))
        rows.append([next_date, lo, mid, hi])
        history.loc[next_date] = {"HHS_Care": mid, "Transfers": base_transfer, "Discharged": base_discharge}

    return pd.DataFrame(rows, columns=["Date", "Lower", "Forecast", "Upper"]).set_index("Date")


def forecast_sarima_series(series, horizon):
    """Fit SARIMA directly on a univariate series and forecast `horizon` steps with an 80% interval."""
    series = series.dropna()
    try:
        fitted = SARIMAX(
            series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        result = fitted.get_forecast(steps=horizon)
        mean = result.predicted_mean.clip(lower=0)
        ci = result.conf_int(alpha=0.2)  # ~80% interval, comparable to the 10/90 quantile band
        dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D", name="Date")
        mean.index = dates
        ci.index = dates
        lower = ci.iloc[:, 0].clip(lower=0)
        upper = ci.iloc[:, 1].clip(lower=lower)
        return pd.DataFrame({"Forecast": mean, "Lower": lower, "Upper": upper})
    except Exception:
        return None


def forecast_naive_series(series, horizon):
    """Persistence forecast: repeat the last observed value for every future day."""
    series = series.dropna()
    last_val = float(series.iloc[-1])
    recent_std = float(series.tail(14).std() or 0)
    dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D", name="Date")
    return pd.DataFrame({
        "Forecast": [last_val] * horizon,
        "Lower": [max(0.0, last_val - recent_std)] * horizon,
        "Upper": [last_val + recent_std] * horizon,
    }, index=dates)


def forecast_discharges_seasonal(df, horizon, lookback=56):
    """Seasonal-naive forecast for daily discharges using each weekday's recent average."""
    hist = df["Discharged"].tail(lookback)
    weekday_avg = hist.groupby(hist.index.dayofweek).mean()
    overall_avg = float(hist.mean())
    dates = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D", name="Date")
    values = [float(weekday_avg.get(d.dayofweek, overall_avg)) for d in dates]
    return pd.DataFrame({"Forecast": values}, index=dates)


# ---------- UI ----------
st.title("UAC Predictive Intelligence Dashboard")
st.caption("Predictive Forecasting of Care Load & Placement Demand")

st.subheader("Step 1: Upload your CSV file")
uploaded = st.file_uploader("Choose a file", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV file to activate the dashboard.")
    st.stop()

df, errors = load_data(uploaded)
if df is None:
    st.warning("The uploaded CSV could not be processed.")
    for e in errors:
        st.info(e)
    st.stop()

st.caption(
    f"Loaded {len(df):,} daily records from {df.index.min().date()} to {df.index.max().date()}."
)
with st.expander("Preview loaded data"):
    st.dataframe(df.head(10), use_container_width=True)

feat = add_features(df)

# Sidebar controls
st.sidebar.header("Forecast Controls")
horizon = st.sidebar.selectbox("Forecast horizon", [7, 14, 30], index=0)
model_choice = st.sidebar.selectbox("Forecast model", ["Gradient Boosting", "SARIMA", "Naive"])
scenario = st.sidebar.selectbox("Incoming transfer scenario", ["Normal", "High Intake (+20%)", "Stress (+40%)"])
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
    st.caption(f"Model: {model_choice} | Scenario: {scenario} | Horizon: {horizon} days")

    clean = feat.dropna(subset=FEATURES + ["HHS_Care"]).copy()

    if model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(**GB_PARAMS)
        model.fit(clean[FEATURES], clean["HHS_Care"])
        gb_forecast = recursive_gb_forecast(clean, model, horizon, scenario_mult)
        forecast = gb_forecast[["Forecast"]]
        interval = quantile_intervals(feat, horizon, scenario_mult)[["Lower", "Upper"]]
        extra_cols = gb_forecast[["Transfers_Assumed", "Discharges_Assumed", "Net_Pressure"]]
    elif model_choice == "SARIMA":
        sarima_result = forecast_sarima_series(df["HHS_Care"], horizon)
        if sarima_result is None:
            st.warning("SARIMA could not converge on this data; showing a Naive forecast instead.")
            sarima_result = forecast_naive_series(df["HHS_Care"], horizon)
        forecast = sarima_result[["Forecast"]]
        interval = sarima_result[["Lower", "Upper"]]
        extra_cols = None
        st.caption("Scenario assumptions apply only to the Gradient Boosting model; SARIMA uses historical time-series patterns only.")
    else:  # Naive
        naive_result = forecast_naive_series(df["HHS_Care"], horizon)
        forecast = naive_result[["Forecast"]]
        interval = naive_result[["Lower", "Upper"]]
        extra_cols = None
        st.caption("Scenario assumptions apply only to the Gradient Boosting model; Naive repeats the latest observed value.")

    hist = df.tail(min(120, len(df)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["HHS_Care"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast["Forecast"], mode="lines+markers", name="Forecast"))
    fig.add_trace(go.Scatter(
        x=list(interval.index) + list(interval.index[::-1]),
        y=list(interval["Upper"]) + list(interval["Lower"][::-1]),
        fill="toself", line=dict(color="rgba(0,0,0,0)"),
        name="Forecast uncertainty", hoverinfo="skip",
    ))
    fig.add_hline(y=capacity, line_dash="dash", annotation_text="Planning capacity")

    # Keep the visible range anchored to the real data (Historical/Forecast/Capacity)
    # so an extreme quantile-uncertainty band can't push the meaningful lines out of view.
    focus_vals = pd.concat([hist["HHS_Care"], forecast["Forecast"], pd.Series([capacity])])
    y_low, y_high = float(focus_vals.min()), float(focus_vals.max())
    pad = max((y_high - y_low) * 0.15, y_high * 0.05, 1.0)

    fig.update_layout(
        height=500, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date", yaxis_title="Children",
        hovermode="x unified", legend=dict(orientation="h"),
        yaxis=dict(range=[max(0, y_low - pad), y_high + pad]),
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

    table = forecast.rename(columns={"Forecast": "Forecast HHS Care"})
    if extra_cols is not None:
        table = table.join(extra_cols).rename(columns={
            "Transfers_Assumed": "Assumed Transfers",
            "Discharges_Assumed": "Assumed Discharges",
            "Net_Pressure": "Assumed Net Pressure",
        })
    st.dataframe(table.reset_index(), use_container_width=True, hide_index=True)

    st.markdown("### Scenario Comparison View")
    st.caption("All three transfer scenarios forecast together with the Gradient Boosting model, so the effect of the scenario assumption is directly comparable.")

    scenario_defs = {"Normal": 1.0, "High Intake (+20%)": 1.2, "Stress (+40%)": 1.4}
    gb_model = GradientBoostingRegressor(**GB_PARAMS)
    gb_model.fit(clean[FEATURES], clean["HHS_Care"])

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(x=hist.index, y=hist["HHS_Care"], mode="lines", name="Historical"))
    cmp_vals = [hist["HHS_Care"]]
    for label, mult in scenario_defs.items():
        cmp_forecast = recursive_gb_forecast(clean, gb_model, horizon, mult)
        fig_cmp.add_trace(go.Scatter(
            x=cmp_forecast.index, y=cmp_forecast["Forecast"],
            mode="lines+markers", name=label,
        ))
        cmp_vals.append(cmp_forecast["Forecast"])

    cmp_all = pd.concat(cmp_vals + [pd.Series([capacity])])
    cmp_low, cmp_high = float(cmp_all.min()), float(cmp_all.max())
    cmp_pad = max((cmp_high - cmp_low) * 0.15, cmp_high * 0.05, 1.0)

    fig_cmp.add_hline(y=capacity, line_dash="dash", annotation_text="Planning capacity")
    fig_cmp.update_layout(
        height=450, margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date", yaxis_title="Children",
        hovermode="x unified", legend=dict(orientation="h"),
        yaxis=dict(range=[max(0, cmp_low - cmp_pad), cmp_high + cmp_pad]),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

with tab2:
    st.subheader("Discharge / Placement Demand")
    st.caption("Short-term demand is estimated from historical discharge patterns and recent flow conditions.")

    discharge_roll = df["Discharged"].rolling(7).mean().iloc[-1]
    discharge_forecast = forecast_discharges_seasonal(df, horizon)
    expected_total = float(discharge_forecast["Forecast"].sum())

    a, b, c = st.columns(3)
    a.metric("Recent 7-day avg. discharges", f"{discharge_roll:,.1f}/day")
    b.metric(f"Estimated {horizon}-day demand", f"{expected_total:,.0f}")
    c.metric("Current net pressure", f"{net:+,.0f}")

    tail60 = df.tail(min(60, len(df)))
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=tail60.index, y=tail60["Discharged"], name="Daily discharges"))
    fig2.add_trace(go.Scatter(
        x=tail60.index, y=tail60["Discharged"].rolling(7).mean(),
        mode="lines", name="7-day average",
    ))
    fig2.add_trace(go.Bar(
        x=discharge_forecast.index, y=discharge_forecast["Forecast"],
        name="Forecasted discharges", marker=dict(color="rgba(99,179,237,0.55)"),
    ))
    fig2.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Date", yaxis_title="Children")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Forecasted discharges use each weekday's recent average (seasonal-naive), reflecting weekly processing patterns.")

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
        st.success(f"Best model on the selected hold-out window by MAE: {results.iloc[0]['Model']}")

        fig3 = go.Figure()
        idx = preds["test_index"]
        fig3.add_trace(go.Scatter(x=idx, y=preds["actual"], mode="lines", name="Actual"))
        for name in ["Naive", "SARIMA", "Gradient Boosting"]:
            p = preds.get(name)
            if p is not None and np.isfinite(p).any():
                fig3.add_trace(go.Scatter(x=idx, y=p, mode="lines", name=name))
        fig3.update_layout(
            height=480, margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Date", yaxis_title="Children", hovermode="x unified",
        )
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
    recent = df.tail(14)[["Apprehended", "CBP_Custody", "Transfers", "HHS_Care", "Discharged"]].copy()
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
    unsafe_allow_html=True,
)
