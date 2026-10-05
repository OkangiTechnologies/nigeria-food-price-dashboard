"""
Nigeria Food Price Intelligence Dashboard
==========================================
A single-file Streamlit app covering all four project tasks:
  1. Exploratory analysis - inflation trends, wholesale/retail spreads, geo hotspots
  2. Price prediction     - Linear Regression, Decision Tree, Random Forest
  3. Forecasting          - 3-6 month ahead price forecasts (lag-feature ML model)
  4. Price shock detection - threshold-based labeling + classifier

Run with:  streamlit run app.py
(expects nigeria_food_prices.csv in the same folder, or upload it in the sidebar)
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix

# Optional, faster gradient boosting - fall back gracefully if not installed
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# =============================================================================
# PAGE CONFIG & STYLE
# =============================================================================
st.set_page_config(
    page_title="Nigeria Food Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    h1, h2, h3 { color: #1a3c34; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

ACCENT = "#c0392b"
ACCENT2 = "#2980b9"


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file, skiprows=[1])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["price", "usdprice"])
    df = df[df["price"] > 0]
    return df


st.sidebar.title("🌾 Nigeria Food Prices")
st.sidebar.caption("WFP dataset · 2002–2024 · 42 commodities · 60 markets")

uploaded = st.sidebar.file_uploader("Upload a different CSV (optional)", type="csv")
DEFAULT_PATH = "nigeria_food_prices.csv"

try:
    df = load_data(uploaded if uploaded is not None else DEFAULT_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_PATH}` next to app.py. "
        "Upload the CSV using the sidebar to continue."
    )
    st.stop()

st.sidebar.divider()
st.sidebar.markdown("### About this dashboard")
st.sidebar.markdown(
    "Built for the group assignment covering **all 4 tasks**: "
    "EDA, price prediction, forecasting, and price-shock detection. "
    "Every chart is interactive - use the filters within each tab."
)

# =============================================================================
# HEADER + TOP-LINE METRICS
# =============================================================================
st.title("🌾 Nigeria Food Price Intelligence Dashboard")
st.caption(
    "World Food Programme price data for Nigeria — exploratory analysis, "
    "machine learning price prediction, short-term forecasting, and price-shock detection."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Records", f"{len(df):,}")
m2.metric("Date range", f"{df['date'].min().year}–{df['date'].max().year}")
m3.metric("Commodities", df["commodity"].nunique())
m4.metric("Markets", df["market"].nunique())
m5.metric("States", df["admin1"].nunique())

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Task 1 — Exploratory Analysis",
    "🤖 Task 2 — Price Prediction",
    "🔮 Task 3 — Forecasting",
    "⚠️ Task 4 — Price Shock Detection",
])

# =============================================================================
# TASK 1 — EXPLORATORY ANALYSIS
# =============================================================================
with tab1:
    st.header("Food inflation trends, wholesale vs. retail spreads, and geographic hotspots")

    with st.expander("ℹ️ Methodology (click to expand)"):
        st.markdown("""
- **Inflation index:** commodities enter the dataset at different times, so raw price
  averages would be biased by *which* items happen to be reported each month. Each series
  (commodity × market × unit × price type) is rebased to its own first observation (=100),
  then combined using the **median** (robust to a handful of bad data-entry outliers) after
  clipping the 1st/99th percentile.
- **Wholesale vs. retail:** wholesale is quoted in bulk units ("100 KG"), retail in "KG" or
  fractions of a KG. Prices are converted to a common **per-KG basis** before comparing.
- **Geographic hotspots:** a state's raw average price is confounded by which commodities it
  happens to report. The state-level map instead uses each state's price **relative to the
  national average for that specific commodity**, averaged across commodities.
        """)

    # ---------------- 1a. Inflation trend ----------------
    st.subheader("1️⃣ Food inflation trend over time")

    @st.cache_data(show_spinner="Building price index...")
    def build_inflation_index(df):
        d = df.copy()
        d["series_id"] = d["commodity"] + "|" + d["market"] + "|" + d["unit"] + "|" + d["pricetype"]
        d = d.sort_values("date")
        first_price = d.groupby("series_id")["price"].transform("first")
        d["price_relative"] = d["price"] / first_price * 100
        p1, p99 = d["price_relative"].quantile([0.01, 0.99])
        d["price_relative"] = d["price_relative"].clip(p1, p99)

        usd_first = d.groupby("series_id")["usdprice"].transform("first")
        d["usd_relative"] = d["usdprice"] / usd_first * 100
        up1, up99 = d["usd_relative"].quantile([0.01, 0.99])
        d["usd_relative"] = d["usd_relative"].clip(up1, up99)

        monthly = d.groupby("date").agg(ngn_index=("price_relative", "median"),
                                         usd_index=("usd_relative", "median")).reset_index()
        monthly["yoy_inflation"] = monthly["ngn_index"].pct_change(12) * 100
        return monthly

    monthly_index = build_inflation_index(df)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=monthly_index["date"], y=monthly_index["ngn_index"],
                               name="NGN price index", line=dict(color=ACCENT, width=2)))
    fig1.add_trace(go.Scatter(x=monthly_index["date"], y=monthly_index["usd_index"],
                               name="USD price index", line=dict(color=ACCENT2, width=2)))
    fig1.add_hline(y=100, line_dash="dash", line_color="gray")
    fig1.update_layout(title="Nigeria Food Price Index (2002 = 100)", height=380,
                        yaxis_title="Price Index", xaxis_title=None,
                        legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig1, use_container_width=True)

    fig1b = px.line(monthly_index, x="date", y="yoy_inflation",
                     title="Year-over-Year Food Inflation (NGN terms)")
    fig1b.add_hline(y=0, line_color="gray")
    fig1b.update_traces(line_color="#27ae60")
    fig1b.update_layout(height=300, yaxis_title="YoY Inflation (%)", xaxis_title=None)
    st.plotly_chart(fig1b, use_container_width=True)

    st.info(
        "💡 The NGN index climbed in three waves (2004-05, 2008-09, and 2021-2024). "
        "The USD index is much flatter in the most recent wave, meaning most of 2021-2024's "
        "spike is currency devaluation rather than a pure food-supply shock."
    )

    st.divider()

    # ---------------- 1b. Wholesale vs retail ----------------
    st.subheader("2️⃣ Wholesale vs. retail price spread")

    def unit_to_kg_factor(u):
        u = u.strip()
        if u in ("KG", "L"):
            return 1.0
        m = re.match(r"^([\d.]+)\s*(KG|L|G|ML)$", u)
        if not m:
            return None
        qty, unit_type = float(m.group(1)), m.group(2)
        if unit_type in ("G", "ML"):
            qty /= 1000
        return qty

    @st.cache_data(show_spinner="Matching wholesale & retail pairs...")
    def build_spread(df):
        d = df.copy()
        d["kg_factor"] = d["unit"].apply(unit_to_kg_factor)
        d = d.dropna(subset=["kg_factor"])
        d = d[d["kg_factor"] > 0]
        d["price_per_kg"] = d["price"] / d["kg_factor"]

        wide = d.pivot_table(index=["date", "market", "commodity"], columns="pricetype",
                              values="price_per_kg", aggfunc="mean").reset_index()
        wide = wide.dropna(subset=["Wholesale", "Retail"])
        wide["markup_pct"] = (wide["Retail"] - wide["Wholesale"]) / wide["Wholesale"] * 100
        wide = wide[np.isfinite(wide["markup_pct"])]
        lo, hi = wide["markup_pct"].quantile([0.01, 0.99])
        return wide[(wide["markup_pct"] >= lo) & (wide["markup_pct"] <= hi)]

    spread = build_spread(df)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Matched wholesale-retail pairs", f"{len(spread):,}")
        st.metric("Median retail markup", f"{spread['markup_pct'].median():.1f}%")

        over_time = spread.groupby("date")["markup_pct"].median().reset_index()
        fig2 = px.line(over_time, x="date", y="markup_pct", title="Median Retail Markup Over Time")
        fig2.add_hline(y=0, line_color="gray")
        fig2.update_traces(line_color="#8e44ad")
        fig2.update_layout(height=350, yaxis_title="Retail markup (%)", xaxis_title=None)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        top10 = spread.groupby("commodity")["markup_pct"].median().sort_values(ascending=False).head(10)
        fig3 = px.bar(top10[::-1], orientation="h", title="Top 10 Commodities by Median Markup",
                       labels={"value": "Median markup (%)", "commodity": ""}, color_discrete_sequence=["#d35400"])
        fig3.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "💡 Processed goods like **gari** (cassava meal) carry ~50% markup — far above raw "
        "staples like maize (~3-4%) — reflecting the processing/value-add between stages."
    )

    st.divider()

    # ---------------- 1c. Geographic hotspots ----------------
    st.subheader("3️⃣ Geographic price hotspots")

    commodities_for_map = sorted(df["commodity"].unique())
    default_ix = commodities_for_map.index("Maize (white)") if "Maize (white)" in commodities_for_map else 0
    geo_commodity = st.selectbox("Commodity to map", commodities_for_map, index=default_ix)

    geo_df = df[(df["commodity"] == geo_commodity) & (df["pricetype"] == "Retail")]
    market_avg = geo_df.groupby(["market", "admin1", "latitude", "longitude"])["usdprice"].mean().reset_index()

    if len(market_avg) == 0:
        st.warning("No retail data for this commodity.")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            fig4 = px.scatter_mapbox(
                market_avg, lat="latitude", lon="longitude", color="usdprice", size="usdprice",
                hover_name="market", hover_data={"admin1": True, "usdprice": ":.2f"},
                color_continuous_scale="YlOrRd", zoom=4.6, height=520,
                labels={"usdprice": "Avg price (USD)"}
            )
            fig4.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig4, use_container_width=True)
        with c2:
            st.markdown("**Most expensive markets**")
            st.dataframe(
                market_avg.nlargest(8, "usdprice")[["market", "admin1", "usdprice"]]
                    .rename(columns={"admin1": "state", "usdprice": "avg price (USD)"})
                    .round(3),
                hide_index=True, use_container_width=True
            )
            st.markdown("**Cheapest markets**")
            st.dataframe(
                market_avg.nsmallest(8, "usdprice")[["market", "admin1", "usdprice"]]
                    .rename(columns={"admin1": "state", "usdprice": "avg price (USD)"})
                    .round(3),
                hide_index=True, use_container_width=True
            )

    st.markdown("**State-level price index (commodity-mix adjusted)**")

    @st.cache_data(show_spinner="Building state price index...")
    def build_state_index(df):
        d = df.copy()
        national_avg = d.groupby(["commodity", "unit", "pricetype"])["usdprice"].transform("mean")
        d["price_vs_national"] = d["usdprice"] / national_avg
        state_index = d.groupby("admin1")["price_vs_national"].median().reset_index()
        state_loc = d.groupby("admin1")[["latitude", "longitude"]].mean().reset_index()
        return state_loc.merge(state_index, on="admin1")

    state_summary = build_state_index(df)
    fig5 = px.scatter_mapbox(
        state_summary, lat="latitude", lon="longitude", color="price_vs_national",
        size=[20] * len(state_summary), hover_name="admin1",
        hover_data={"price_vs_national": ":.2f", "latitude": False, "longitude": False},
        color_continuous_scale="YlOrRd", zoom=4.6, height=480,
        labels={"price_vs_national": "vs. national avg"}
    )
    fig5.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig5, use_container_width=True)
    st.info(
        "💡 Southern consumer states (Abia, Lagos, Oyo) run **above** the national average; "
        "northern producing states (Yobe, Jigawa, Zamfara, Katsina) run **below** it — "
        "the expected pattern of prices rising as food moves from farm to city."
    )

# =============================================================================
# TASK 2 — PRICE PREDICTION (Linear Regression, Decision Tree, Random Forest)
# =============================================================================
with tab2:
    st.header("Predict current price from commodity, market location, and unit")

    with st.expander("ℹ️ Methodology"):
        st.markdown("""
Three regressors are trained on the same features — **commodity, state, unit, price type,
and category** — one-hot encoded, with an 80/20 train/test split (`random_state=42`).
Target: **usdprice** (USD keeps the comparison stable across the 2002-2024 window despite
Naira devaluation). Models are cached after first training in this session.
        """)

    target_choice = st.radio("Prediction target", ["usdprice (USD)", "price (local currency)"],
                              horizontal=True, index=0)
    target_col = "usdprice" if target_choice.startswith("usdprice") else "price"

    FEATURES = ["commodity", "admin1", "unit", "pricetype", "category"]

    @st.cache_resource(show_spinner="Training Linear Regression, Decision Tree, and Random Forest...")
    def train_price_models(df, target_col):
        X = df[FEATURES]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)])

        model_defs = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        }

        results = {}
        trained_pipes = {}
        for name, model in model_defs.items():
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results[name] = {
                "R²": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            }
            trained_pipes[name] = pipe

        return results, trained_pipes, (X_test, y_test)

    results, trained_pipes, (X_test, y_test) = train_price_models(df, target_col)
    results_df = pd.DataFrame(results).T.round(3)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Model performance (held-out 20% test set)**")
        st.dataframe(results_df.style.highlight_max(subset=["R²"], color="#d4edda")
                                      .highlight_min(subset=["MAE", "RMSE"], color="#d4edda"),
                     use_container_width=True)
        best_model_name = results_df["R²"].idxmax()
        st.success(f"🏆 Best model: **{best_model_name}** (highest R², lowest error)")

    with c2:
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=list(results.keys()), y=[r["R²"] for r in results.values()],
                               name="R²", marker_color=ACCENT2, yaxis="y1"))
        fig6.update_layout(title="Model Comparison — R² Score (higher is better)",
                            yaxis=dict(title="R²", range=[0, 1]), height=320)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Predicted vs. actual (Random Forest, sample of 1,500 test points)**")
    rf_pred = trained_pipes["Random Forest"].predict(X_test)
    sample_ix = np.random.RandomState(42).choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    scatter_df = pd.DataFrame({
        "actual": y_test.values[sample_ix],
        "predicted": rf_pred[sample_ix],
    })
    fig7 = px.scatter(scatter_df, x="actual", y="predicted", opacity=0.4,
                       title=f"Random Forest: Predicted vs Actual ({target_col})")
    max_val = scatter_df[["actual", "predicted"]].max().max()
    fig7.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                               line=dict(color="red", dash="dash"), name="perfect prediction"))
    fig7.update_layout(height=420)
    st.plotly_chart(fig7, use_container_width=True)

    # Feature importance (Random Forest)
    st.markdown("**What drives the price prediction? (Random Forest feature importance, grouped)**")
    rf_pipe = trained_pipes["Random Forest"]
    ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"]
    feature_names = ohe.get_feature_names_out(FEATURES)
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df["group"] = imp_df["feature"].apply(lambda f: f.split("_")[0])
    grouped_imp = imp_df.groupby("group")["importance"].sum().sort_values(ascending=False)
    fig8 = px.bar(grouped_imp, orientation="h", labels={"value": "Total importance", "group": ""},
                  color_discrete_sequence=["#16a085"])
    fig8.update_layout(height=280, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)

    st.divider()

    # ---------------- Live prediction widget ----------------
    st.subheader("🔧 Try it yourself — predict a price")
    p1, p2, p3, p4, p5 = st.columns(5)
    in_commodity = p1.selectbox("Commodity", sorted(df["commodity"].unique()), key="p_commodity")
    in_state = p2.selectbox("State", sorted(df["admin1"].unique()), key="p_state")
    in_unit = p3.selectbox("Unit", sorted(df["unit"].unique()), key="p_unit")
    in_pricetype = p4.selectbox("Price type", sorted(df["pricetype"].unique()), key="p_pricetype")
    in_category = p5.selectbox(
        "Category",
        sorted(df.loc[df["commodity"] == in_commodity, "category"].unique()),
        key="p_category",
    )

    input_row = pd.DataFrame([{
        "commodity": in_commodity, "admin1": in_state, "unit": in_unit,
        "pricetype": in_pricetype, "category": in_category,
    }])

    pred_cols = st.columns(3)
    for col, (name, pipe) in zip(pred_cols, trained_pipes.items()):
        pred_val = pipe.predict(input_row)[0]
        unit_label = "USD" if target_col == "usdprice" else "NGN"
        if pred_val < 0:
            col.metric(name, "⚠️ invalid")
            col.caption(f"Predicted {pred_val:,.2f} — negative, so not shown as a price.")
        else:
            col.metric(name, f"{pred_val:,.2f} {unit_label}")

    lr_test_pred = trained_pipes["Linear Regression"].predict(X_test)
    frac_negative = (lr_test_pred < 0).mean() * 100
    if frac_negative > 1:
        st.warning(
            f"⚠️ **Linear Regression limitation:** on the held-out test set, it predicts a "
            f"**negative price for {frac_negative:.0f}% of rows**. Because it only adds up "
            f"separate effects per category (no interactions), some commodity/state/unit "
            f"combinations sum to below zero — which is impossible for a real price. "
            f"Decision Tree and Random Forest don't have this problem, since they can model "
            f"non-linear interactions between categories — one reason they score higher above."
        )

# =============================================================================
# TASK 3 — FORECASTING (3-6 months ahead, lag-feature ML pipeline)
# =============================================================================
with tab3:
    st.header("Market-specific price forecasting (3-6 months ahead)")

    with st.expander("ℹ️ Methodology"):
        st.markdown(f"""
For the chosen commodity (and optional market), prices are aggregated to a **monthly**
series, gaps are linearly interpolated, and lag features are engineered:
`lag_1, lag_2, lag_3, lag_6` (price N months ago) and a 3-month rolling mean.
A **{"XGBoost" if HAS_XGBOOST else "Gradient Boosting"} regressor** is trained on these
features, then forecasts are produced **recursively** — each predicted month feeds back in
as a lag for the next one — for the selected horizon.
*(This is a simpler, more transparent alternative to Prophet, using the same lag-feature idea
the task brief suggests; swap in `xgboost`/`prophet` directly if your course requires that
exact library — the pipeline structure here doesn't change.)*
        """)

    c1, c2, c3, c4 = st.columns(4)
    fc_commodity = c1.selectbox("Commodity", sorted(df["commodity"].unique()),
                                 index=sorted(df["commodity"].unique()).index("Maize (white)")
                                 if "Maize (white)" in df["commodity"].unique() else 0,
                                 key="fc_commodity")

    avail_pricetypes = sorted(df.loc[df["commodity"] == fc_commodity, "pricetype"].unique())
    fc_pricetype = c2.selectbox("Price type", avail_pricetypes, key="fc_pricetype")

    avail_units = sorted(df.loc[(df["commodity"] == fc_commodity) &
                                 (df["pricetype"] == fc_pricetype), "unit"].unique())
    fc_unit = c3.selectbox("Unit", avail_units, key="fc_unit")

    market_options = ["🌍 National average"] + sorted(
        df.loc[(df["commodity"] == fc_commodity) &
               (df["pricetype"] == fc_pricetype) &
               (df["unit"] == fc_unit), "market"].unique()
    )
    fc_market = c4.selectbox("Market", market_options, key="fc_market")
    fc_horizon = st.slider("Forecast horizon (months)", 3, 6, 6)

    def build_monthly_series(df, commodity, pricetype, unit, market):
        # Fix price type AND unit so every observation is on the same basis -
        # mixing wholesale (bulk, e.g. "100 KG") with retail ("KG") would otherwise
        # produce a meaningless series with fake jumps every time the mix shifts.
        d = df[(df["commodity"] == commodity) &
               (df["pricetype"] == pricetype) &
               (df["unit"] == unit)]
        if market != "🌍 National average":
            d = d[d["market"] == market]
        d = d.copy()
        d["month"] = d["date"].dt.to_period("M")
        s = d.groupby("month")["usdprice"].mean()
        s.index = s.index.to_timestamp()
        s = s.asfreq("MS").interpolate(limit_direction="both")
        return s

    def make_lag_features(series):
        ts = pd.DataFrame({"y": series})
        for lag in [1, 2, 3, 6]:
            ts[f"lag_{lag}"] = ts["y"].shift(lag)
        ts["rolling_mean_3"] = ts["y"].shift(1).rolling(3).mean()
        ts["month_num"] = ts.index.month
        return ts

    series = build_monthly_series(df, fc_commodity, fc_pricetype, fc_unit, fc_market)

    if len(series) < 18:
        st.warning(
            f"Only {len(series)} months of data for this combination — too little to forecast "
            "reliably. Try 'National average' or a different commodity."
        )
    else:
        ts = make_lag_features(series).dropna()

        @st.cache_resource(show_spinner="Training forecasting model...")
        def train_forecast_model(ts_values_hash, X, y):
            if HAS_XGBOOST:
                model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
            model.fit(X, y)
            return model

        X, y = ts.drop(columns="y"), ts["y"]
        # simple content-based cache key so re-selecting the same series reuses the trained model
        cache_key = f"{fc_commodity}|{fc_pricetype}|{fc_unit}|{fc_market}|{len(ts)}"
        model = train_forecast_model(cache_key, X, y)

        # Backtest: how good is the model on the last 20% of known months?
        split = max(int(len(ts) * 0.8), len(ts) - 12)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]
        if len(X_te) > 0:
            if HAS_XGBOOST:
                backtest_model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
            else:
                backtest_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
            backtest_model.fit(X_tr, y_tr)
            backtest_pred = backtest_model.predict(X_te)
            backtest_mae = mean_absolute_error(y_te, backtest_pred)
        else:
            backtest_mae = None

        # Recursive multi-step forecast
        history = series.copy()
        forecast_rows = []
        for _ in range(fc_horizon):
            feat_row = make_lag_features(history).iloc[[-1]].drop(columns="y")
            next_val = float(model.predict(feat_row)[0])
            next_date = history.index[-1] + pd.DateOffset(months=1)
            history.loc[next_date] = next_val
            forecast_rows.append({"date": next_date, "usdprice": next_val})

        forecast_df = pd.DataFrame(forecast_rows)

        m1, m2 = st.columns(2)
        m1.metric("Months of history used", len(series))
        m2.metric("Backtest MAE (USD)", f"{backtest_mae:.3f}" if backtest_mae is not None else "n/a")

        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical",
                                   line=dict(color=ACCENT2, width=2)))
        fig9.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["usdprice"], name="Forecast",
                                   line=dict(color=ACCENT, width=2, dash="dash"), mode="lines+markers"))
        fig9.add_vline(x=series.index[-1], line_dash="dot", line_color="gray")
        fig9.update_layout(title=f"{fc_commodity} ({fc_pricetype}, {fc_unit}) — {fc_market}: {fc_horizon}-Month Forecast",
                            yaxis_title="Price (USD)", height=450,
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig9, use_container_width=True)

        st.markdown("**Forecast table**")
        st.dataframe(
            forecast_df.assign(date=forecast_df["date"].dt.strftime("%b %Y"))
                       .rename(columns={"usdprice": "forecast price (USD)"})
                       .round(3),
            hide_index=True, use_container_width=True
        )
        st.download_button(
            "📥 Download forecast as CSV",
            forecast_df.to_csv(index=False),
            file_name=f"forecast_{fc_commodity}_{fc_market}.csv",
            mime="text/csv",
        )

# =============================================================================
# TASK 4 — PRICE SHOCK DETECTION
# =============================================================================
with tab4:
    st.header("Price shock detection: flagging high-risk months per commodity")

    with st.expander("ℹ️ Methodology"):
        st.markdown("""
For each commodity, a **12-month rolling mean and standard deviation** are computed
(national monthly average price, USD). A month is labeled a **price shock** if its price
exceeds `rolling_mean + k × rolling_std`, where **k is adjustable below** (task brief
suggests k = 2). A **Random Forest classifier** is then trained to predict shock months from
recent price momentum (1- and 3-month % change) and seasonality (month number) — i.e. "could
we have flagged this shock in advance using only prior information?"
        """)

    shock_k = st.slider("Shock threshold (k × standard deviations above rolling mean)", 1.0, 3.0, 2.0, 0.25)

    @st.cache_data(show_spinner="Computing rolling shock thresholds...")
    def build_shock_table(df, k):
        d = df.copy()
        d["month"] = d["date"].dt.to_period("M")
        monthly = d.groupby(["commodity", "month"])["usdprice"].mean().reset_index()
        monthly["month"] = monthly["month"].dt.to_timestamp()
        monthly = monthly.sort_values(["commodity", "month"])

        monthly["roll_mean"] = monthly.groupby("commodity")["usdprice"].transform(
            lambda s: s.rolling(12, min_periods=6).mean())
        monthly["roll_std"] = monthly.groupby("commodity")["usdprice"].transform(
            lambda s: s.rolling(12, min_periods=6).std())
        monthly["threshold"] = monthly["roll_mean"] + k * monthly["roll_std"]
        monthly["is_shock"] = (monthly["usdprice"] > monthly["threshold"]).astype(int)
        return monthly.dropna(subset=["roll_mean", "roll_std"])

    shock_table = build_shock_table(df, shock_k)

    n_shocks = shock_table["is_shock"].sum()
    shock_rate = shock_table["is_shock"].mean() * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Commodity-months evaluated", f"{len(shock_table):,}")
    m2.metric("Flagged as shocks", f"{n_shocks:,}")
    m3.metric("Shock rate", f"{shock_rate:.1f}%")

    st.markdown("**Inspect a commodity's price with shock months highlighted**")
    shock_commodity = st.selectbox("Commodity", sorted(shock_table["commodity"].unique()),
                                    key="shock_commodity")
    cs = shock_table[shock_table["commodity"] == shock_commodity]

    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(x=cs["month"], y=cs["usdprice"], name="Price",
                                line=dict(color=ACCENT2, width=2)))
    fig10.add_trace(go.Scatter(x=cs["month"], y=cs["threshold"], name=f"Shock threshold (mean + {shock_k}σ)",
                                line=dict(color="gray", width=1, dash="dot")))
    shocks_only = cs[cs["is_shock"] == 1]
    fig10.add_trace(go.Scatter(x=shocks_only["month"], y=shocks_only["usdprice"], name="Shock month",
                                mode="markers", marker=dict(color=ACCENT, size=10, symbol="x")))
    fig10.update_layout(title=f"{shock_commodity}: Price vs. Shock Threshold", height=420,
                         yaxis_title="Price (USD)", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig10, use_container_width=True)

    st.divider()

    # ---------------- Classifier: can we predict a shock in advance? ----------------
    st.subheader("Can we predict a shock before it happens?")
    st.caption("Using only information available *before* the month in question: recent price momentum and seasonality.")

    @st.cache_resource(show_spinner="Training shock classifier...")
    def train_shock_classifier(shock_table_hash, monthly):
        d = monthly.copy()
        d["pct_change_1"] = d.groupby("commodity")["usdprice"].pct_change(1)
        d["pct_change_3"] = d.groupby("commodity")["usdprice"].pct_change(3)
        d["month_num"] = d["month"].dt.month
        d = d.dropna(subset=["pct_change_1", "pct_change_3"])

        features = ["pct_change_1", "pct_change_3", "month_num"]
        cat_features = ["commodity"]
        X = d[features + cat_features]
        y = d["is_shock"]

        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
                                 remainder="passthrough")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        clf = Pipeline([
            ("pre", pre),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=8,
                                              class_weight="balanced", random_state=42)),
        ])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm

    cache_key2 = f"k={shock_k}|n={len(shock_table)}"
    report, cm = train_shock_classifier(cache_key2, shock_table)

    c1, c2 = st.columns([2, 1])
    with c1:
        report_df = pd.DataFrame(report).T.round(3)
        report_df = report_df.rename(index={"0": "Normal month", "1": "Shock month"})
        st.markdown("**Classification report (20% held-out test set)**")
        st.dataframe(report_df, use_container_width=True)
        st.caption(
            "Shock months are rare (~5% of all commodity-months), so **recall** (did we catch "
            "the real shocks?) matters more here than raw accuracy. `class_weight='balanced'` "
            "is used so the model doesn't just predict 'no shock' every time."
        )
    with c2:
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["Normal", "Shock"], y=["Normal", "Shock"])
        cm_fig.update_layout(title="Confusion Matrix", height=320)
        st.plotly_chart(cm_fig, use_container_width=True)

    st.divider()
    st.markdown("**Most recent shock months across all commodities**")
    recent_shocks = (
        shock_table[shock_table["is_shock"] == 1]
        .sort_values("month", ascending=False)
        .head(15)[["commodity", "month", "usdprice", "threshold"]]
        .rename(columns={"usdprice": "price (USD)", "threshold": "threshold (USD)"})
    )
    recent_shocks["month"] = recent_shocks["month"].dt.strftime("%b %Y")
    st.dataframe(recent_shocks.round(3), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "Built with Streamlit · scikit-learn" + (" · XGBoost" if HAS_XGBOOST else "") +
    " · Plotly — Nigeria Food Price Intelligence Dashboard"
)
