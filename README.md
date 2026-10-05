# Nigeria Food Price Intelligence Dashboard

A single-file Streamlit app covering **all four tasks** of the group assignment:

1. **Exploratory analysis** — food inflation trends, wholesale vs. retail spreads, and
   geographic price hotspots (interactive maps using latitude/longitude).
2. **Price prediction** — Linear Regression, Decision Tree, and Random Forest trained to
   predict price from commodity, market location, and unit — with a live "try it yourself"
   prediction widget.
3. **Forecasting** — 3–6 month ahead price forecasts for any commodity/market combination,
   using lag features and a gradient-boosted model (XGBoost if installed, otherwise
   scikit-learn's GradientBoostingRegressor as an automatic fallback).
4. **Price shock detection** — a rolling mean/std-based shock threshold (adjustable, default
   2 standard deviations) plus a Random Forest classifier that tries to predict shocks in
   advance from recent price momentum.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure nigeria_food_prices.csv is in the same folder as app.py
#    (it's included in this package)

# 3. Launch the dashboard
streamlit run app.py
```

It will open automatically at `http://localhost:8501`.

If you don't have the CSV in the folder for some reason, the sidebar has a file uploader —
just drop the CSV there instead.

## Notes for your write-up / presentation

- Every tab has a **"ℹ️ Methodology" expander** at the top explaining exactly what was done
  and why (normalization choices, outlier handling, feature engineering) — useful to quote
  directly in your report.
- The app deliberately **surfaces model limitations** rather than hiding them — e.g. Task 2
  shows that Linear Regression predicts a negative (impossible) price for ~17% of test cases,
  which is a genuine, explainable weakness of additive linear models with categorical
  features, and a good discussion point for "why does Random Forest perform better here?"
- `xgboost` is optional — if it's not installed, the forecasting tab automatically falls back
  to scikit-learn's `GradientBoostingRegressor`, which uses the same lag-feature approach and
  gives very similar results, so the dashboard still works out of the box.
