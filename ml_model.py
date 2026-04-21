"""
ml_model.py  —  Step 2: Train all ML models
Run after generate_data.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_absolute_percentage_error, r2_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

orders    = pd.read_csv('orders.csv')
suppliers = pd.read_csv('suppliers.csv')
df = orders.merge(suppliers, on='supplier_id')

print("=" * 60)
print("   PROCUREMENT RELIABILITY — ML TRAINING PIPELINE")
print("=" * 60)

# ── MODEL 1: Delay Prediction (Random Forest) ─────────────
print("\nMODEL 1: Delay Prediction  [Random Forest Classifier]")

features_delay = [
    'distance_km', 'quantity', 'price_per_unit',
    'reliability_score', 'geopolitical_risk',
    'strike_risk', 'transport_cost', 'weather_risk'
]
X_delay = df[features_delay]
y_delay = df['delayed']

X_train, X_test, y_train, y_test = train_test_split(
    X_delay, y_delay, test_size=0.2, random_state=42, stratify=y_delay
)

delay_model = RandomForestClassifier(
    n_estimators=200, random_state=42, max_depth=8,
    min_samples_leaf=5, class_weight='balanced'
)
delay_model.fit(X_train, y_train)
y_pred = delay_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(delay_model, X_delay, y_delay, cv=5, scoring='accuracy')
print(f"  Test Accuracy     : {acc*100:.2f}%")
print(f"  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['On-Time','Delayed']))

# ── MODEL 2: Cost Overrun Prediction (Gradient Boosting) ──
print("\nMODEL 2: Cost Overrun Prediction  [Gradient Boosting Regressor]")
print("  (Predicts extra cost beyond baseline due to delays/storage/penalties)")

df['baseline_cost'] = df['quantity'] * df['price_per_unit'] + df['transport_cost']
df['cost_overrun']  = df['total_cost'] - df['baseline_cost']

features_overrun = [
    'distance_km', 'quantity', 'reliability_score',
    'geopolitical_risk', 'strike_risk', 'weather_risk',
    'delay_days', 'partial_delivery', 'quality_issue'
]
X_ov = df[features_overrun]
y_ov = df['cost_overrun']

X_tr_ov, X_te_ov, y_tr_ov, y_te_ov = train_test_split(
    X_ov, y_ov, test_size=0.2, random_state=42
)

overrun_model = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr',    GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    ))
])
overrun_model.fit(X_tr_ov, y_tr_ov)
y_pred_ov = overrun_model.predict(X_te_ov)
print(f"  R2 Score: {r2_score(y_te_ov, y_pred_ov):.4f}")
print(f"  MAE    : Rs {mean_absolute_error(y_te_ov, y_pred_ov):,.2f}")

# Keep LR for cost display transparency
features_cost = ['distance_km', 'quantity', 'price_per_unit', 'transport_cost', 'storage_cost']
X_cost = df[features_cost]
y_cost = df['total_cost']
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)
cost_model = LinearRegression()
cost_model.fit(X_tr_c, y_tr_c)

# ── MODEL 3: Supplier Health Score ────────────────────────
print("\nMODEL 3: Supplier Health Score  [Weighted KPI Model]")

supplier_stats = df.groupby('supplier_id').agg(
    total_orders       = ('order_id',          'count'),
    delayed_orders     = ('delayed',            'sum'),
    avg_delay_days     = ('delay_days',         'mean'),
    avg_total_cost     = ('total_cost',         'mean'),
    partial_deliveries = ('partial_delivery',   'sum'),
    quality_issues     = ('quality_issue',      'sum'),
    backup_activations = ('backup_used',        'sum'),
    avg_overrun        = ('cost_overrun',       'mean'),
).reset_index()

supplier_stats = supplier_stats.merge(
    suppliers[['supplier_id','supplier_name','reliability_score',
               'geopolitical_risk','strike_risk','available_stock','quality_score']],
    on='supplier_id'
)

supplier_stats['delay_rate']         = supplier_stats['delayed_orders']     / supplier_stats['total_orders']
supplier_stats['partial_rate']       = supplier_stats['partial_deliveries'] / supplier_stats['total_orders']
supplier_stats['quality_issue_rate'] = supplier_stats['quality_issues']     / supplier_stats['total_orders']

max_overrun = supplier_stats['avg_overrun'].max() + 1
supplier_stats['health_score'] = (
    (supplier_stats['reliability_score']         * 30) +
    ((1 - supplier_stats['delay_rate'])           * 25) +
    ((1 - supplier_stats['geopolitical_risk'])    * 15) +
    ((1 - supplier_stats['strike_risk'])          * 10) +
    (supplier_stats['quality_score']              * 15) +
    (np.clip(1 - supplier_stats['avg_overrun'] / max_overrun, 0, 1) * 5)
).round(2)

def get_grade(score):
    if score >= 80:   return 'A - Low Risk'
    elif score >= 65: return 'B - Medium Risk'
    elif score >= 50: return 'C - High Risk'
    else:             return 'D - Very High Risk'

supplier_stats['grade'] = supplier_stats['health_score'].apply(get_grade)
print(supplier_stats[['supplier_name','health_score','grade','delay_rate']].to_string(index=False))

# ── MODEL 4: Demand Forecasting (Seasonal Trend) ──────────
print("\nMODEL 4: Demand Forecasting  [Seasonal Linear Trend Model]")

orders_m = orders.copy()
orders_m['order_date']  = pd.to_datetime(orders_m['order_date'])
orders_m['year']        = orders_m['order_date'].dt.year
orders_m['month']       = orders_m['order_date'].dt.month
orders_m['month_index'] = (orders_m['year'] - orders_m['year'].min()) * 12 + orders_m['month']

monthly_demand = (
    orders_m.groupby(['material_id','year','month','month_index'])['quantity']
    .sum().reset_index()
)

forecasts = []
for material in monthly_demand['material_id'].unique():
    mat_df = monthly_demand[monthly_demand['material_id'] == material].copy().sort_values('month_index')

    # Seasonal index per calendar month
    overall_mean = mat_df['quantity'].mean()
    seasonal_idx = (mat_df.groupby('month')['quantity'].mean() / overall_mean).to_dict()

    # Linear trend
    X_t = mat_df[['month_index']].values
    y_t = mat_df['quantity'].values
    lr  = LinearRegression().fit(X_t, y_t)

    last_idx   = int(mat_df['month_index'].max())
    last_month = int(mat_df['month'].iloc[-1])

    for ahead in range(1, 4):
        future_idx   = last_idx + ahead
        future_month = ((last_month - 1 + ahead) % 12) + 1
        trend_pred   = lr.predict([[future_idx]])[0]
        seas_factor  = seasonal_idx.get(future_month, 1.0)
        final_pred   = max(trend_pred * seas_factor, 0)

        forecasts.append({
            'material_id':      material,
            'next_month':       future_month,
            'months_ahead':     ahead,
            'predicted_demand': round(final_pred),
            'trend_component':  round(trend_pred),
            'seasonal_factor':  round(seas_factor, 3),
        })

forecast_df = pd.DataFrame(forecasts)
print(forecast_df[forecast_df['months_ahead'] == 1].to_string(index=False))

# ── Save ─────────────────────────────────────────────────
pickle.dump(delay_model,   open('delay_model.pkl',   'wb'))
pickle.dump(overrun_model, open('overrun_model.pkl', 'wb'))
pickle.dump(cost_model,    open('cost_model.pkl',    'wb'))
supplier_stats.to_csv('supplier_stats.csv', index=False)
forecast_df.to_csv('forecast.csv', index=False)

print("\n✅ All models saved: delay_model.pkl, overrun_model.pkl, cost_model.pkl")
