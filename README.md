# 🏭 Smart Procurement Reliability Dashboard
### P2 — Procurement Reliability | UE23CS342BA1 SCME | PES University

---

## 📁 Project File Structure

```
procurement_project/
│
├── generate_data.py      ← Step 1:  Generate all CSV datasets
├── create_db.py          ← Step 1b: Build SQLite database (procurement.db)
├── generate_erd.py       ← Step 1c: Render ER diagram (erd_schema.png)
├── ml_model.py           ← Step 2:  Train all ML models
├── explainability.py     ← Helper: SHAP + report cards
├── app.py                ← Main Streamlit dashboard (8 pages)
├── test_all.py           ← Self-test to verify everything
│
├── suppliers.csv         ← 10 suppliers with risk factors
├── materials.csv         ← 8 materials (electronics + raw)
├── inventory.csv         ← Current stock levels & reorder logic
├── orders.csv            ← 1000 orders over 2 years
├── supplier_stats.csv    ← Aggregated supplier performance + health scores
├── forecast.csv          ← 3-month demand predictions per material
│
├── delay_model.pkl       ← Random Forest (delay classification)
├── overrun_model.pkl     ← Gradient Boosting (cost overrun regression)
├── cost_model.pkl        ← Linear Regression (baseline cost display)
├── procurement.db        ← SQLite database (6 tables + 3 views)
└── erd_schema.png        ← Entity Relationship Diagram
```

---

## ⚙️ Setup (First Time Only)

### 1. Install Python libraries
```bash
pip install pandas numpy scikit-learn streamlit plotly shap matplotlib seaborn openpyxl
```

### 2. Generate dataset + database + ER diagram
```bash
python generate_data.py
python create_db.py
python generate_erd.py
```

### 3. Train ML models
```bash
python ml_model.py
```

### 4. Self-test (recommended)
```bash
python test_all.py
```

### 5. Launch dashboard
```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

---

## 🧠 ML Models

| Model | Algorithm | What it predicts | Key Metric |
|-------|-----------|-----------------|------------|
| Model 1 | Random Forest Classifier | Will this order be delayed? (0/1) | Accuracy + 5-fold CV |
| Model 2 | Gradient Boosting Regressor | How much cost overrun (₹) beyond baseline? | R², MAE |
| Model 3 | Weighted KPI Score | Supplier health score (0–100) | Interpretable formula |
| Model 4 | Seasonal Linear Trend | Demand per material (1/2/3 months ahead) | Trend + seasonal factor |

> **Note on Model 2:** Unlike a simple total-cost formula, this model predicts the *excess* cost
> caused by supply chain disruptions (delays, partial deliveries, quality failures) — genuine ML.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, spend trend, supplier health, seasonal comparison |
| 📦 Supplier Analysis | Scorecards, radar chart, delay/quality/cost comparison |
| 📊 Procurement Orders | Stacked cost, spend heatmap, filterable order table |
| 🤖 ML Predictions | Delay gauge, cost overrun predictor (Gradient Boosting), backup recommendations |
| ⚠️ Alerts | Stockout alerts, JIT reorder, What-If scenario analysis |
| 📈 Demand Forecast | 3-month ahead forecast, seasonal factors, historical trends |
| 🔍 Explainability | SHAP global importance, order explanation, supplier report cards, cost breakdown |
| 🗄️ Database & Schema | ER diagram, SQL DDL, live SQLite query explorer |

---

## 🗄️ Database Schema

6 tables with enforced foreign keys:

```
suppliers   ──< orders >── materials
               │
supplier_stats─┘         └── inventory
                          └── forecast
```

3 pre-built views: `v_order_details`, `v_supplier_summary`, `v_inventory_alert`

---

## 🌟 Key Features

1. **SHAP Explainability** — not just predictions, but *why* each prediction was made
2. **Cost Overrun Model** — Gradient Boosting on disruption risk factors (not formula recovery)
3. **3-Month Demand Forecast** — seasonal decomposition with trend + seasonal multiplier
4. **Supplier Health Score** — 6-factor weighted KPI model (like a credit score)
5. **Live DB Explorer** — run custom SQL queries on the live database from the dashboard
6. **What-If Scenario** — "what if Supplier X fails?" with combination ordering fallback
7. **JIT Smart Reorder** — calculates exact reorder trigger based on lead time + daily consumption
8. **Radar Chart Comparison** — multi-dimensional supplier comparison in one view

---

## 🛠️ Troubleshooting

**ModuleNotFoundError** → `pip install <module_name>`

**FileNotFoundError** → Re-run `generate_data.py` then `ml_model.py`

**Dashboard not opening** → Make sure you're in the project folder: `cd procurement_project`
