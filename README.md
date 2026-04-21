# 🏭 Smart Procurement Reliability Dashboard

### P2 — Procurement Reliability | UE23CS342BA1 SCME | PES University

---

## 📌 Problem Statement

Organizations face challenges in ensuring timely and reliable procurement of raw materials due to supplier delays, cost fluctuations, and lack of predictive insights.

This leads to supply chain disruptions, increased costs, and inefficient decision-making.

---

## 💡 Solution

This project provides an interactive Streamlit dashboard integrated with Machine Learning to:

* Predict procurement delays
* Estimate cost overruns
* Forecast demand
* Evaluate supplier performance

It enables proactive, data-driven decision-making instead of reactive management.

---

## 🚀 How to Run

### 1. Install dependencies

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

### 4. (Optional) Run self-test

```bash
python test_all.py
```

### 5. Launch dashboard

```bash
streamlit run app.py
```

### 6. Open in browser

http://localhost:8501

---

## 📁 Project Structure

```
procurement_project/
│
├── generate_data.py      
├── create_db.py          
├── generate_erd.py       
├── ml_model.py           
├── explainability.py     
├── app.py                
├── test_all.py           
│
├── suppliers.csv         
├── materials.csv         
├── inventory.csv         
├── orders.csv            
├── supplier_stats.csv    
├── forecast.csv          
│
├── delay_model.pkl       
├── overrun_model.pkl     
├── cost_model.pkl        
├── procurement.db        
└── erd_schema.png        
```

---

## 🤖 Machine Learning Models

| Model            | Algorithm            | Purpose                                | Metric              |
| ---------------- | -------------------- | -------------------------------------- | ------------------- |
| Delay Prediction | Random Forest        | Predicts if an order will be delayed   | Accuracy            |
| Cost Overrun     | Gradient Boosting    | Predicts extra cost due to disruptions | R², MAE             |
| Supplier Score   | Weighted KPI Model   | Evaluates supplier reliability (0–100) | Interpretable       |
| Demand Forecast  | Seasonal Trend Model | Predicts demand for next 3 months      | Trend + Seasonality |

---

## 📊 Dashboard Features

* KPI Dashboard (delay rate, cost, supplier score)
* Supplier Analysis (performance comparison)
* Order Insights (spend trends, heatmaps)
* ML Predictions (delay + cost forecasting)
* Alerts (stockout & risk alerts)
* Demand Forecast (future demand trends)
* Explainability (SHAP-based insights)
* Database Explorer (SQL + ER diagram)

---

## 🗄️ Database Design

Relational schema:

Suppliers (1) ────< Orders >──── (1) Materials
│
└── Inventory (1:1 with Materials)

* Foreign keys: supplier_id, material_id
* Supports analytical queries

---

## 🌟 Key Highlights

* SHAP Explainability (transparent ML decisions)
* Cost Overrun Prediction (real-world modeling)
* Demand Forecasting (seasonal trends)
* Supplier Health Scoring
* What-if Scenario Analysis
* Interactive Dashboard

---

## 📊 Business Impact

* Reduces procurement delays
* Improves supplier selection
* Controls cost overruns
* Prevents stockouts
* Enables proactive decision-making

---

## 🔮 Future Work

* Real-time data integration
* Cloud deployment
* ERP system integration
* Advanced ML models

---

## 🛠️ Troubleshooting

* ModuleNotFoundError → pip install <module>
* FileNotFoundError → Run generate_data.py
* Dashboard not opening → Check project folder

---

## 👨‍💻 Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas, NumPy
* SHAP

---

## 📎 Note

This project uses synthetic data designed to simulate real-world procurement scenarios.
 Monica – PES1UG24CS813
 Bhavani – PES1UG23CS144
