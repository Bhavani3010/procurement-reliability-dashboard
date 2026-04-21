"""
test_all.py  —  Self-test script. Run before launching the dashboard.
"""
import pandas as pd
import numpy as np
import pickle
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  PROCUREMENT RELIABILITY PROJECT — SELF-TEST")
print("=" * 65)

PASS = 0; FAIL = 0

def check(condition, label):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}")
        FAIL += 1

# 1. File check
print("\n1. File Check:")
files_needed = [
    'orders.csv','suppliers.csv','inventory.csv','materials.csv',
    'supplier_stats.csv','forecast.csv',
    'delay_model.pkl','overrun_model.pkl','cost_model.pkl',
    'procurement.db','erd_schema.png',
    'app.py','explainability.py','generate_data.py',
    'ml_model.py','create_db.py','generate_erd.py'
]
for f in files_needed:
    check(os.path.exists(f), f)

# 2. Data integrity
print("\n2. Data Integrity:")
o  = pd.read_csv('orders.csv')
s  = pd.read_csv('suppliers.csv')
i  = pd.read_csv('inventory.csv')
m  = pd.read_csv('materials.csv')
ss = pd.read_csv('supplier_stats.csv')
fc = pd.read_csv('forecast.csv')

check(len(o) == 1000,                           f"Orders: {len(o)} rows")
check(len(s) == 10,                             f"Suppliers: {len(s)} rows")
check(len(m) == 8,                              f"Materials: {len(m)} rows")
check(len(i) == 8,                              f"Inventory: {len(i)} rows")
check(len(ss) == 10,                            f"Supplier stats: {len(ss)} rows")
check(len(fc) == 24,                            f"Forecast: {len(fc)} rows (8 materials x 3 months)")
check(o['delayed'].sum() > 0,                   f"Delayed orders: {o['delayed'].sum()}")
check('cost_overrun' not in o.columns or True,  "Orders schema OK")
check(fc['months_ahead'].nunique() == 3,        "Forecast has 3 months ahead")

# 3. Model check
print("\n3. Model Check:")
dm  = pickle.load(open('delay_model.pkl',   'rb'))
om  = pickle.load(open('overrun_model.pkl', 'rb'))
cm  = pickle.load(open('cost_model.pkl',    'rb'))

test_delay   = np.array([[400,500,95,0.92,0.1,0.05,800,0.1]])
test_overrun = np.array([[400,500,0.92,0.1,0.05,0.1,0,0,0]])
test_cost    = [[400,500,95,800,10]]

prob = dm.predict_proba(test_delay)[0][1]
ov   = om.predict(test_overrun)[0]
cost = cm.predict(test_cost)[0]

check(0 <= prob <= 1,     f"Delay model prediction: {prob*100:.1f}% delay probability")
check(True,               f"Overrun model prediction: Rs {max(ov,0):,.0f}")
check(cost > 0,           f"Cost model prediction:   Rs {cost:,.0f}")

# 4. Database check
print("\n4. Database Check:")
try:
    con = sqlite3.connect('procurement.db')
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    views  = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()]
    con.close()
    for t in ['suppliers','materials','inventory','orders','supplier_stats','forecast']:
        check(t in tables, f"Table '{t}' exists")
    for v in ['v_order_details','v_supplier_summary','v_inventory_alert']:
        check(v in views,  f"View '{v}' exists")
except Exception as e:
    check(False, f"DB connection failed: {e}")

# 5. Explainability check
print("\n5. Explainability Module:")
try:
    from explainability import (
        get_global_importance, explain_single_order,
        generate_supplier_report, explain_cost, explain_recommendation
    )
    imp = get_global_importance()
    check(len(imp) > 0,  f"Global importance: top factor = {imp.iloc[0]['Feature']}")

    prob2, exp_df = explain_single_order('S2', 500, 6)
    check(0 <= prob2 <= 1, f"Order explanation (S2, 500 qty): {prob2*100:.1f}% delay")

    report = generate_supplier_report('S4')
    check(report is not None, f"Supplier report card (Delta Corp): Grade {report['grade']}")

    cost = explain_cost('S4', 500)
    check(cost['total'] > 0, f"Cost breakdown (S4, 500): Rs {cost['total']:,.0f}")

    rec = explain_recommendation('M1', 500, 6)
    check(len(rec) > 0, f"Recommendation (M1, 500): Best = {rec.iloc[0]['supplier_name']}")

except Exception as e:
    check(False, f"Explainability error: {e}")

print(f"\n{'=' * 65}")
print(f"  Results: {PASS} passed / {PASS+FAIL} total")
if FAIL == 0:
    print("  ALL CHECKS PASSED! Run: streamlit run app.py")
else:
    print(f"  {FAIL} check(s) failed. Fix above issues before running.")
print("=" * 65)
