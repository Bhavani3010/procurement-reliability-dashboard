import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Suppliers (10) ──────────────────────────────────────
suppliers = pd.DataFrame({
    'supplier_id': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10'],
    'supplier_name': [
        'Alpha Materials', 'Beta Supplies', 'Gamma Industries',
        'Delta Corp', 'Echo Traders', 'Falcon Parts',
        'Global Sources', 'Horizon Goods', 'Indus Supply', 'Jetstream Co'
    ],
    'location': [
        'Chennai', 'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad',
        'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat'
    ],
    'distance_km': [400, 200, 1200, 50, 600, 150, 1800, 900, 1100, 700],
    'price_per_unit': [95, 85, 78, 105, 90, 88, 72, 98, 82, 93],
    'transport_cost_per_km': [2, 2.5, 1.8, 3, 2.2, 2.8, 1.5, 2.1, 1.9, 2.3],
    'reliability_score': [0.92, 0.65, 0.88, 0.97, 0.70, 0.85, 0.60, 0.91, 0.78, 0.83],
    'available_stock': [1200, 800, 1500, 600, 900, 1100, 2000, 750, 1300, 950],
    'storage_cost_per_day': [5, 4, 6, 7, 5, 4.5, 3.5, 6.5, 5.5, 4.8],
    'geopolitical_risk': [0.1, 0.3, 0.5, 0.05, 0.4, 0.1, 0.6, 0.2, 0.35, 0.25],
    'strike_risk': [0.05, 0.2, 0.15, 0.02, 0.25, 0.08, 0.3, 0.12, 0.18, 0.1],
    'quality_score': [0.95, 0.75, 0.90, 0.98, 0.72, 0.88, 0.65, 0.93, 0.82, 0.87],
    'max_capacity': [2000, 1500, 3000, 1000, 1800, 2200, 4000, 1500, 2500, 1900]
})

# ── Materials (8) ───────────────────────────────────────
materials = pd.DataFrame({
    'material_id': ['M1','M2','M3','M4','M5','M6','M7','M8'],
    'material_name': [
        'Circuit Boards', 'Battery Cells', 'Display Units',
        'Copper Wire', 'Resistors', 'Capacitors',
        'Microchips', 'Steel Frames'
    ],
    'category': [
        'Electronics', 'Electronics', 'Electronics',
        'Raw Material', 'Electronics', 'Electronics',
        'Electronics', 'Raw Material'
    ],
    'unit': ['piece','piece','piece','kg','piece','piece','piece','kg'],
    'base_demand': [500, 600, 450, 800, 1000, 900, 700, 600]
})

# ── Inventory ───────────────────────────────────────────
inventory = pd.DataFrame({
    'material_id': ['M1','M2','M3','M4','M5','M6','M7','M8'],
    'material_name': [
        'Circuit Boards', 'Battery Cells', 'Display Units',
        'Copper Wire', 'Resistors', 'Capacitors',
        'Microchips', 'Steel Frames'
    ],
    'current_stock': [300, 150, 400, 800, 1200, 950, 280, 600],
    'reorder_level': [500, 400, 600, 1000, 1500, 1200, 500, 800],
    'storage_capacity': [2000, 1500, 2500, 3000, 4000, 3500, 2000, 2500],
    'daily_consumption': [45, 60, 55, 80, 100, 90, 65, 70]
})

# ── Seasonal multipliers ────────────────────────────────
seasonal_multiplier = {
    1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95,
    5: 1.00, 6: 1.05, 7: 1.00, 8: 1.10,
    9: 1.15, 10: 1.30, 11: 1.40, 12: 1.35
}

weather_risk = {
    1: 0.1, 2: 0.1, 3: 0.05, 4: 0.05,
    5: 0.1, 6: 0.35, 7: 0.40, 8: 0.35,
    9: 0.25, 10: 0.15, 11: 0.1, 12: 0.1
}

# ── Orders (1000 rows) ──────────────────────────────────
order_records = []
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)
date_range = (end_date - start_date).days

for i in range(1000):
    supplier = random.choice(suppliers['supplier_id'].tolist())
    material = random.choice(materials['material_id'].tolist())
    order_date = start_date + timedelta(days=random.randint(0, date_range))
    month = order_date.month

    sup_row = suppliers[suppliers['supplier_id'] == supplier].iloc[0]
    mat_row = materials[materials['material_id'] == material].iloc[0]

    base_qty = mat_row['base_demand']
    season_factor = seasonal_multiplier[month]
    quantity = int(random.uniform(0.5, 1.5) * base_qty * season_factor)
    quantity = max(100, min(quantity, 2000))

    expected_days = int(sup_row['distance_km'] / 200) + random.randint(2, 5)
    expected_delivery = order_date + timedelta(days=expected_days)

    w_risk = weather_risk[month]
    delay_chance = (
        (1 - sup_row['reliability_score']) * 0.35 +
        sup_row['geopolitical_risk'] * 0.25 +
        sup_row['strike_risk'] * 0.20 +
        w_risk * 0.20
    )

    delayed = random.random() < delay_chance
    actual_delay = random.randint(1, 10) if delayed else 0
    actual_delivery = expected_delivery + timedelta(days=actual_delay)

    partial = random.random() < (1 - sup_row['reliability_score']) * 0.4
    delivered_qty = int(quantity * random.uniform(0.5, 0.9)) if partial else quantity

    quality_issue = random.random() < (1 - sup_row['quality_score']) * 0.3
    rejected_qty = int(delivered_qty * random.uniform(0.05, 0.2)) if quality_issue else 0
    accepted_qty = delivered_qty - rejected_qty

    unit_price = sup_row['price_per_unit'] * random.uniform(0.95, 1.05)
    transport_cost = sup_row['distance_km'] * sup_row['transport_cost_per_km']
    storage_cost = actual_delay * sup_row['storage_cost_per_day']
    total_cost = (accepted_qty * unit_price) + transport_cost + storage_cost

    backup_used = 1 if (delayed and actual_delay > 5) else 0
    jit_triggered = 1 if quantity <= 300 else 0

    order_records.append({
        'order_id': f'O{i+1:04d}',
        'supplier_id': supplier,
        'material_id': material,
        'quantity': quantity,
        'delivered_qty': delivered_qty,
        'accepted_qty': accepted_qty,
        'rejected_qty': rejected_qty,
        'order_date': order_date.date(),
        'expected_delivery': expected_delivery.date(),
        'actual_delivery': actual_delivery.date(),
        'delay_days': actual_delay,
        'delayed': int(delayed),
        'partial_delivery': int(partial),
        'quality_issue': int(quality_issue),
        'backup_used': int(backup_used),
        'jit_triggered': int(jit_triggered),
        'unit_price': round(unit_price, 2),
        'transport_cost': round(transport_cost, 2),
        'storage_cost': round(storage_cost, 2),
        'total_cost': round(total_cost, 2),
        'weather_risk': w_risk,
        'month': month,
        'season': 'Peak' if month in [10,11,12] else 'Normal'
    })

orders = pd.DataFrame(order_records)

suppliers.to_csv('suppliers.csv', index=False)
materials.to_csv('materials.csv', index=False)
inventory.to_csv('inventory.csv', index=False)
orders.to_csv('orders.csv', index=False)

print("✅ All datasets created successfully!")
print(f"Orders:     {len(orders)} rows")
print(f"Suppliers:  {len(suppliers)} rows")
print(f"Materials:  {len(materials)} rows")
print(f"Inventory:  {len(inventory)} rows")
print(f"\nDelayed orders:        {orders['delayed'].sum()}")
print(f"Partial deliveries:    {orders['partial_delivery'].sum()}")
print(f"Quality issues:        {orders['quality_issue'].sum()}")
print(f"Backup supplier used:  {orders['backup_used'].sum()}")
print(f"JIT triggered:         {orders['jit_triggered'].sum()}")
print(f"\nPeak season orders:    {len(orders[orders['season']=='Peak'])}")
print(f"Normal season orders:  {len(orders[orders['season']=='Normal'])}")
