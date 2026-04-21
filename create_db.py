"""
create_db.py  —  Step 1b (run after generate_data.py)
Creates a normalised SQLite database (procurement.db) from the CSV files.
This satisfies the 'build a database and populate all required tables' deliverable.
"""

import sqlite3
import pandas as pd
import os

DB_FILE = 'procurement.db'

def create_database():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()

    # ── DDL ────────────────────────────────────────────────────────────────────
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS suppliers (
        supplier_id             TEXT PRIMARY KEY,
        supplier_name           TEXT NOT NULL,
        location                TEXT,
        distance_km             REAL,
        price_per_unit          REAL,
        transport_cost_per_km   REAL,
        reliability_score       REAL,
        available_stock         INTEGER,
        storage_cost_per_day    REAL,
        geopolitical_risk       REAL,
        strike_risk             REAL,
        quality_score           REAL,
        max_capacity            INTEGER
    );

    CREATE TABLE IF NOT EXISTS materials (
        material_id   TEXT PRIMARY KEY,
        material_name TEXT NOT NULL,
        category      TEXT,
        unit          TEXT,
        base_demand   INTEGER
    );

    CREATE TABLE IF NOT EXISTS inventory (
        material_id       TEXT PRIMARY KEY REFERENCES materials(material_id),
        material_name     TEXT,
        current_stock     INTEGER,
        reorder_level     INTEGER,
        storage_capacity  INTEGER,
        daily_consumption INTEGER
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id            TEXT PRIMARY KEY,
        supplier_id         TEXT REFERENCES suppliers(supplier_id),
        material_id         TEXT REFERENCES materials(material_id),
        quantity            INTEGER,
        delivered_qty       INTEGER,
        accepted_qty        INTEGER,
        rejected_qty        INTEGER,
        order_date          TEXT,
        expected_delivery   TEXT,
        actual_delivery     TEXT,
        delay_days          INTEGER,
        delayed             INTEGER,
        partial_delivery    INTEGER,
        quality_issue       INTEGER,
        backup_used         INTEGER,
        jit_triggered       INTEGER,
        unit_price          REAL,
        transport_cost      REAL,
        storage_cost        REAL,
        total_cost          REAL,
        weather_risk        REAL,
        month               INTEGER,
        season              TEXT
    );

    CREATE TABLE IF NOT EXISTS supplier_stats (
        supplier_id         TEXT PRIMARY KEY REFERENCES suppliers(supplier_id),
        supplier_name       TEXT,
        total_orders        INTEGER,
        delayed_orders      INTEGER,
        avg_delay_days      REAL,
        avg_total_cost      REAL,
        partial_deliveries  INTEGER,
        quality_issues      INTEGER,
        backup_activations  INTEGER,
        reliability_score   REAL,
        geopolitical_risk   REAL,
        strike_risk         REAL,
        available_stock     INTEGER,
        quality_score       REAL,
        delay_rate          REAL,
        partial_rate        REAL,
        quality_issue_rate  REAL,
        health_score        REAL,
        grade               TEXT
    );

    CREATE TABLE IF NOT EXISTS forecast (
        material_id       TEXT REFERENCES materials(material_id),
        next_month        INTEGER,
        predicted_demand  INTEGER,
        PRIMARY KEY (material_id, next_month)
    );
    """)

    # ── Load CSVs → DB ─────────────────────────────────────────────────────────
    tables = ['suppliers', 'materials', 'inventory', 'orders', 'supplier_stats', 'forecast']
    for t in tables:
        df = pd.read_csv(f'{t}.csv')
        df.to_sql(t, con, if_exists='replace', index=False)
        print(f"  ✅ Loaded {len(df):,} rows → table '{t}'")

    # ── Useful views ───────────────────────────────────────────────────────────
    cur.executescript("""
    DROP VIEW IF EXISTS v_order_details;
    CREATE VIEW v_order_details AS
        SELECT
            o.order_id, o.order_date, o.season,
            s.supplier_name, s.location,
            m.material_name, m.category,
            o.quantity, o.accepted_qty, o.rejected_qty,
            o.delay_days, o.delayed, o.partial_delivery, o.quality_issue,
            o.total_cost, o.unit_price, o.transport_cost, o.storage_cost
        FROM orders o
        JOIN suppliers s ON o.supplier_id = s.supplier_id
        JOIN materials m ON o.material_id = m.material_id;

    DROP VIEW IF EXISTS v_supplier_summary;
    CREATE VIEW v_supplier_summary AS
        SELECT
            ss.supplier_name, ss.health_score, ss.grade,
            ss.delay_rate, ss.quality_issue_rate,
            ss.avg_total_cost, ss.avg_delay_days,
            s.location, s.distance_km, s.reliability_score
        FROM supplier_stats ss
        JOIN suppliers s ON ss.supplier_id = s.supplier_id;

    DROP VIEW IF EXISTS v_inventory_alert;
    CREATE VIEW v_inventory_alert AS
        SELECT
            material_name,
            current_stock,
            reorder_level,
            daily_consumption,
            CAST(current_stock AS REAL) / daily_consumption AS days_remaining,
            CASE
                WHEN current_stock < reorder_level THEN 'CRITICAL'
                WHEN current_stock < reorder_level * 1.2 THEN 'WARNING'
                ELSE 'OK'
            END AS alert_status
        FROM inventory;
    """)

    con.commit()
    con.close()
    print(f"\n✅ Database '{DB_FILE}' created with {len(tables)} tables + 3 views.")
    print("   Views: v_order_details, v_supplier_summary, v_inventory_alert")

if __name__ == '__main__':
    create_database()
