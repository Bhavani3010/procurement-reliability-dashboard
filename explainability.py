import pandas as pd
import numpy as np
import pickle
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Load once at module level ───────────────────────────
suppliers = pd.read_csv('suppliers.csv')
orders = pd.read_csv('orders.csv')
supplier_stats = pd.read_csv('supplier_stats.csv')
materials = pd.read_csv('materials.csv')
delay_model = pickle.load(open('delay_model.pkl', 'rb'))
cost_model = pickle.load(open('cost_model.pkl', 'rb'))

df = orders.merge(suppliers, on='supplier_id')

FEATURES = [
    'distance_km', 'quantity', 'price_per_unit',
    'reliability_score', 'geopolitical_risk',
    'strike_risk', 'transport_cost', 'weather_risk'
]

FEATURE_LABELS = {
    'distance_km':        'Distance (km)',
    'quantity':           'Order Quantity',
    'price_per_unit':     'Price per Unit (₹)',
    'reliability_score':  'Supplier Reliability',
    'geopolitical_risk':  'Geopolitical Risk',
    'strike_risk':        'Strike Risk',
    'transport_cost':     'Transport Cost (₹)',
    'weather_risk':       'Weather / Season Risk'
}

WEATHER_RISK = {
    1: 0.1, 2: 0.1, 3: 0.05, 4: 0.05,
    5: 0.1, 6: 0.35, 7: 0.40, 8: 0.35,
    9: 0.25, 10: 0.15, 11: 0.1, 12: 0.1
}

# ── SHAP explainer (cached) ─────────────────────────────
_explainer = None
_shap_values = None

def _extract_shap_class1(sv):
    """Normalise SHAP output to 2-D array (n_samples, n_features) for class 1."""
    arr = np.array(sv)
    if arr.ndim == 3:          # shape (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    elif arr.ndim == 2:        # shape (n_samples, n_features) — already flat
        return arr
    elif isinstance(sv, list): # old list-of-arrays format
        return np.array(sv[1])
    return arr


def _get_explainer():
    global _explainer, _shap_values
    if _explainer is None:
        X = df[FEATURES]
        _explainer = shap.TreeExplainer(delay_model)
        _shap_values = _explainer.shap_values(X)
    return _explainer, _shap_values


def get_global_importance():
    """Returns DataFrame of mean |SHAP| per feature."""
    _, sv = _get_explainer()
    arr = _extract_shap_class1(sv)
    mean_shap = np.abs(arr).mean(axis=0)
    df_imp = pd.DataFrame({
        'Feature': [FEATURE_LABELS[f] for f in FEATURES],
        'Impact':  mean_shap
    }).sort_values('Impact', ascending=False).reset_index(drop=True)
    return df_imp


def explain_single_order(supplier_id, quantity, month=6):
    """
    Returns (delay_prob, explanation_df) for a hypothetical order.
    explanation_df columns: feature, value, shap_value, impact, strength, explanation
    """
    sup = suppliers[suppliers['supplier_id'] == supplier_id].iloc[0]
    transport_cost = sup['distance_km'] * sup['transport_cost_per_km']
    w_risk = WEATHER_RISK.get(month, 0.1)

    order_feat = pd.DataFrame([{
        'distance_km':       sup['distance_km'],
        'quantity':          quantity,
        'price_per_unit':    sup['price_per_unit'],
        'reliability_score': sup['reliability_score'],
        'geopolitical_risk': sup['geopolitical_risk'],
        'strike_risk':       sup['strike_risk'],
        'transport_cost':    transport_cost,
        'weather_risk':      w_risk
    }])

    prob = delay_model.predict_proba(order_feat)[0][1]

    explainer, _ = _get_explainer()
    sv = explainer.shap_values(order_feat)
    order_shap = _extract_shap_class1(sv)[0]

    rows = []
    for feat, shap_val in zip(FEATURES, order_shap):
        actual = order_feat[feat].values[0]
        impact = 'increases' if shap_val > 0 else 'reduces'
        strength = (
            'strongly'   if abs(shap_val) > 0.08
            else 'moderately' if abs(shap_val) > 0.03
            else 'slightly'
        )
        rows.append({
            'feature':     FEATURE_LABELS[feat],
            'value':       actual,
            'shap_value':  shap_val,
            'impact':      impact,
            'strength':    strength,
            'explanation': f"{FEATURE_LABELS[feat]} = {actual:.2f} → {strength} {impact} delay risk"
        })

    exp_df = pd.DataFrame(rows).sort_values('shap_value', ascending=False)
    return prob, exp_df


def generate_supplier_report(supplier_id):
    """Returns a full explainable report dict for a supplier."""
    sup = suppliers[suppliers['supplier_id'] == supplier_id].iloc[0]
    stats_rows = supplier_stats[supplier_stats['supplier_id'] == supplier_id]
    if len(stats_rows) == 0:
        return None
    stats = stats_rows.iloc[0]

    factors = []

    rel_contribution = sup['reliability_score'] * 35
    factors.append({
        'factor': 'Reliability Score',
        'value': sup['reliability_score'],
        'contribution': round(rel_contribution, 2),
        'max': 35,
        'explanation': (
            f"Reliability {sup['reliability_score']:.0%} → "
            f"contributes {rel_contribution:.1f}/35 points. "
            + ("Excellent ✅" if sup['reliability_score'] > 0.85
               else "Needs improvement ⚠️" if sup['reliability_score'] > 0.65
               else "Poor ❌")
        )
    })

    ontime_contribution = (1 - stats['delay_rate']) * 25
    factors.append({
        'factor': 'On-Time Delivery Rate',
        'value': round(1 - stats['delay_rate'], 3),
        'contribution': round(ontime_contribution, 2),
        'max': 25,
        'explanation': (
            f"On-time rate {(1-stats['delay_rate']):.0%} → "
            f"contributes {ontime_contribution:.1f}/25 points. "
            + ("Excellent ✅" if stats['delay_rate'] < 0.2
               else "Acceptable ⚠️" if stats['delay_rate'] < 0.4
               else "Too many delays ❌")
        )
    })

    geo_contribution = (1 - sup['geopolitical_risk']) * 15
    factors.append({
        'factor': 'Geopolitical Stability',
        'value': round(1 - sup['geopolitical_risk'], 3),
        'contribution': round(geo_contribution, 2),
        'max': 15,
        'explanation': (
            f"Geopolitical risk {sup['geopolitical_risk']:.0%} → "
            f"contributes {geo_contribution:.1f}/15 points. "
            + ("Low risk ✅" if sup['geopolitical_risk'] < 0.2
               else "Medium risk ⚠️" if sup['geopolitical_risk'] < 0.4
               else "High risk ❌")
        )
    })

    strike_contribution = (1 - sup['strike_risk']) * 15
    factors.append({
        'factor': 'Strike Risk',
        'value': round(1 - sup['strike_risk'], 3),
        'contribution': round(strike_contribution, 2),
        'max': 15,
        'explanation': (
            f"Strike risk {sup['strike_risk']:.0%} → "
            f"contributes {strike_contribution:.1f}/15 points. "
            + ("Low risk ✅" if sup['strike_risk'] < 0.1
               else "Medium risk ⚠️" if sup['strike_risk'] < 0.2
               else "High risk ❌")
        )
    })

    quality_contribution = sup['quality_score'] * 10
    factors.append({
        'factor': 'Quality Score',
        'value': sup['quality_score'],
        'contribution': round(quality_contribution, 2),
        'max': 10,
        'explanation': (
            f"Quality score {sup['quality_score']:.0%} → "
            f"contributes {quality_contribution:.1f}/10 points. "
            + ("High quality ✅" if sup['quality_score'] > 0.88
               else "Average ⚠️" if sup['quality_score'] > 0.72
               else "Low quality ❌")
        )
    })

    return {
        'supplier_name': sup['supplier_name'],
        'grade': stats['grade'],
        'health_score': stats['health_score'],
        'factors': factors
    }


def explain_cost(supplier_id, quantity, avg_delay_days=2):
    """Full cost breakdown with explanations."""
    sup = suppliers[suppliers['supplier_id'] == supplier_id].iloc[0]
    unit_cost = quantity * sup['price_per_unit']
    transport = sup['distance_km'] * sup['transport_cost_per_km']
    storage = avg_delay_days * sup['storage_cost_per_day']
    total = unit_cost + transport + storage

    return {
        'supplier': sup['supplier_name'],
        'quantity': quantity,
        'unit_cost': round(unit_cost, 2),
        'transport_cost': round(transport, 2),
        'storage_cost': round(storage, 2),
        'total': round(total, 2),
        'unit_pct':      round(unit_cost / total * 100, 1),
        'transport_pct': round(transport / total * 100, 1),
        'storage_pct':   round(storage / total * 100, 1),
        'explanation': (
            f"Total ₹{total:,.0f} = "
            f"Material {unit_cost/total*100:.0f}% + "
            f"Transport {transport/total*100:.0f}% + "
            f"Storage {storage/total*100:.0f}%"
        )
    }


def explain_recommendation(material_id, quantity_needed, month=6):
    """Ranks all suppliers with reasoning for a given material + quantity."""
    w_risk = WEATHER_RISK.get(month, 0.1)
    rows = []

    for _, sup in suppliers.iterrows():
        transport = sup['distance_km'] * sup['transport_cost_per_km']
        total_cost = (quantity_needed * sup['price_per_unit']) + transport

        order_feat = pd.DataFrame([{
            'distance_km':       sup['distance_km'],
            'quantity':          quantity_needed,
            'price_per_unit':    sup['price_per_unit'],
            'reliability_score': sup['reliability_score'],
            'geopolitical_risk': sup['geopolitical_risk'],
            'strike_risk':       sup['strike_risk'],
            'transport_cost':    transport,
            'weather_risk':      w_risk
        }])

        delay_prob = delay_model.predict_proba(order_feat)[0][1]

        stats_row = supplier_stats[supplier_stats['supplier_id'] == sup['supplier_id']]
        health = stats_row['health_score'].values[0] if len(stats_row) > 0 else 50

        max_cost = suppliers.apply(
            lambda r: quantity_needed * r['price_per_unit'] + r['distance_km'] * r['transport_cost_per_km'],
            axis=1
        ).max()
        cost_score  = (1 - total_cost / max_cost) * 30
        delay_score = (1 - delay_prob) * 40
        h_score     = (health / 100) * 30
        final_score = cost_score + delay_score + h_score

        rows.append({
            'supplier_id':      sup['supplier_id'],
            'supplier_name':    sup['supplier_name'],
            'total_cost':       round(total_cost),
            'delay_probability': round(delay_prob * 100, 1),
            'health_score':     round(health, 1),
            'cost_score':       round(cost_score, 1),
            'delay_score':      round(delay_score, 1),
            'h_score':          round(h_score, 1),
            'final_score':      round(final_score, 2),
            'can_fulfill':      sup['available_stock'] >= quantity_needed,
            'reason': (
                f"Cost score: {cost_score:.1f}/30 | "
                f"Reliability score: {delay_score:.1f}/40 | "
                f"Health score: {h_score:.1f}/30"
            )
        })

    rec_df = pd.DataFrame(rows).sort_values('final_score', ascending=False).reset_index(drop=True)
    return rec_df
