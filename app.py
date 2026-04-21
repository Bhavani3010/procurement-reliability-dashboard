import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Smart Procurement Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric { border-left: 3px solid #2d6a9f; padding-left: 10px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: bold; }
    .section-header {
        background: #f0f4fa; padding: 8px 16px; border-radius: 8px;
        border-left: 4px solid #2d6a9f; margin: 12px 0 8px 0; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────
@st.cache_data
def load_data():
    orders         = pd.read_csv('orders.csv')
    suppliers      = pd.read_csv('suppliers.csv')
    inventory      = pd.read_csv('inventory.csv')
    supplier_stats = pd.read_csv('supplier_stats.csv')
    forecast       = pd.read_csv('forecast.csv')
    materials      = pd.read_csv('materials.csv')
    return orders, suppliers, inventory, supplier_stats, forecast, materials

@st.cache_data
def load_db_query(query):
    con = sqlite3.connect('procurement.db')
    df = pd.read_sql_query(query, con)
    con.close()
    return df

orders, suppliers, inventory, supplier_stats, forecast, materials = load_data()
delay_model    = pickle.load(open('delay_model.pkl',   'rb'))
overrun_model  = pickle.load(open('overrun_model.pkl', 'rb'))
cost_model     = pickle.load(open('cost_model.pkl',    'rb'))

orders['order_date'] = pd.to_datetime(orders['order_date'])
df = orders.merge(suppliers, on='supplier_id').merge(materials, on='material_id')

WEATHER_RISK = {
    1:0.1,2:0.1,3:0.05,4:0.05,5:0.1,
    6:0.35,7:0.40,8:0.35,9:0.25,10:0.15,11:0.1,12:0.1
}
FEATURES = [
    'distance_km','quantity','price_per_unit',
    'reliability_score','geopolitical_risk',
    'strike_risk','transport_cost','weather_risk'
]
OVERRUN_FEATURES = [
    'distance_km','quantity','reliability_score',
    'geopolitical_risk','strike_risk','weather_risk',
    'delay_days','partial_delivery','quality_issue'
]

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/supply-chain.png", width=72)
st.sidebar.title("ProcureAI Dashboard")
st.sidebar.caption("P2 — Procurement Reliability | PES University")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("📂 Navigate", [
    "🏠 Overview",
    "📦 Supplier Analysis",
    "📊 Procurement Orders",
    "🤖 ML Predictions",
    "⚠️ Alerts & Recommendations",
    "📈 Demand Forecast",
    "🔍 Explainability Center",
    "🗄️ Database & Schema"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**🔽 Filters**")
supplier_filter = st.sidebar.multiselect(
    "Suppliers", suppliers['supplier_name'].tolist(),
    default=suppliers['supplier_name'].tolist()
)
material_filter = st.sidebar.multiselect(
    "Materials", materials['material_name'].tolist(),
    default=materials['material_name'].tolist()
)
date_range = st.sidebar.date_input(
    "Date Range",
    value=[orders['order_date'].min(), orders['order_date'].max()]
)

filtered_df = df[
    df['supplier_name'].isin(supplier_filter) &
    df['material_name'].isin(material_filter)
]
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['order_date'] >= pd.to_datetime(date_range[0])) &
        (filtered_df['order_date'] <= pd.to_datetime(date_range[1]))
    ]

# ════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏭 Smart Procurement Reliability Dashboard")
    st.markdown("**Electronics Manufacturing Co. Ltd** — AI-Powered Procurement Intelligence | P2: Procurement Reliability")
    st.markdown("---")

    total          = len(filtered_df)
    delayed        = int(filtered_df['delayed'].sum())
    ontime_pct     = ((total - delayed) / total * 100) if total > 0 else 0
    avg_delay      = filtered_df['delay_days'].mean()
    total_cost     = filtered_df['total_cost'].sum()
    partial        = int(filtered_df['partial_delivery'].sum())
    quality_issues = int(filtered_df['quality_issue'].sum())

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("📦 Total Orders",     total)
    c2.metric("✅ On-Time %",        f"{ontime_pct:.1f}%")
    c3.metric("⏱️ Avg Delay",        f"{avg_delay:.1f} days")
    c4.metric("❌ Delayed",          delayed)
    c5.metric("📉 Partial Delivery", partial)
    c6.metric("⚠️ Quality Issues",   quality_issues)
    c7.metric("💰 Total Cost",       f"₹{total_cost/1e6:.1f}M")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 On-Time vs Delayed Orders")
        pie = pd.DataFrame({'Status':['On-Time','Delayed'], 'Count':[total-delayed, delayed]})
        fig = px.pie(pie, values='Count', names='Status',
                     color_discrete_map={'On-Time':'#00CC96','Delayed':'#EF553B'}, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("📈 Monthly Procurement Spend")
        filtered_df = filtered_df.copy()
        filtered_df['month_label'] = filtered_df['order_date'].dt.to_period('M').astype(str)
        monthly = filtered_df.groupby('month_label')['total_cost'].sum().reset_index()
        fig2 = px.area(monthly, x='month_label', y='total_cost',
                       color_discrete_sequence=['#636EFA'], markers=True)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Supplier Health Scores")
        fig3 = px.bar(supplier_stats.sort_values('health_score', ascending=False),
                      x='supplier_name', y='health_score', color='grade',
                      color_discrete_map={
                          'A - Low Risk':'#00CC96','B - Medium Risk':'#FFA15A',
                          'C - High Risk':'#EF553B','D - Very High Risk':'#AB63FA'
                      })
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.subheader("📋 Delivery Performance Breakdown")
        perf = pd.DataFrame({
            'Status': ['On-Time','Delayed','Partial Delivery','Quality Issue'],
            'Count':  [total-delayed, delayed, partial, quality_issues]
        })
        fig4 = px.bar(perf, x='Status', y='Count', color='Status',
                      color_discrete_sequence=['#00CC96','#EF553B','#FFA15A','#AB63FA'])
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📅 Peak Season vs Normal Season Analysis")
    season_df = filtered_df.groupby('season').agg(
        Orders=('order_id','count'),
        Avg_Cost=('total_cost','mean'),
        Delay_Rate=('delayed','mean'),
        Total_Cost=('total_cost','sum')
    ).reset_index()
    fig5 = px.bar(season_df, x='season', y=['Orders','Delay_Rate'],
                  barmode='group', title='Orders & Delay Rate by Season')
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 2 — SUPPLIER ANALYSIS
# ════════════════════════════════════════════════════════
elif page == "📦 Supplier Analysis":
    st.title("📦 Supplier Analysis")
    st.markdown("---")

    st.subheader("🏅 Supplier Scorecards")
    cols = st.columns(5)
    for i, (_, row) in enumerate(supplier_stats.iterrows()):
        col = cols[i % 5]
        grade = row['grade']
        icon  = "🟢" if "A" in grade else "🟡" if "B" in grade else "🔴"
        col.metric(row['supplier_name'], f"{row['health_score']:.1f}/100")
        col.caption(f"{icon} {grade}")
        col.caption(f"Delay: {row['delay_rate']*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⏱️ Delay Rate by Supplier")
        fig = px.bar(supplier_stats.sort_values('delay_rate', ascending=False),
                     x='supplier_name', y='delay_rate',
                     color='delay_rate', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💰 Average Procurement Cost by Supplier")
        fig2 = px.bar(supplier_stats.sort_values('avg_total_cost', ascending=False),
                      x='supplier_name', y='avg_total_cost', color='supplier_name')
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📦 Partial Delivery Rate")
        fig3 = px.bar(supplier_stats, x='supplier_name', y='partial_rate',
                      color='partial_rate', color_continuous_scale='Oranges')
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.subheader("🔬 Quality Issue Rate")
        fig4 = px.bar(supplier_stats, x='supplier_name', y='quality_issue_rate',
                      color='quality_issue_rate', color_continuous_scale='Purples')
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📡 Supplier Risk Radar")
    sel_radar = st.multiselect("Select suppliers for radar",
                               suppliers['supplier_name'].tolist(),
                               default=suppliers['supplier_name'].tolist()[:4])
    radar_data = supplier_stats[supplier_stats['supplier_name'].isin(sel_radar)]
    categories = ['Reliability', 'On-Time Rate', 'Geo Safety', 'Strike Safety', 'Quality']
    fig_radar = go.Figure()
    for _, row in radar_data.iterrows():
        s = suppliers[suppliers['supplier_id'] == row['supplier_id']].iloc[0]
        vals = [
            s['reliability_score']*100,
            (1 - row['delay_rate'])*100,
            (1 - s['geopolitical_risk'])*100,
            (1 - s['strike_risk'])*100,
            s['quality_score']*100
        ]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill='toself', name=row['supplier_name']
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0,100])), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("📋 Full Supplier Comparison Table")
    display = supplier_stats[[
        'supplier_name','health_score','grade','delay_rate',
        'partial_rate','quality_issue_rate','avg_delay_days',
        'avg_total_cost','available_stock','backup_activations'
    ]].copy()
    display.columns = [
        'Supplier','Health Score','Grade','Delay Rate',
        'Partial Rate','Quality Issue Rate','Avg Delay Days',
        'Avg Cost (₹)','Available Stock','Backup Activations'
    ]
    st.dataframe(display.style.background_gradient(subset=['Health Score'], cmap='Greens')
                              .background_gradient(subset=['Delay Rate'], cmap='Reds_r'),
                 use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 3 — PROCUREMENT ORDERS
# ════════════════════════════════════════════════════════
elif page == "📊 Procurement Orders":
    st.title("📊 Procurement Orders")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💸 Stacked Cost Breakdown by Supplier")
        cost_data = filtered_df.groupby('supplier_name').agg(
            unit_cost=('unit_price','mean'),
            transport=('transport_cost','mean'),
            storage=('storage_cost','mean')
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Unit Cost',   x=cost_data['supplier_name'], y=cost_data['unit_cost'],  marker_color='#636EFA'))
        fig.add_trace(go.Bar(name='Transport',   x=cost_data['supplier_name'], y=cost_data['transport'],  marker_color='#EF553B'))
        fig.add_trace(go.Bar(name='Storage',     x=cost_data['supplier_name'], y=cost_data['storage'],    marker_color='#FFA15A'))
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("📦 Quantity Split by Material")
        qty = filtered_df.groupby('material_name')['quantity'].sum().reset_index()
        fig2 = px.pie(qty, values='quantity', names='material_name', hole=0.3)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📆 Cost Over Time — Heatmap")
    filtered_df = filtered_df.copy()
    filtered_df['month_label'] = filtered_df['order_date'].dt.to_period('M').astype(str)
    heat = filtered_df.groupby(['supplier_name','month_label'])['total_cost'].sum().reset_index()
    heat_pivot = heat.pivot(index='supplier_name', columns='month_label', values='total_cost').fillna(0)
    fig_heat = px.imshow(heat_pivot, aspect='auto', color_continuous_scale='Blues',
                         title='Monthly Spend by Supplier (₹)')
    fig_heat.update_xaxes(tickangle=45)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("🗂️ All Orders (filterable)")
    cols_show = ['order_id','supplier_name','material_name','quantity',
                 'order_date','expected_delivery','actual_delivery',
                 'delay_days','delayed','partial_delivery','quality_issue',
                 'unit_price','transport_cost','storage_cost','total_cost','season']
    st.dataframe(filtered_df[cols_show], use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 4 — ML PREDICTIONS
# ════════════════════════════════════════════════════════
elif page == "🤖 ML Predictions":
    st.title("🤖 ML Predictions")
    st.markdown("---")

    tab_delay, tab_overrun = st.tabs(["⚠️ Delay Prediction", "💸 Cost Overrun Prediction"])

    with tab_delay:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔮 Predict Order Delay Risk")
            sel_supplier = st.selectbox("Select Supplier", suppliers['supplier_name'].tolist())
            quantity     = st.slider("Order Quantity", 100, 2000, 500)
            sel_month    = st.selectbox("Order Month", list(range(1,13)),
                                        format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun',
                                                               'Jul','Aug','Sep','Oct','Nov','Dec'][m-1])

            sup_row = suppliers[suppliers['supplier_name'] == sel_supplier].iloc[0]
            transport_cost = sup_row['distance_km'] * sup_row['transport_cost_per_km']
            w_risk = WEATHER_RISK[sel_month]

            features_input = np.array([[
                sup_row['distance_km'], quantity, sup_row['price_per_unit'],
                sup_row['reliability_score'], sup_row['geopolitical_risk'],
                sup_row['strike_risk'], transport_cost, w_risk
            ]])

            delay_prob = delay_model.predict_proba(features_input)[0][1]
            pred_cost  = cost_model.predict([[
                sup_row['distance_km'], quantity, sup_row['price_per_unit'],
                transport_cost, sup_row['storage_cost_per_day'] * 2
            ]])[0]

            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("⚠️ Delay Probability", f"{delay_prob*100:.1f}%")
            m2.metric("💰 Predicted Cost",    f"₹{pred_cost:,.0f}")

            if delay_prob > 0.6:
                st.error("🔴 HIGH DELAY RISK — Use backup supplier")
            elif delay_prob > 0.35:
                st.warning("🟡 MEDIUM RISK — Monitor closely")
            else:
                st.success("🟢 LOW RISK — Safe to proceed")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=delay_prob * 100,
                title={'text': "Delay Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#EF553B' if delay_prob>0.6 else '#FFA15A' if delay_prob>0.35 else '#00CC96'},
                    'steps': [
                        {'range':[0,35],   'color':'#d4edda'},
                        {'range':[35,60],  'color':'#fff3cd'},
                        {'range':[60,100], 'color':'#f8d7da'}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.subheader("🏆 Backup Supplier Recommendations")
            backup = supplier_stats[
                supplier_stats['supplier_name'] != sel_supplier
            ].sort_values('health_score', ascending=False)[
                ['supplier_name','health_score','grade','available_stock','delay_rate']
            ].head(4)
            st.dataframe(backup, use_container_width=True)

            st.subheader("📊 Delay Risk — All Suppliers")
            probs = []
            for _, r in suppliers.iterrows():
                tc = r['distance_km'] * r['transport_cost_per_km']
                f  = np.array([[r['distance_km'], quantity, r['price_per_unit'],
                                r['reliability_score'], r['geopolitical_risk'],
                                r['strike_risk'], tc, w_risk]])
                p = delay_model.predict_proba(f)[0][1]
                probs.append({'Supplier': r['supplier_name'], 'Delay %': round(p*100,1)})
            prob_df = pd.DataFrame(probs).sort_values('Delay %', ascending=False)
            fig_bar = px.bar(prob_df, x='Supplier', y='Delay %',
                             color='Delay %', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_overrun:
        st.subheader("💸 Cost Overrun Prediction")
        st.markdown("""
        This model predicts **extra cost beyond the baseline** (material + transport).
        Overrun is caused by storage charges, delay penalties, and quality rework costs.
        Uses **Gradient Boosting Regressor** trained on procurement risk factors.
        """)

        col1, col2 = st.columns(2)
        with col1:
            ov_supplier = st.selectbox("Supplier", suppliers['supplier_name'].tolist(), key='ov_sup')
            ov_qty      = st.slider("Order Quantity", 100, 2000, 500, key='ov_qty')
            ov_month    = st.selectbox("Month", list(range(1,13)), key='ov_month',
                                       format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun',
                                                              'Jul','Aug','Sep','Oct','Nov','Dec'][m-1])
            ov_delay_days     = st.slider("Expected Delay Days (if any)", 0, 20, 2, key='ov_delay')
            ov_partial        = st.checkbox("Partial Delivery Risk?", key='ov_partial')
            ov_quality        = st.checkbox("Quality Issue Risk?",    key='ov_quality')

            sup_ov = suppliers[suppliers['supplier_name'] == ov_supplier].iloc[0]
            ov_features = np.array([[
                sup_ov['distance_km'], ov_qty, sup_ov['reliability_score'],
                sup_ov['geopolitical_risk'], sup_ov['strike_risk'],
                WEATHER_RISK[ov_month],
                ov_delay_days, int(ov_partial), int(ov_quality)
            ]])

            predicted_overrun = max(overrun_model.predict(ov_features)[0], 0)
            baseline = ov_qty * sup_ov['price_per_unit'] + sup_ov['distance_km'] * sup_ov['transport_cost_per_km']
            total_predicted   = baseline + predicted_overrun

            st.metric("📦 Baseline Cost",      f"₹{baseline:,.0f}")
            st.metric("📈 Predicted Overrun",  f"₹{predicted_overrun:,.0f}")
            st.metric("💰 Total Predicted Cost",f"₹{total_predicted:,.0f}")

            fig_ov = px.pie(
                values=[baseline, predicted_overrun],
                names=['Baseline Cost','Cost Overrun'],
                hole=0.4,
                color_discrete_sequence=['#636EFA','#EF553B'],
                title='Baseline vs Overrun'
            )
            st.plotly_chart(fig_ov, use_container_width=True)

        with col2:
            st.subheader("📊 Overrun Comparison — All Suppliers")
            overrun_rows = []
            for _, r in suppliers.iterrows():
                f = np.array([[
                    r['distance_km'], ov_qty, r['reliability_score'],
                    r['geopolitical_risk'], r['strike_risk'],
                    WEATHER_RISK[ov_month], ov_delay_days,
                    int(ov_partial), int(ov_quality)
                ]])
                ov = max(overrun_model.predict(f)[0], 0)
                bl = ov_qty * r['price_per_unit'] + r['distance_km'] * r['transport_cost_per_km']
                overrun_rows.append({
                    'Supplier':  r['supplier_name'],
                    'Baseline':  round(bl),
                    'Overrun':   round(ov),
                    'Total':     round(bl + ov)
                })
            ov_df = pd.DataFrame(overrun_rows).sort_values('Total')
            fig_ov2 = px.bar(ov_df, x='Supplier', y=['Baseline','Overrun'],
                             barmode='stack', title='Total Cost = Baseline + Predicted Overrun',
                             color_discrete_map={'Baseline':'#636EFA','Overrun':'#EF553B'})
            st.plotly_chart(fig_ov2, use_container_width=True)
            st.dataframe(ov_df, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 5 — ALERTS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════
elif page == "⚠️ Alerts & Recommendations":
    st.title("⚠️ Alerts & Recommendations")
    st.markdown("---")

    st.subheader("📦 Inventory Stockout Alerts")
    for _, row in inventory.iterrows():
        days_left = int(row['current_stock'] / row['daily_consumption'])
        if row['current_stock'] < row['reorder_level']:
            st.error(
                f"🔴 **{row['material_name']}** — "
                f"Stock: {row['current_stock']} (Reorder at: {row['reorder_level']}) | "
                f"Only **{days_left} days** left — **ORDER NOW**"
            )
        else:
            st.success(
                f"🟢 **{row['material_name']}** — "
                f"Stock OK: {row['current_stock']} | ~{days_left} days remaining"
            )

    st.markdown("---")
    st.subheader("🚨 High Risk Supplier Alerts")
    high_risk = supplier_stats[supplier_stats['delay_rate'] > 0.35]
    if len(high_risk) > 0:
        for _, row in high_risk.iterrows():
            st.warning(
                f"⚠️ **{row['supplier_name']}** — "
                f"Delay Rate: {row['delay_rate']*100:.1f}% | "
                f"Health Score: {row['health_score']:.1f} | "
                f"Grade: {row['grade']}"
            )
    else:
        st.success("✅ No high risk suppliers detected")

    st.markdown("---")
    st.subheader("💡 Smart JIT Reorder Suggestions")
    # Get best supplier: merge only the health_score from supplier_stats, keep all suppliers cols
    best_sup_id = supplier_stats.sort_values('health_score', ascending=False).iloc[0]['supplier_id']
    best_sup = suppliers[suppliers['supplier_id'] == best_sup_id].iloc[0]
    best_sup_name = best_sup['supplier_name']
    best_sup_dist = best_sup['distance_km']

    for _, row in inventory.iterrows():
        days_left = int(row['current_stock'] / row['daily_consumption'])
        lead_time = int(best_sup_dist / 200) + 2
        if days_left <= lead_time + 1:
            st.error(f"🔴 **{row['material_name']}** — Order TODAY via **{best_sup_name}** (lead ~{lead_time}d, stock {days_left}d)")
        elif days_left <= lead_time + 5:
            st.warning(f"🟡 **{row['material_name']}** — Order within {days_left - lead_time} days via **{best_sup_name}**")
        else:
            st.info(f"🔵 **{row['material_name']}** — Reorder in ~{days_left - lead_time} days")

    st.markdown("---")
    st.subheader("🔄 What-If Scenario: If Primary Supplier Fails?")
    what_if_sup = st.selectbox("Select Primary Supplier", suppliers['supplier_name'].tolist())
    what_if_qty = st.slider("Required Quantity", 100, 2000, 600)

    # Filter supplier_stats for alternatives, then look up available_stock from suppliers directly
    alt_stats = supplier_stats[supplier_stats['supplier_name'] != what_if_sup].copy()
    alt_sup   = suppliers[suppliers['supplier_name'] != what_if_sup][['supplier_id','supplier_name','available_stock']].copy()
    alternatives = alt_stats[['supplier_id','health_score','grade','delay_rate']].merge(
        alt_sup, on='supplier_id'
    )

    viable = alternatives[alternatives['available_stock'] >= what_if_qty].sort_values(
        'health_score', ascending=False).head(3)

    if len(viable) > 0:
        st.success(f"✅ If **{what_if_sup}** fails, best alternatives:")
        st.dataframe(viable[['supplier_name','health_score','grade','available_stock','delay_rate']],
                     use_container_width=True)
    else:
        st.error("❌ No single supplier can fulfil this quantity. Combination ordering recommended.")

    remaining = what_if_qty
    combo_text = []
    for _, r in alternatives.sort_values('health_score', ascending=False).iterrows():
        if remaining <= 0: break
        take = min(r['available_stock'], remaining)
        combo_text.append(f"• **{r['supplier_name']}** → {take} units (stock: {r['available_stock']})")
        remaining -= take
    st.markdown("**📦 Combination Order Split:**")
    for t in combo_text:
        st.markdown(t)
    if remaining <= 0:
        st.success("✅ Full quantity covered through combination ordering!")
    else:
        st.warning(f"⚠️ Only {what_if_qty - remaining} units coverable across all suppliers.")

# ════════════════════════════════════════════════════════
# PAGE 6 — DEMAND FORECAST
# ════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecast")
    st.markdown("---")

    forecast_display = forecast.merge(materials, on='material_id')
    next_month_fc = forecast_display[forecast_display['months_ahead'] == 1]

    st.subheader("🔮 Predicted Demand — Next Month")
    cols = st.columns(len(next_month_fc))
    for i, (_, row) in enumerate(next_month_fc.iterrows()):
        cols[i].metric(row['material_name'], f"{int(row['predicted_demand'])} units")
        cols[i].caption(f"Trend: {int(row['trend_component'])} | Season x{row['seasonal_factor']:.2f}")

    st.markdown("---")
    st.subheader("📊 3-Month Forecast per Material")
    fc3 = forecast_display.copy()
    fc3['month_name'] = fc3['next_month'].apply(
        lambda m: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]
    )
    fig_fc3 = px.bar(fc3, x='material_name', y='predicted_demand', color='months_ahead',
                     barmode='group', title='1, 2 & 3-Month Demand Forecast by Material',
                     labels={'months_ahead': 'Months Ahead'},
                     color_continuous_scale='Blues')
    st.plotly_chart(fig_fc3, use_container_width=True)

    orders_m = orders.copy()
    orders_m['month_label'] = orders_m['order_date'].dt.to_period('M').astype(str)
    hist = orders_m.merge(materials, on='material_id')
    monthly_mat = hist.groupby(['month_label','material_name'])['quantity'].sum().reset_index()

    st.subheader("📈 Historical Demand by Material")
    fig = px.line(monthly_mat, x='month_label', y='quantity',
                  color='material_name', markers=True)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏔️ Peak vs Normal Season Demand")
        season_demand = orders.merge(materials, on='material_id').groupby(
            ['season','material_name'])['quantity'].sum().reset_index()
        fig2 = px.bar(season_demand, x='material_name', y='quantity', color='season',
                      barmode='group', color_discrete_map={'Peak':'#EF553B','Normal':'#636EFA'})
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("📋 Seasonal Factor by Material")
        st.dataframe(
            fc3[fc3['months_ahead']==1][['material_name','next_month','predicted_demand',
                                         'trend_component','seasonal_factor']].rename(
                columns={'material_name':'Material','next_month':'Month',
                         'predicted_demand':'Predicted','trend_component':'Trend',
                         'seasonal_factor':'Seasonal Factor'}
            ), use_container_width=True
        )

# ════════════════════════════════════════════════════════
# PAGE 7 — EXPLAINABILITY CENTER
# ════════════════════════════════════════════════════════
elif page == "🔍 Explainability Center":
    st.title("🔍 Explainability Center")
    st.markdown("Understand **WHY** the system makes every prediction.")
    st.markdown("---")

    from explainability import (
        get_global_importance, explain_single_order,
        generate_supplier_report, explain_cost, explain_recommendation,
        WEATHER_RISK as WR
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Global Feature Importance",
        "📦 Order Explanation",
        "🏅 Supplier Report Card",
        "💰 Cost Breakdown",
        "💡 Why This Supplier?"
    ])

    with tab1:
        st.subheader("🌍 What Causes Delays the Most?")
        st.markdown("SHAP (SHapley Additive exPlanations) measures each factor's real contribution to the ML prediction.")
        with st.spinner("Computing SHAP values..."):
            importance_df = get_global_importance()
        fig = px.bar(importance_df, x='Impact', y='Feature', orientation='h',
                     color='Impact', color_continuous_scale='Reds',
                     title='Mean |SHAP| — Feature Importance for Delay Prediction')
        st.plotly_chart(fig, use_container_width=True)
        for _, row in importance_df.iterrows():
            if row['Impact'] > importance_df['Impact'].median():
                st.error(f"🔴 **{row['Feature']}** — High impact (SHAP={row['Impact']:.4f})")
            else:
                st.info(f"🔵 **{row['Feature']}** — Lower impact (SHAP={row['Impact']:.4f})")

    with tab2:
        st.subheader("📦 Why Will This Order Be Delayed?")
        col1, col2, col3 = st.columns(3)
        with col1: sel_sup = st.selectbox("Supplier", suppliers['supplier_name'].tolist(), key='exp_sup')
        with col2: qty = st.slider("Quantity", 100, 2000, 500, key='exp_qty')
        with col3:
            sel_month = st.selectbox("Month", list(range(1,13)), key='exp_month',
                                     format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun',
                                                            'Jul','Aug','Sep','Oct','Nov','Dec'][m-1])
        sup_id = suppliers[suppliers['supplier_name']==sel_sup]['supplier_id'].values[0]
        with st.spinner("Analysing order..."):
            prob, exp_df = explain_single_order(sup_id, qty, sel_month)
        c1,c2,c3 = st.columns(3)
        c1.metric("⚠️ Delay Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Level", "🔴 HIGH" if prob>0.6 else "🟡 MEDIUM" if prob>0.35 else "🟢 LOW")
        c3.metric("Action", "Use Backup ⚠️" if prob>0.6 else "Safe to Order ✅")
        for _, row in exp_df.iterrows():
            if row['shap_value'] > 0.05:   st.error(f"🔴 {row['explanation']}")
            elif row['shap_value'] > 0:    st.warning(f"🟡 {row['explanation']}")
            else:                          st.success(f"🟢 {row['explanation']}")
        fig = px.bar(exp_df, x='shap_value', y='feature', orientation='h',
                     color='shap_value', color_continuous_scale='RdYlGn_r',
                     title='SHAP Values — Positive = increases delay risk')
        fig.add_vline(x=0, line_dash='dash', line_color='black')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🏅 Supplier Report Card — Why This Grade?")
        sel_sup2 = st.selectbox("Select Supplier", suppliers['supplier_name'].tolist(), key='rc_sup')
        sup_id2  = suppliers[suppliers['supplier_name']==sel_sup2]['supplier_id'].values[0]
        report   = generate_supplier_report(sup_id2)
        if report:
            c1,c2 = st.columns(2)
            c1.metric("Health Score", f"{report['health_score']:.1f}/100")
            icon = "🟢" if "A" in report['grade'] else "🟡" if "B" in report['grade'] else "🔴"
            c2.metric("Grade", f"{icon} {report['grade']}")
            factor_df = pd.DataFrame(report['factors'])
            fig = px.bar(factor_df, x='factor', y=['contribution','max'],
                         barmode='group', title='Actual Score vs Maximum Possible',
                         color_discrete_map={'contribution':'#00CC96','max':'#EF553B'})
            st.plotly_chart(fig, use_container_width=True)
            for f in report['factors']:
                pct = f['contribution'] / f['max'] * 100
                if pct >= 75:  st.success(f"✅ **{f['factor']}**: {f['explanation']}")
                elif pct >= 50: st.warning(f"⚠️ **{f['factor']}**: {f['explanation']}")
                else:           st.error(f"❌ **{f['factor']}**: {f['explanation']}")

    with tab4:
        st.subheader("💰 Where Is Your Procurement Money Going?")
        col1, col2 = st.columns(2)
        with col1: sel_sup3 = st.selectbox("Supplier", suppliers['supplier_name'].tolist(), key='cost_sup')
        with col2: qty3 = st.slider("Quantity", 100, 2000, 500, key='cost_qty')
        sup_id3 = suppliers[suppliers['supplier_name']==sel_sup3]['supplier_id'].values[0]
        cost = explain_cost(sup_id3, qty3)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📦 Material",  f"₹{cost['unit_cost']:,.0f}")
        c2.metric("🚚 Transport", f"₹{cost['transport_cost']:,.0f}")
        c3.metric("🏭 Storage",   f"₹{cost['storage_cost']:,.0f}")
        c4.metric("💰 Total",     f"₹{cost['total']:,.0f}")
        fig = px.pie(
            values=[cost['unit_cost'], cost['transport_cost'], cost['storage_cost']],
            names=['Material','Transport','Storage'], hole=0.35,
            color_discrete_sequence=['#636EFA','#EF553B','#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"📌 {cost['explanation']}")
        all_costs = []
        for _, s in suppliers.iterrows():
            c = explain_cost(s['supplier_id'], qty3)
            all_costs.append({'Supplier':s['supplier_name'],'Material':c['unit_cost'],
                              'Transport':c['transport_cost'],'Storage':c['storage_cost']})
        cost_df = pd.DataFrame(all_costs)
        fig2 = px.bar(cost_df, x='Supplier', y=['Material','Transport','Storage'],
                      barmode='stack', title='Total Cost Breakdown per Supplier')
        st.plotly_chart(fig2, use_container_width=True)

    with tab5:
        st.subheader("💡 Why Is This Supplier Recommended?")
        col1, col2, col3 = st.columns(3)
        with col1: sel_mat = st.selectbox("Material", materials['material_name'].tolist(), key='rec_mat')
        with col2: qty5 = st.slider("Quantity Needed", 100, 2000, 500, key='rec_qty')
        with col3:
            sel_m5 = st.selectbox("Month", list(range(1,13)), key='rec_month',
                                   format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun',
                                                          'Jul','Aug','Sep','Oct','Nov','Dec'][m-1])
        mat_id = materials[materials['material_name']==sel_mat]['material_id'].values[0]
        with st.spinner("Ranking suppliers..."):
            rec_df = explain_recommendation(mat_id, qty5, sel_m5)
        best = rec_df.iloc[0]
        st.success(f"✅ **Best: {best['supplier_name']}** — Score: {best['final_score']:.1f}/100 | "
                   f"Delay Risk: {best['delay_probability']}% | Cost: ₹{best['total_cost']:,}")
        fig = px.bar(rec_df, x='supplier_name', y='final_score',
                     color='final_score', color_continuous_scale='Greens',
                     title='Supplier Recommendation Score (Higher = Better)')
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(rec_df, x='supplier_name', y=['cost_score','delay_score','h_score'],
                      barmode='stack', title='Score Component Breakdown',
                      color_discrete_map={'cost_score':'#636EFA','delay_score':'#00CC96','h_score':'#FFA15A'})
        st.plotly_chart(fig2, use_container_width=True)
        for _, row in rec_df.iterrows():
            can = "✅ Can fulfil" if row['can_fulfill'] else "❌ Insufficient stock"
            if row['final_score'] == rec_df['final_score'].max():
                st.success(f"🥇 **{row['supplier_name']}** — Score: {row['final_score']:.1f} | {can} | {row['reason']}")
            elif row['delay_probability'] > 60:
                st.error(f"🔴 **{row['supplier_name']}** — Score: {row['final_score']:.1f} | Delay: {row['delay_probability']}% | {can}")
            else:
                st.info(f"🔵 **{row['supplier_name']}** — Score: {row['final_score']:.1f} | Delay: {row['delay_probability']}% | {can}")

# ════════════════════════════════════════════════════════
# PAGE 8 — DATABASE & SCHEMA
# ════════════════════════════════════════════════════════
elif page == "🗄️ Database & Schema":
    st.title("🗄️ Database & ER Schema")
    st.markdown("---")

    tab_erd, tab_sql, tab_explore = st.tabs([
        "📐 ER Diagram",
        "🧱 SQL Schema",
        "🔎 Live DB Explorer"
    ])

    with tab_erd:
        st.subheader("📐 Entity Relationship Diagram")
        st.markdown("""
        The procurement database follows a **normalised relational schema** with 6 tables
        and 3 pre-built views. All foreign key relationships are enforced via SQLite constraints.
        """)
        if os.path.exists('erd_schema.png'):
            st.image('erd_schema.png', caption='ERD — Procurement Reliability Database',
                     use_container_width=True)
        else:
            st.warning("Run `python generate_erd.py` to generate the ER diagram.")

        st.markdown("""
        **Relationships:**
        - `orders.supplier_id` → `suppliers.supplier_id` (Many-to-One)
        - `orders.material_id` → `materials.material_id` (Many-to-One)
        - `inventory.material_id` → `materials.material_id` (One-to-One)
        - `supplier_stats.supplier_id` → `suppliers.supplier_id` (One-to-One)
        - `forecast.material_id` → `materials.material_id` (Many-to-One)
        """)

    with tab_sql:
        st.subheader("🧱 Table Definitions & Views")

        ddl_snippets = {
            "orders (fact table)": """CREATE TABLE orders (
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
    delayed             INTEGER,   -- 0/1 flag
    partial_delivery    INTEGER,   -- 0/1 flag
    quality_issue       INTEGER,   -- 0/1 flag
    backup_used         INTEGER,   -- 0/1 flag
    total_cost          REAL,
    weather_risk        REAL,
    season              TEXT       -- 'Peak' / 'Normal'
);""",
            "suppliers (dimension)": """CREATE TABLE suppliers (
    supplier_id             TEXT PRIMARY KEY,
    supplier_name           TEXT NOT NULL,
    location                TEXT,
    distance_km             REAL,
    price_per_unit          REAL,
    reliability_score       REAL,  -- 0-1
    geopolitical_risk       REAL,  -- 0-1
    strike_risk             REAL,  -- 0-1
    quality_score           REAL,  -- 0-1
    available_stock         INTEGER,
    max_capacity            INTEGER
);""",
            "v_inventory_alert (view)": """CREATE VIEW v_inventory_alert AS
    SELECT
        material_name,
        current_stock,
        reorder_level,
        daily_consumption,
        CAST(current_stock AS REAL) / daily_consumption AS days_remaining,
        CASE
            WHEN current_stock < reorder_level         THEN 'CRITICAL'
            WHEN current_stock < reorder_level * 1.2   THEN 'WARNING'
            ELSE 'OK'
        END AS alert_status
    FROM inventory;"""
        }

        for title, ddl in ddl_snippets.items():
            with st.expander(f"📋 {title}"):
                st.code(ddl, language='sql')

    with tab_explore:
        st.subheader("🔎 Live Database Explorer")
        st.markdown("Query the live SQLite database directly from the dashboard.")

        if os.path.exists('procurement.db'):
            # Table picker
            table_choice = st.selectbox("Select table / view", [
                "suppliers", "materials", "inventory", "orders",
                "supplier_stats", "forecast",
                "v_order_details", "v_supplier_summary", "v_inventory_alert"
            ])
            limit = st.slider("Rows to display", 5, 100, 20)

            try:
                preview_df = load_db_query(f"SELECT * FROM {table_choice} LIMIT {limit}")
                st.dataframe(preview_df, use_container_width=True)
                st.caption(f"Showing {len(preview_df)} rows from `{table_choice}`")
            except Exception as e:
                st.error(f"Query error: {e}")

            st.markdown("**✍️ Custom SQL Query**")
            custom_sql = st.text_area("Enter SQL", value="SELECT supplier_name, health_score, grade FROM v_supplier_summary ORDER BY health_score DESC")
            if st.button("▶️ Run Query"):
                try:
                    result = load_db_query(custom_sql)
                    st.dataframe(result, use_container_width=True)
                    st.success(f"✅ Returned {len(result)} rows")
                except Exception as e:
                    st.error(f"SQL Error: {e}")
        else:
            st.warning("Database not found. Run `python create_db.py` first.")