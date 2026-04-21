"""
generate_erd.py  —  Generates the ER diagram as a PNG (erd_schema.png)
Uses matplotlib only — no extra dependencies.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Entity definitions: (table_name, [(col, type, pk/fk/none)]) ───────────────
ENTITIES = {
    'suppliers': [
        ('supplier_id',           'TEXT',  'PK'),
        ('supplier_name',         'TEXT',  ''),
        ('location',              'TEXT',  ''),
        ('distance_km',           'REAL',  ''),
        ('price_per_unit',        'REAL',  ''),
        ('transport_cost_per_km', 'REAL',  ''),
        ('reliability_score',     'REAL',  ''),
        ('geopolitical_risk',     'REAL',  ''),
        ('strike_risk',           'REAL',  ''),
        ('quality_score',         'REAL',  ''),
        ('available_stock',       'INT',   ''),
        ('max_capacity',          'INT',   ''),
    ],
    'materials': [
        ('material_id',   'TEXT', 'PK'),
        ('material_name', 'TEXT', ''),
        ('category',      'TEXT', ''),
        ('unit',          'TEXT', ''),
        ('base_demand',   'INT',  ''),
    ],
    'inventory': [
        ('material_id',       'TEXT', 'PK/FK→materials'),
        ('current_stock',     'INT',  ''),
        ('reorder_level',     'INT',  ''),
        ('storage_capacity',  'INT',  ''),
        ('daily_consumption', 'INT',  ''),
    ],
    'orders': [
        ('order_id',          'TEXT', 'PK'),
        ('supplier_id',       'TEXT', 'FK→suppliers'),
        ('material_id',       'TEXT', 'FK→materials'),
        ('quantity',          'INT',  ''),
        ('delivered_qty',     'INT',  ''),
        ('accepted_qty',      'INT',  ''),
        ('rejected_qty',      'INT',  ''),
        ('order_date',        'TEXT', ''),
        ('expected_delivery', 'TEXT', ''),
        ('actual_delivery',   'TEXT', ''),
        ('delay_days',        'INT',  ''),
        ('delayed',           'INT',  ''),
        ('partial_delivery',  'INT',  ''),
        ('quality_issue',     'INT',  ''),
        ('backup_used',       'INT',  ''),
        ('total_cost',        'REAL', ''),
        ('weather_risk',      'REAL', ''),
        ('season',            'TEXT', ''),
    ],
    'supplier_stats': [
        ('supplier_id',        'TEXT', 'PK/FK→suppliers'),
        ('total_orders',       'INT',  ''),
        ('delayed_orders',     'INT',  ''),
        ('avg_delay_days',     'REAL', ''),
        ('avg_total_cost',     'REAL', ''),
        ('partial_deliveries', 'INT',  ''),
        ('quality_issues',     'INT',  ''),
        ('backup_activations', 'INT',  ''),
        ('delay_rate',         'REAL', ''),
        ('health_score',       'REAL', ''),
        ('grade',              'TEXT', ''),
    ],
    'forecast': [
        ('material_id',      'TEXT', 'PK/FK→materials'),
        ('next_month',       'INT',  'PK'),
        ('predicted_demand', 'INT',  ''),
    ],
}

# ── Layout positions (x, y) in data coords ────────────────────────────────────
POSITIONS = {
    'orders':         (5.5, 5.0),
    'suppliers':      (1.0, 7.5),
    'materials':      (10.0, 7.5),
    'inventory':      (10.0, 2.5),
    'supplier_stats': (1.0, 2.5),
    'forecast':       (10.0, 0.2),
}

BOX_W = 3.2
ROW_H = 0.32
HEADER_H = 0.45

COLORS = {
    'header_main': '#1e3a5f',
    'header_alt':  '#2d6a9f',
    'pk_bg':       '#fff3cd',
    'fk_bg':       '#d4edda',
    'row_even':    '#f8f9fa',
    'row_odd':     '#ffffff',
    'border':      '#adb5bd',
    'arrow':       '#555555',
    'text_header': 'white',
    'text_pk':     '#856404',
    'text_fk':     '#155724',
    'text_normal': '#343a40',
}

def draw_entity(ax, name, columns, pos):
    x, y = pos
    n = len(columns)
    total_h = HEADER_H + n * ROW_H

    # Shadow
    shadow = FancyBboxPatch((x+0.04, y - total_h - 0.04), BOX_W, total_h,
                             boxstyle='round,pad=0.02',
                             fc='#cccccc', ec='none', zorder=1, alpha=0.4)
    ax.add_patch(shadow)

    # Header
    header = FancyBboxPatch((x, y - HEADER_H), BOX_W, HEADER_H,
                             boxstyle='round,pad=0.02',
                             fc=COLORS['header_main'], ec=COLORS['border'],
                             linewidth=1.5, zorder=2)
    ax.add_patch(header)
    ax.text(x + BOX_W/2, y - HEADER_H/2, name.upper(),
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=COLORS['text_header'], zorder=3)

    # Rows
    for i, (col, dtype, key) in enumerate(columns):
        ry = y - HEADER_H - (i+1)*ROW_H
        if key.startswith('PK'):
            bg = COLORS['pk_bg']
            tc = COLORS['text_pk']
            fw = 'bold'
            prefix = "[PK] "
        elif key.startswith('FK'):
            bg = COLORS['fk_bg']
            tc = COLORS['text_fk']
            fw = 'normal'
            prefix = "[FK] "
        else:
            bg = COLORS['row_even'] if i % 2 == 0 else COLORS['row_odd']
            tc = COLORS['text_normal']
            fw = 'normal'
            prefix = "     "

        row_rect = FancyBboxPatch((x, ry), BOX_W, ROW_H,
                                   boxstyle='square,pad=0',
                                   fc=bg, ec=COLORS['border'],
                                   linewidth=0.5, zorder=2)
        ax.add_patch(row_rect)
        ax.text(x + 0.1, ry + ROW_H/2, f"{prefix}{col}",
                ha='left', va='center', fontsize=6.5, fontweight=fw,
                color=tc, zorder=3)
        ax.text(x + BOX_W - 0.1, ry + ROW_H/2, dtype,
                ha='right', va='center', fontsize=6.0,
                color='#6c757d', zorder=3)

    return (x, y - HEADER_H - n*ROW_H, BOX_W, HEADER_H + n*ROW_H)  # bbox


def center(pos, n_cols):
    x, y = pos
    total_h = HEADER_H + n_cols * ROW_H
    cx = x + BOX_W / 2
    cy = y - total_h / 2
    return cx, cy


def box_edge(pos, n_cols, side='right'):
    x, y = pos
    total_h = HEADER_H + n_cols * ROW_H
    cy = y - total_h / 2
    if side == 'right':  return (x + BOX_W, cy)
    if side == 'left':   return (x,          cy)
    if side == 'top':    return (x + BOX_W/2, y)
    if side == 'bottom': return (x + BOX_W/2, y - total_h)


def draw_arrow(ax, start, end, label=''):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                                lw=1.4, connectionstyle='arc3,rad=0.08'),
                zorder=1)
    if label:
        mx, my = (start[0]+end[0])/2, (start[1]+end[1])/2
        ax.text(mx, my, label, fontsize=6, color='#555', ha='center',
                bbox=dict(fc='white', ec='none', alpha=0.8))


def main():
    fig, ax = plt.subplots(figsize=(17, 12))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_facecolor('#f0f4fa')
    fig.patch.set_facecolor('#f0f4fa')

    ax.text(7, 10.6, 'Procurement Reliability Dashboard — ER Schema',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=COLORS['header_main'])
    ax.text(7, 10.25, 'UE23CS342BA1 | Problem Statement P2 | PES University',
            ha='center', va='center', fontsize=9, color='#555')

    # Draw all entities
    for name, cols in ENTITIES.items():
        draw_entity(ax, name, cols, POSITIONS[name])

    # Relationships (arrows)
    # orders → suppliers
    draw_arrow(ax,
               box_edge(POSITIONS['orders'],        len(ENTITIES['orders']),        'left'),
               box_edge(POSITIONS['suppliers'],     len(ENTITIES['suppliers']),     'right'),
               'supplier_id')

    # orders → materials
    draw_arrow(ax,
               box_edge(POSITIONS['orders'],        len(ENTITIES['orders']),        'right'),
               box_edge(POSITIONS['materials'],     len(ENTITIES['materials']),     'left'),
               'material_id')

    # inventory → materials
    draw_arrow(ax,
               box_edge(POSITIONS['inventory'],     len(ENTITIES['inventory']),     'left'),
               box_edge(POSITIONS['materials'],     len(ENTITIES['materials']),     'right'),
               'material_id')

    # supplier_stats → suppliers
    draw_arrow(ax,
               box_edge(POSITIONS['supplier_stats'],len(ENTITIES['supplier_stats']),'right'),
               box_edge(POSITIONS['suppliers'],     len(ENTITIES['suppliers']),     'left'),
               'supplier_id')

    # forecast → materials
    draw_arrow(ax,
               box_edge(POSITIONS['forecast'],      len(ENTITIES['forecast']),      'left'),
               box_edge(POSITIONS['materials'],     len(ENTITIES['materials']),     'right'),
               'material_id')

    # Legend
    legend_items = [
        mpatches.Patch(fc=COLORS['pk_bg'],  ec='#aaa', label='Primary Key (PK)'),
        mpatches.Patch(fc=COLORS['fk_bg'],  ec='#aaa', label='Foreign Key (FK)'),
        mpatches.Patch(fc=COLORS['row_even'],ec='#aaa', label='Regular Column'),
    ]
    ax.legend(handles=legend_items, loc='lower left', fontsize=8,
              framealpha=0.9, fancybox=True)

    plt.tight_layout()
    plt.savefig('erd_schema.png', dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("✅ ER diagram saved → erd_schema.png")


if __name__ == '__main__':
    main()
