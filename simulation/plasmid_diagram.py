"""
plasmid_diagram.py
==================
Draws the engineered Bifidobacterium plasmid pDOJH10S (4572 bp) and saves it
as 'plasmid_diagram.png'.  Displayed in the simulation dashboard as a side-panel.

Run this SECOND (after gut_background.py, before gut_simulation.py).

pDOJH10S contains:
  • Native backbone  (broad-host-range scaffold)
  • RBS 1 & RBS 2   (ribosome binding sites)
  • PhlF R          (PhlF repressor / NOT-gate element)
  • queD            (Quercetin Dioxygenase – therapeutic payload)
  • Terminator

Dependencies: matplotlib, numpy
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
ax.set_xlim(-1.65, 1.65)
ax.set_ylim(-1.65, 1.65)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0f0f1e')

RING_R    = 0.85
RING_W    = 0.13
TOTAL_BP  = 4572

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
#   bp 0 = top (12 o'clock), positions increase clockwise
#   → standard-math angle (CCW from right) = 90 – (bp/TOTAL_BP)*360
# ─────────────────────────────────────────────────────────────────────────────
def bp_to_deg(bp):
    return 90.0 - (bp / TOTAL_BP * 360.0)

def bp_to_rad(bp):
    return np.radians(bp_to_deg(bp))

def arc_patch(ax, bp_start, bp_end, color, label=None, label_r=1.16, fontsize=7.5):
    """Draw a filled arc from bp_start to bp_end (clockwise)."""
    t1 = bp_to_deg(bp_start)
    t2 = bp_to_deg(bp_end)
    # CW on plasmid = decreasing standard-math angle
    theta = np.linspace(np.radians(t1), np.radians(t2), 300)
    x_out = (RING_R + RING_W / 2) * np.cos(theta)
    y_out = (RING_R + RING_W / 2) * np.sin(theta)
    x_in  = (RING_R - RING_W / 2) * np.cos(theta[::-1])
    y_in  = (RING_R - RING_W / 2) * np.sin(theta[::-1])
    xs = np.concatenate([x_out, x_in, [x_out[0]]])
    ys = np.concatenate([y_out, y_in, [y_out[0]]])
    ax.fill(xs, ys, color=color, alpha=0.93, zorder=3)
    ax.plot(xs, ys, color='#ffffff', lw=0.3, alpha=0.35, zorder=4)

    if label:
        mid_bp  = (bp_start + bp_end) / 2
        mid_ang = bp_to_rad(mid_bp)
        lx = label_r * np.cos(mid_ang)
        ly = label_r * np.sin(mid_ang)
        ha = 'left' if lx > 0.08 else ('right' if lx < -0.08 else 'center')
        ax.text(lx, ly, label, ha=ha, va='center', fontsize=fontsize,
                color='#e0e0e0', fontweight='bold', fontfamily='monospace',
                zorder=10,
                path_effects=[pe.withStroke(linewidth=1.8, foreground='#0f0f1e')])

# ─────────────────────────────────────────────────────────────────────────────
# Backbone ring (grey outline)
# ─────────────────────────────────────────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), RING_R, fill=False,
                         edgecolor='#334455', linewidth=4, zorder=2))

# ─────────────────────────────────────────────────────────────────────────────
# Plasmid elements  (bp ranges approximate, matching pDOJH10S map)
# ─────────────────────────────────────────────────────────────────────────────

# 1. Native backbone — large teal arc  (550 → 3580 bp)
arc_patch(ax, 550, 3580, '#4db88a',
          label='native backbone', label_r=1.22, fontsize=7.5)

# 2. RBS 1  (3580 → 3640 bp)
arc_patch(ax, 3580, 3640, '#3377bb',
          label='RBS 1', label_r=1.14, fontsize=6.5)

# 3. RBS 2  (3640 → 3700 bp)
arc_patch(ax, 3640, 3700, '#5599dd',
          label='RBS 2', label_r=1.12, fontsize=6.5)

# 4. PhlF R — repressor / NOT-gate  (3700 → 3920 bp)
arc_patch(ax, 3700, 3920, '#b07848',
          label='PhlF R\n(Repressor)', label_r=1.16, fontsize=7)

# 5. queD — Quercetin Dioxygenase  (3920 → 4430 bp)
arc_patch(ax, 3920, 4430, '#d4b84a',
          label='Quercetin\nDioxygenase', label_r=1.22, fontsize=7.5)

# 6. Terminator — wraps over 0 bp  (4430 → 550 bp, two segments)
arc_patch(ax, 4430, 4572, '#2ecc71')
arc_patch(ax, 0,    550,  '#2ecc71',
          label='Terminator', label_r=1.20, fontsize=7)

# Terminator triangle marker at ~490 bp
t_ang = bp_to_rad(490)
tx, ty = RING_R * np.cos(t_ang), RING_R * np.sin(t_ang)
nx, ny = -np.sin(t_ang), np.cos(t_ang)   # outward normal (CW tangent rotated 90°)
tri_pts = [
    (tx - 0.04 * ny, ty + 0.04 * nx),
    (tx + 0.04 * ny, ty - 0.04 * nx),
    (tx + 0.07 * np.cos(t_ang), ty + 0.07 * np.sin(t_ang)),
]
ax.add_patch(plt.Polygon(tri_pts, closed=True, color='#2ecc71', zorder=6))

# ─────────────────────────────────────────────────────────────────────────────
# Directional arrows (show clockwise transcription direction)
# ─────────────────────────────────────────────────────────────────────────────
def ring_arrow(ax, bp, color='#888888'):
    t   = bp_to_rad(bp)
    dt  = 0.065
    x0  = RING_R * np.cos(t + dt)
    y0  = RING_R * np.sin(t + dt)
    x1  = RING_R * np.cos(t - dt)
    y1  = RING_R * np.sin(t - dt)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2), zorder=7)

for bp in [1200, 2500, 4150]:
    ring_arrow(ax, bp)
ring_arrow(ax, 3610, color='#5599dd')   # RBS direction
ring_arrow(ax, 3670, color='#5599dd')

# ─────────────────────────────────────────────────────────────────────────────
# BP position tick labels
# ─────────────────────────────────────────────────────────────────────────────
for bp_pos in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]:
    ang   = bp_to_rad(bp_pos)
    inner = RING_R - RING_W / 2 - 0.06
    tick0 = RING_R - RING_W / 2 - 0.01
    ax.plot([tick0 * np.cos(ang), inner * np.cos(ang)],
            [tick0 * np.sin(ang), inner * np.sin(ang)],
            color='#445566', lw=0.8, zorder=5)
    ax.text(inner * np.cos(ang), inner * np.sin(ang),
            str(bp_pos), ha='center', va='center',
            fontsize=5.5, color='#778899', fontfamily='monospace', zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# Internal circuit flow arrows
# ─────────────────────────────────────────────────────────────────────────────
# Quercetin input → PhlF R
ax.annotate('', xy=(-0.15, 0.42), xytext=(-0.15, 0.70),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.6), zorder=8)
ax.text(-0.15, 0.80, 'Quercetin\n(Input signal)', ha='center', fontsize=6.5,
        color='#e74c3c', fontfamily='monospace')

# PhlF R ⊣ queD  (repression bar)
ax.annotate('', xy=(0.05, 0.20), xytext=(-0.20, 0.38),
            arrowprops=dict(arrowstyle='-[', color='#b07848', lw=1.5,
                            connectionstyle='arc3,rad=0.25'), zorder=8)
ax.text(-0.10, 0.05, 'NOT-gate\nInversion', ha='center', fontsize=6.5,
        color='#b07848', fontfamily='monospace')

# queD → DOPAC output
ax.annotate('', xy=(-0.55, 0.30), xytext=(-0.25, 0.30),
            arrowprops=dict(arrowstyle='->', color='#d4b84a', lw=2.0), zorder=8)
ax.text(-0.70, 0.30, 'DOPAC\nOutput', ha='center', va='center', fontsize=6.5,
        color='#d4b84a', fontfamily='monospace')

# ─────────────────────────────────────────────────────────────────────────────
# Central label
# ─────────────────────────────────────────────────────────────────────────────
ax.text(0,  0.12, 'pDOJH10S', ha='center', va='center',
        fontsize=12, color='#ffffff', fontweight='bold',
        fontfamily='monospace', zorder=10)
ax.text(0, -0.10, '4572 bp', ha='center', va='center',
        fontsize=9, color='#aaaaaa', fontfamily='monospace', zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color='#4db88a', label='Native backbone  (broad-host range)'),
    mpatches.Patch(color='#d4b84a', label='queD  (Quercetin Dioxygenase)'),
    mpatches.Patch(color='#b07848', label='PhlF R  (NOT-gate repressor)'),
    mpatches.Patch(color='#5599dd', label='RBS 1 & RBS 2'),
    mpatches.Patch(color='#2ecc71', label='Terminator'),
]
ax.legend(handles=legend_items, loc='lower center',
          bbox_to_anchor=(0.5, -0.20), ncol=2,
          fontsize=6.5, framealpha=0.15, labelcolor='white',
          facecolor='#0f0f1e', edgecolor='#333355',
          handlelength=1.2, handletextpad=0.5, columnspacing=0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
ax.set_title('Engineered Plasmid: pDOJH10S\nPhlF NOT-Gate Circuit  ·  Quercetin → DOPAC',
             fontsize=9.5, color='white', fontweight='bold',
             fontfamily='monospace', pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig('plasmid_diagram.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print("✅  plasmid_diagram.png saved  (pDOJH10S · 4572 bp)")
print("    Run gut_simulation.py next.")
