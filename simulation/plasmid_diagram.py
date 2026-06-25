"""
plasmid_diagram.py
==================
Draws the engineered Bifidobacterium plasmid pDOJH10S (4572 bp) and saves it
as 'plasmid_diagram.png'.  Displayed in the simulation dashboard as a side-panel.

pDOJH10S contains:
  • Native backbone  (broad-host-range scaffold)
  • RBS 1 & RBS 2   (ribosome binding sites)
  • PhlF R          (PhlF repressor / NOT-gate element)
  • queD            (Quercetin Dioxygenase – therapeutic payload)
  • Terminator

Run this SECOND (after gut_background.py, before gut_simulation.py).
Dependencies: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
ax.set_xlim(-1.70, 1.70)
ax.set_ylim(-1.18, 1.82)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0f0f1e')
ax.set_facecolor('#0f0f1e')

RING_R   = 1.0
RING_W   = 0.14
TOTAL_BP = 4572

# ── Coordinate helpers ────────────────────────────────────────────────────────
# bp 0 = top (12 o'clock), increases clockwise
# standard-math angle (CCW from 3 o'clock) = 90 – (bp/TOTAL_BP)*360
def bp_to_deg(bp):
    return 90.0 - (bp / TOTAL_BP * 360.0)

def bp_to_rad(bp):
    return np.radians(bp_to_deg(bp))

# ── Arc drawing ───────────────────────────────────────────────────────────────
def arc_patch(ax, bp_start, bp_end, color, label=None, label_r=1.22, fontsize=8):
    """Filled arc from bp_start to bp_end (clockwise). Label placed at midpoint."""
    t1 = np.radians(bp_to_deg(bp_start))
    t2 = np.radians(bp_to_deg(bp_end))
    theta = np.linspace(t1, t2, 400)           # t2 < t1  →  clockwise
    xo = (RING_R + RING_W / 2) * np.cos(theta)
    yo = (RING_R + RING_W / 2) * np.sin(theta)
    xi = (RING_R - RING_W / 2) * np.cos(theta[::-1])
    yi = (RING_R - RING_W / 2) * np.sin(theta[::-1])
    xs = np.concatenate([xo, xi, [xo[0]]])
    ys = np.concatenate([yo, yi, [yo[0]]])
    ax.fill(xs, ys, color=color, alpha=0.93, zorder=3)
    ax.plot(xs, ys, color='#ffffff', lw=0.25, alpha=0.3, zorder=4)

    if label:
        mid_ang = bp_to_rad((bp_start + bp_end) / 2)
        lx = label_r * np.cos(mid_ang)
        ly = label_r * np.sin(mid_ang)
        ha = 'left' if lx > 0.12 else ('right' if lx < -0.12 else 'center')
        ax.text(lx, ly, label,
                ha=ha, va='center', fontsize=fontsize,
                color='#e8e8e8', fontweight='bold', fontfamily='monospace',
                zorder=10,
                path_effects=[pe.withStroke(linewidth=2.0, foreground='#0f0f1e')])

# ── Backbone grey ring outline ────────────────────────────────────────────────
ax.add_patch(plt.Circle((0, 0), RING_R, fill=False,
                         edgecolor='#334455', linewidth=3.5, zorder=2))

# ── Plasmid elements ──────────────────────────────────────────────────────────
#  native backbone  550 → 3580 bp  (large teal arc)
arc_patch(ax, 550,  3580, '#3aaa78',
          label='native backbone', label_r=1.30, fontsize=8)

#  RBS 1  3580 → 3640 bp
arc_patch(ax, 3580, 3640, '#2266bb',
          label='RBS 1', label_r=1.40, fontsize=7.5)

#  RBS 2  3640 → 3700 bp
arc_patch(ax, 3640, 3700, '#4488dd',
          label='RBS 2', label_r=1.55, fontsize=7.5)

#  PhlF R (repressor)  3700 → 3930 bp
arc_patch(ax, 3700, 3930, '#a06030',
          label='PhlF R\n(Repressor)', label_r=1.42, fontsize=7.5)

#  queD (Quercetin Dioxygenase)  3930 → 4450 bp
arc_patch(ax, 3930, 4450, '#c8a830',
          label='Quercetin\nDioxygenase', label_r=1.30, fontsize=8)

#  Terminator  4450 → 550 bp  (wraps over top — two segments)
arc_patch(ax, 4450, 4572, '#27ae60')                                   # tail segment
arc_patch(ax, 0,    550,  '#27ae60',
          label='Terminator', label_r=1.30, fontsize=7.5)              # labelled segment

# Terminator arrowhead triangle at ~480 bp
t_ang = bp_to_rad(480)
# outward unit vector
nx_out, ny_out = np.cos(t_ang), np.sin(t_ang)
# CW-tangent unit vector
nx_tan, ny_tan = np.sin(t_ang), -np.cos(t_ang)
tri = [
    (RING_R * nx_out - 0.045 * nx_tan, RING_R * ny_out - 0.045 * ny_tan),
    (RING_R * nx_out + 0.045 * nx_tan, RING_R * ny_out + 0.045 * ny_tan),
    ((RING_R + 0.09) * nx_out,          (RING_R + 0.09) * ny_out),
]
ax.add_patch(plt.Polygon(tri, closed=True, color='#27ae60', zorder=6))

# ── Directional arrows on ring (CW transcription) ────────────────────────────
def ring_arrow(ax, bp, color='#667788'):
    t   = bp_to_rad(bp)
    dt  = 0.07   # half-span in radians for arrow shaft
    x0  = RING_R * np.cos(t + dt)
    y0  = RING_R * np.sin(t + dt)
    x1  = RING_R * np.cos(t - dt)
    y1  = RING_R * np.sin(t - dt)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.1, mutation_scale=10),
                zorder=7)

ring_arrow(ax, 1500)             # backbone mid-point
ring_arrow(ax, 2800)             # backbone lower
ring_arrow(ax, 4180, '#c8a830') # queD direction

# ── Tick marks + bp labels (outside the ring) ────────────────────────────────
for bp_pos in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]:
    ang   = bp_to_rad(bp_pos)
    r_out = RING_R + RING_W / 2 + 0.03   # just outside arc
    r_lbl = RING_R + RING_W / 2 + 0.14   # label further out
    r_in  = RING_R + RING_W / 2 + 0.00
    ax.plot([r_in * np.cos(ang), r_out * np.cos(ang)],
            [r_in * np.sin(ang), r_out * np.sin(ang)],
            color='#445566', lw=0.8, zorder=5)
    ax.text(r_lbl * np.cos(ang), r_lbl * np.sin(ang),
            str(bp_pos),
            ha='center', va='center', fontsize=6,
            color='#778899', fontfamily='monospace', zorder=10)

# ── Central label (well clear of all arcs) ───────────────────────────────────
ax.text(0,  0.18, 'pDOJH10S',
        ha='center', va='center', fontsize=14,
        color='#ffffff', fontweight='bold', fontfamily='monospace', zorder=10)
ax.text(0, -0.12, '4572 bp',
        ha='center', va='center', fontsize=10,
        color='#aaaaaa', fontfamily='monospace', zorder=10)

# ── Circuit logic — 3 lines inside ring, safely above inner edge (y > -0.78) ────
circuit_lines = [
    ('Quercetin rises', '#e74c3c'),
    ('-> PhlF R off  -> queD on', '#c8a830'),
    ('-> DOPAC produced', '#3aaa78'),
]
for i, (txt, col) in enumerate(circuit_lines):
    ax.text(0, -0.40 - i * 0.18, txt,
            ha='center', va='center', fontsize=7,
            color=col, fontfamily='monospace',
            fontweight='bold', zorder=10,
            path_effects=[pe.withStroke(linewidth=1.8, foreground='#0f0f1e')])

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color='#3aaa78', label='Native backbone'),
    mpatches.Patch(color='#c8a830', label='queD  (Quercetin Dioxygenase)'),
    mpatches.Patch(color='#a06030', label='PhlF R  (NOT-gate repressor)'),
    mpatches.Patch(color='#4488dd', label='RBS 1 & RBS 2'),
    mpatches.Patch(color='#27ae60', label='Terminator'),
]
fig.legend(handles=legend_items, loc='lower center',
           bbox_to_anchor=(0.5, 0.03),
           ncol=2, fontsize=7.5, framealpha=0.18,
           labelcolor='white', facecolor='#0f0f1e',
           edgecolor='#333355', handlelength=1.2,
           handletextpad=0.5, columnspacing=0.8)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title('Engineered Plasmid: pDOJH10S  ·  4572 bp\n'
             'PhlF NOT-Gate Circuit  |  Quercetin → DOPAC',
             fontsize=10, color='#d0d8ff', fontweight='bold',
             fontfamily='monospace', pad=10)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.14)
plt.savefig('plasmid_diagram.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print("plasmid_diagram.png saved  (pDOJH10S - 4572 bp)")
print("    Run gut_simulation.py next.")

