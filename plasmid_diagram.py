"""
plasmid_diagram.py
==================
Draws the engineered Bifidobacterium plasmid circuit and saves it as
'plasmid_diagram.png'.  This image is displayed in the simulation dashboard
as a side-panel inset.

Run this SECOND (after gut_background.py, before gut_simulation.py).

The plasmid contains:
  • Origin of Replication (pBBR1 ori)
  • Antibiotic resistance marker (AmpR)
  • PmarR promoter  →  phlF gene  (SENSING / NOT-gate stage 1)
  • PphlF promoter  →  queD gene  (INVERSION / NOT-gate stage 2)
  • queD = Quercetin Dioxygenase (codon-optimised therapeutic payload)

Dependencies: matplotlib, numpy
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────────────────
# Canvas
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0f0f1e')

# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw a thick arc segment on the plasmid ring
# ─────────────────────────────────────────────────────────────────────────────
RING_R   = 0.85      # radius of plasmid backbone
RING_W   = 0.12      # visual width

def arc_patch(ax, theta1, theta2, color, label=None, label_r=1.04, fontsize=8):
    """Draw a coloured arc sector on the plasmid ring."""
    theta = np.linspace(np.radians(theta1), np.radians(theta2), 300)
    # Outer edge
    x_out = (RING_R + RING_W/2) * np.cos(theta)
    y_out = (RING_R + RING_W/2) * np.sin(theta)
    # Inner edge (reversed)
    x_in  = (RING_R - RING_W/2) * np.cos(theta[::-1])
    y_in  = (RING_R - RING_W/2) * np.sin(theta[::-1])
    xs = np.concatenate([x_out, x_in, [x_out[0]]])
    ys = np.concatenate([y_out, y_in, [y_out[0]]])
    ax.fill(xs, ys, color=color, alpha=0.92, zorder=3)
    ax.plot(xs, ys, color='#ffffff', lw=0.4, alpha=0.5, zorder=4)

    if label:
        mid = np.radians((theta1 + theta2) / 2)
        lx  = label_r * np.cos(mid)
        ly  = label_r * np.sin(mid)
        ha  = 'left' if lx > 0.05 else ('right' if lx < -0.05 else 'center')
        ax.text(lx, ly, label, ha=ha, va='center', fontsize=fontsize,
                color='#e0e0e0', fontweight='bold', fontfamily='monospace',
                zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='#0f0f1e')])

# ─────────────────────────────────────────────────────────────────────────────
# Draw the plasmid backbone (grey ring)
# ─────────────────────────────────────────────────────────────────────────────
backbone = plt.Circle((0, 0), RING_R, fill=False,
                       edgecolor='#445566', linewidth=5, zorder=2)
ax.add_patch(backbone)

# ─────────────────────────────────────────────────────────────────────────────
# Plasmid elements: (start_deg, end_deg, colour, label)
# Convention: 0° = right, counter-clockwise positive
# ─────────────────────────────────────────────────────────────────────────────
elements = [
    # (theta1, theta2, colour, label, label_r)
    (20,   95,  '#e74c3c', 'PmarR\nPromoter',  1.18),   # Sensing promoter
    (100,  175, '#e67e22', 'phlF\n(Repressor)', 1.18),  # NOT-gate gene 1
    (185,  260, '#2ecc71', 'PphlF NOT-Gate\nPromoter', 1.22),  # Inverted promoter
    (265,  340, '#3498db', 'queD\n(Quercetin\nDioxygenase)', 1.20),  # Therapeutic enzyme
    (345,  380, '#9b59b6', 'AmpR',              1.12),  # Resistance marker (wraps 345→20)
    (   0, 20,  '#9b59b6', '',                  1.12),  # AmpR continuation
    # pBBR1 ori sits between 190–220 (drawn as a special marker below)
]

for (t1, t2, col, lbl, lr) in elements:
    arc_patch(ax, t1, t2, col, label=lbl, label_r=lr, fontsize=7.5)

# ─────────────────────────────────────────────────────────────────────────────
# Origin of Replication marker (pBBR1 ori)
# ─────────────────────────────────────────────────────────────────────────────
ori_angle = np.radians(175 + (185-175)/2)   # between phlF and PphlF
ori_x = RING_R * np.cos(ori_angle)
ori_y = RING_R * np.sin(ori_angle)
ax.plot(ori_x, ori_y, 'o', ms=9, color='#f1c40f', zorder=6,
        path_effects=[pe.withStroke(linewidth=2, foreground='black')])
ax.text(ori_x * 1.28, ori_y * 1.18, 'pBBR1\nori', ha='center', va='center',
        fontsize=7, color='#f1c40f', fontweight='bold', fontfamily='monospace',
        zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# Transcription direction arrows on the ring
# ─────────────────────────────────────────────────────────────────────────────
def ring_arrow(ax, theta_deg, color='white'):
    """Small arrow tangent to the ring showing transcription direction."""
    t = np.radians(theta_deg)
    x0 = RING_R * np.cos(t - 0.06)
    y0 = RING_R * np.sin(t - 0.06)
    x1 = RING_R * np.cos(t + 0.06)
    y1 = RING_R * np.sin(t + 0.06)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                zorder=7)

for deg in [55, 135, 220, 300]:
    ring_arrow(ax, deg)

# ─────────────────────────────────────────────────────────────────────────────
# Central plasmid label
# ─────────────────────────────────────────────────────────────────────────────
ax.text(0, 0.10, 'pBBR1-IDP', ha='center', va='center',
        fontsize=11, color='#ffffff', fontweight='bold',
        fontfamily='monospace', zorder=10)
ax.text(0, -0.10, '~4.2 kbp', ha='center', va='center',
        fontsize=8.5, color='#aaaaaa', fontfamily='monospace', zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# Logic flow arrows inside the circle (circuit topology)
# ─────────────────────────────────────────────────────────────────────────────
# Arrow 1: PmarR → phlF (upper arc, sensing activation)
arrow_style_act = dict(arrowstyle='->', color='#e74c3c', lw=1.5,
                        connectionstyle='arc3,rad=0.35')
ax.annotate('', xy=(-0.25, 0.30), xytext=(0.25, 0.30),
            arrowprops=arrow_style_act, zorder=8)
ax.text(0, 0.50, 'Quercetin\nSensing', ha='center', fontsize=6.5,
        color='#e74c3c', fontfamily='monospace')

# Arrow 2: phlF ⊣ PphlF (repression, lower arc with flat head)
arrow_style_rep = dict(arrowstyle='-[', color='#e67e22', lw=1.5,
                        connectionstyle='arc3,rad=0.35')
ax.annotate('', xy=(-0.25, -0.28), xytext=(0.25, -0.28),
            arrowprops=arrow_style_rep, zorder=8)
ax.text(0, -0.48, 'NOT-gate\nInversion', ha='center', fontsize=6.5,
        color='#e67e22', fontfamily='monospace')

# Arrow 3: queD → DOPAC payload (rightward, positive output)
ax.annotate('', xy=(0.48, 0), xytext=(0.22, 0),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=2.0),
            zorder=8)
ax.text(0.62, 0, 'DOPAC\nOutput', ha='left', va='center', fontsize=6.5,
        color='#3498db', fontfamily='monospace')

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color='#e74c3c', label='PmarR Promoter  (Quercetin-sensing)'),
    mpatches.Patch(color='#e67e22', label='phlF  (Repressor / NOT-gate input)'),
    mpatches.Patch(color='#2ecc71', label='PphlF Promoter  (Inverted NOT-gate)'),
    mpatches.Patch(color='#3498db', label='queD  (Quercetin Dioxygenase payload)'),
    mpatches.Patch(color='#9b59b6', label='AmpR  (Selection marker)'),
    mpatches.Patch(color='#f1c40f', label='pBBR1 ori  (Broad-host replication)'),
]
leg = ax.legend(handles=legend_items, loc='lower center',
                bbox_to_anchor=(0.5, -0.22), ncol=2,
                fontsize=6.5, framealpha=0.15, labelcolor='white',
                facecolor='#0f0f1e', edgecolor='#333355',
                handlelength=1.2, handletextpad=0.5, columnspacing=0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
ax.set_title('Engineered Plasmid: pBBR1-IDP\nMarR–PhlF Double-Inverter Circuit',
             fontsize=9.5, color='white', fontweight='bold',
             fontfamily='monospace', pad=8)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig('plasmid_diagram.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print("✅  plasmid_diagram.png saved.")
print("    Run gut_simulation.py next.")
