"""
gut_background.py
=================
Generates a 2D cross-sectional gut background image saved as 'gut_bg.png'.
This image is loaded by gut_simulation.py as the scene backdrop.

Run this FIRST before running gut_simulation.py.

Dependencies: matplotlib, numpy
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arc, Wedge
from matplotlib.path import Path
import matplotlib.patheffects as pe

# ── Canvas ──────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 9          # inches (matches simulation canvas exactly)
DPI = 120
WIDTH  = FIG_W * DPI          # 1920 px
HEIGHT = FIG_H * DPI          # 1080 px

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0a0a1a')   # deep navy background

# ── 1. Dark background gradient effect (lumen = centre of gut) ───────────────
# Simulate a cross-section: top = outer serosa, centre = lumen
lumen_y     = HEIGHT * 0.55          # vertical centre of the gut tube
lumen_r_x   = WIDTH  * 0.44          # half-width of lumen ellipse
lumen_r_y   = HEIGHT * 0.22          # half-height

# Outer gut wall band (lighter rose/pink region)
outer_gut = patches.Ellipse(
    (WIDTH * 0.5, lumen_y),
    width=WIDTH * 0.96, height=HEIGHT * 0.78,
    facecolor='#2d0a0a', edgecolor='#5c1a1a', linewidth=2, zorder=1
)
ax.add_patch(outer_gut)

# Muscularis layer
muscularis = patches.Ellipse(
    (WIDTH * 0.5, lumen_y),
    width=WIDTH * 0.88, height=HEIGHT * 0.68,
    facecolor='#3d1212', edgecolor='#7a2a2a', linewidth=2.5, zorder=2
)
ax.add_patch(muscularis)

# Submucosa layer
submucosa = patches.Ellipse(
    (WIDTH * 0.5, lumen_y),
    width=WIDTH * 0.80, height=HEIGHT * 0.57,
    facecolor='#4a1a1a', edgecolor='#8b3030', linewidth=2, zorder=3
)
ax.add_patch(submucosa)

# ── 2. Epithelial lining (villi edge) ──────────────────────────────────────
# Draw wavy villi along the inner wall of the lumen as a series of bumps
villi_count = 40
villi_amplitude = HEIGHT * 0.045
villi_base_r_x  = WIDTH  * 0.365
villi_base_r_y  = HEIGHT * 0.215

for i in range(villi_count):
    angle = (i / villi_count) * 2 * np.pi
    # Base of villus on epithelial circle
    bx = WIDTH  * 0.5 + villi_base_r_x * np.cos(angle)
    by = lumen_y       + villi_base_r_y * np.sin(angle)
    # Tip of villus (extends inward toward lumen centre)
    tip_scale = 0.80
    tx = WIDTH  * 0.5 + villi_base_r_x * tip_scale * np.cos(angle)
    ty = lumen_y       + villi_base_r_y * tip_scale * np.sin(angle)

    villi_patch = patches.FancyArrowPatch(
        (bx, by), (tx, ty),
        arrowstyle='-',
        linewidth=3.5,
        color='#c0504a',
        alpha=0.85,
        zorder=4
    )
    ax.add_patch(villi_patch)

# ── 3. Lumen (hollow centre of the gut) ─────────────────────────────────────
lumen = patches.Ellipse(
    (WIDTH * 0.5, lumen_y),
    width=WIDTH * 0.68, height=HEIGHT * 0.40,
    facecolor='#12081a', edgecolor='#6a1f8a', linewidth=3, zorder=5
)
ax.add_patch(lumen)

# ── 4. Mucus layer (semi-transparent gel coat on inner epithelium) ───────────
mucus = patches.Ellipse(
    (WIDTH * 0.5, lumen_y),
    width=WIDTH * 0.715, height=HEIGHT * 0.425,
    facecolor='none',
    edgecolor='#88ccaa',
    linewidth=5,
    linestyle='-',
    alpha=0.30,
    zorder=6
)
ax.add_patch(mucus)

# ── 5. Labels for gut layers ─────────────────────────────────────────────────
label_props = dict(fontsize=9, color='#cccccc', fontstyle='italic', alpha=0.80,
                   fontfamily='monospace')

ax.text(WIDTH * 0.04, HEIGHT * 0.88, 'Serosa', **label_props, zorder=20)
ax.text(WIDTH * 0.04, HEIGHT * 0.80, 'Muscularis',  **label_props, zorder=20)
ax.text(WIDTH * 0.04, HEIGHT * 0.73, 'Submucosa',  **label_props, zorder=20)
ax.text(WIDTH * 0.04, HEIGHT * 0.65, 'Epithelium\n(with villi)', **label_props, zorder=20)
ax.text(WIDTH * 0.04, HEIGHT * 0.54, 'Mucus layer', **label_props, zorder=20)
ax.text(WIDTH * 0.04, HEIGHT * 0.47, 'Lumen', **label_props, zorder=20)

# Draw small indicator lines
for (tx, ty, lx, ly) in [
    (WIDTH*0.04, HEIGHT*0.875, WIDTH*0.13, HEIGHT*0.92),
    (WIDTH*0.04, HEIGHT*0.795, WIDTH*0.12, HEIGHT*0.84),
    (WIDTH*0.04, HEIGHT*0.725, WIDTH*0.12, HEIGHT*0.76),
    (WIDTH*0.04, HEIGHT*0.645, WIDTH*0.14, HEIGHT*0.69),
    (WIDTH*0.04, HEIGHT*0.545, WIDTH*0.155, HEIGHT*0.60),
    (WIDTH*0.04, HEIGHT*0.475, WIDTH*0.165, HEIGHT*0.56),
]:
    ax.annotate('', xy=(lx, ly), xytext=(tx + 60, ty),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8),
                zorder=20)

# ── 6. Title watermark ───────────────────────────────────────────────────────
ax.text(WIDTH * 0.5, HEIGHT * 0.05,
        'Engineered Bifidobacterium IBD Therapeutic Simulation  ·  Background Asset',
        ha='center', va='bottom', fontsize=10, color='#444466',
        fontfamily='monospace', zorder=20)

# ── 7. Save ──────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
plt.savefig('gut_bg.png', dpi=DPI, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("✅  gut_bg.png saved  —  size: {0}×{1} px".format(int(FIG_W*DPI), int(FIG_H*DPI)))
print("    Run gut_simulation.py next.")
