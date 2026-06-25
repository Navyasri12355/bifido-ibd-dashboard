"""
gut_simulation.py
=================
Main 2D animated simulation dashboard.

LAYOUT  (3-column × 2-row):
┌─────────────────┬─────────────────┬──────────────────┐
│  Healthy Gut    │  IBD Gut        │  Plasmid Circuit │
│  (static ref)   │  (animated)     │  (diagram inset) │
├─────────────────┴─────────────────┼──────────────────┤
│  ODE kinetic traces               │  Molecule legend │
│  (Quercetin, Enzyme, DOPAC)       │  & status panel  │
└───────────────────────────────────┴──────────────────┘

Molecules (colour-coded dots in the lumen):
  ● Crimson   = Active Quercetin (inflammatory marker / IBD flare)
  ● Orange    = Quercetin bound to MarR (sensing / circuit activation)
  ● DodgerBlue= DOPAC (therapeutic metabolite, resolved)
  ● LimeGreen = Bifidobacterium cells (with glow when enzyme-active)
  ● White α   = Neutral luminal proteins / background noise

PREREQUISITES — run in order:
  1. python gut_background.py     →  gut_bg.png
  2. python plasmid_diagram.py    →  plasmid_diagram.png
  3. python gut_simulation.py     →  displays animation (save optional)

Dependencies:
    pip install matplotlib numpy scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.image import imread
from matplotlib.widgets import RadioButtons
from scipy.integrate import odeint
import os

# Knowledge Base Engine
from knowledge_base_engine import KnowledgeBaseEngine, DISEASE_TYPES, DISEASE_STAGES
_KB = KnowledgeBaseEngine()

# ── Active KB selection (mutable via widget) ──────────────────────────────────
_ACTIVE = {"disease_type": "Ulcerative Colitis", "disease_stage": "Moderate"}

np.random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
# 0.  STYLE CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
BG_COLOR       = '#0a0a1a'
PANEL_COLOR    = '#0f0f28'
TEXT_COLOR     = '#d0d8f0'
GRID_COLOR     = '#1e1e3a'

C_QUERCETIN    = '#e74c3c'   # crimson   – active inflammation
C_SENSING      = '#e67e22'   # orange    – MarR-bound / sensing
C_DOPAC        = '#3498db'   # blue      – therapeutic product
C_BACTERIA_OFF = '#1a6b3a'   # dark green– inactive Bifidobacterium
C_BACTERIA_ON  = '#2ecc71'   # lime green– enzyme-producing Bifidobacterium
C_NOISE        = '#ffffff'   # white     – neutral luminal particles

# ═════════════════════════════════════════════════════════════════════════════
# 1.  ODE KINETICS  –  Literature-informed 5-phase timeline  (units: hours)
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1  (0–4 h)  : Gastric transit – circuit silent, no quercetin sensing
# Phase 2  (4–8 h)  : Colon arrival, MarR sensing begins – trace activation
# Phase 3  (8–12 h) : Gene expression, dioxygenase accumulates, DOPAC rises
# Phase 4  (12–24 h): Active metabolism – peak conversion rate
# Phase 5  (24–48 h): Steady state – quercetin approaches residual level
#
# Three-state model: Q (quercetin, µM), E (dioxygenase enzyme, µM), D (DOPAC, µM)
# ─────────────────────────────────────────────────────────────────────────────
TRANSIT_T0    = 5.0     # h  – sigmoid centre (bacteria reach colon)
TRANSIT_SLOPE = 1.8     # /h – sigmoid steepness → 10–90 % transition over ~2.5 h

alpha_Enz = 2.6         # /h – max dioxygenase mRNA synthesis
beta_t    = 0.9         # /h – translation rate
deg_mRNA  = 1.4         # /h – mRNA degradation  (t½ ≈ 30 min)
deg_prot  = 0.22        # /h – protein degradation (t½ ≈ 3.2 h)
K_Q_sense = 8.0         # µM – quercetin half-activation of MarR
n_hill    = 2           # Hill coefficient
K_m_enz   = 12.0        # µM – Michaelis constant
k_cat_h   = 0.55        # /h – enzyme turnover (slow → spread over 12-24 h)

def ode_system(y, t):
    Q, mE, E, D = y
    Q = max(Q, 0.0)
    # Transit gate: bacteria absent before ~4 h, present after ~6 h
    phi = 1.0 / (1.0 + np.exp(-TRANSIT_SLOPE * (t - TRANSIT_T0)))
    # MarR inactivation by quercetin → Hill-type enzyme expression
    marr_off = (Q / K_Q_sense)**n_hill / (1.0 + (Q / K_Q_sense)**n_hill)
    dmE_dt = phi * alpha_Enz * marr_off - deg_mRNA * mE
    dE_dt  = beta_t * mE - deg_prot * E
    v      = k_cat_h * max(E, 0.0) * Q / (K_m_enz + Q)
    dQ_dt  = -v
    dD_dt  =  v
    return [dQ_dt, dmE_dt, dE_dt, dD_dt]

T = np.linspace(0, 48, 300)          # 0–48 h, 300 ODE points → 120 frames

def _solve_ode(kb_params: dict):
    """Solve ODE with initial conditions from KB parameters."""
    Q0 = kb_params["quercetin_uM"]
    y0_local = [Q0, 0.0, 0.0, 0.0]   # Q, mE, E, D
    sol_local = odeint(ode_system, y0_local, T)
    return sol_local

# Initial solve using default KB selection
_kb_params = _KB.get_params(_ACTIVE["disease_type"], _ACTIVE["disease_stage"])
y0  = [_kb_params["quercetin_uM"], 0.0, 0.0, 0.0]
sol = _solve_ode(_kb_params)

Q_trace = sol[:, 0]   # Quercetin
E_trace = sol[:, 2]   # Dioxygenase enzyme
D_trace = sol[:, 3]   # DOPAC

# ═════════════════════════════════════════════════════════════════════════════
# 2.  SPATIAL PARTICLE SIMULATION SETUP
#     Coordinate space: [0..1] × [0..1] (normalised).
#     Lumen occupies  x ∈ [0.1, 0.9],  y ∈ [0.20, 0.80] (ellipse).
# ═════════════════════════════════════════════════════════════════════════════
N_FRAMES    = 120
N_BACTERIA  = _kb_params["n_bacteria"]
N_QUERCETIN = 160
N_NOISE     = 40

LUMEN_CX, LUMEN_CY = 0.50, 0.50
LUMEN_RX, LUMEN_RY = 0.38, 0.28

def random_in_ellipse(n, cx, cy, rx, ry, rng=None):
    """Uniform random points inside an ellipse."""
    if rng is None:
        rng = np.random
    pts = []
    while len(pts) < n:
        x = rng.uniform(cx - rx, cx + rx, n * 4)
        y = rng.uniform(cy - ry, cy + ry, n * 4)
        inside = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2) <= 1.0
        for xi, yi, ok in zip(x, y, inside):
            if ok:
                pts.append([xi, yi])
            if len(pts) == n:
                break
    return np.array(pts[:n])

# Bacteria: placed near the epithelial wall (ellipse perimeter, inner edge)
def bacteria_on_wall(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles += np.random.uniform(-0.05, 0.05, n)
    wall_r = 0.90   # fraction of lumen radius – just inside epithelium
    bx = LUMEN_CX + LUMEN_RX * wall_r * np.cos(angles)
    by = LUMEN_CY + LUMEN_RY * wall_r * np.sin(angles)
    return np.stack([bx, by], axis=1), angles

bact_pos, bact_angles = bacteria_on_wall(N_BACTERIA)
bact_enzyme_level     = np.zeros(N_BACTERIA)   # 0..5, grows as Quercetin sensed

# Particles: positions, states
# state: 1=Quercetin(red)  2=Sensing(orange)  3=DOPAC(blue)
q_pos   = random_in_ellipse(N_QUERCETIN, LUMEN_CX, LUMEN_CY, LUMEN_RX*0.85, LUMEN_RY*0.85)
q_state = np.ones(N_QUERCETIN, dtype=int)

# Noise (neutral) particles
noise_pos = random_in_ellipse(N_NOISE, LUMEN_CX, LUMEN_CY, LUMEN_RX*0.80, LUMEN_RY*0.80)

INTERACT_R    = 0.055   # radius within which Quercetin "sees" a bacterium
K_CAT_PARTICLE= 0.12    # per-frame conversion probability scaling

# Pre-seed healthy-gut particles (always DOPAC, no inflammation)
healthy_pos   = random_in_ellipse(N_QUERCETIN // 3,
                                  LUMEN_CX, LUMEN_CY,
                                  LUMEN_RX * 0.80, LUMEN_RY * 0.80)

# ═════════════════════════════════════════════════════════════════════════════
# 3.  LOAD EXTERNAL ASSETS
# ═════════════════════════════════════════════════════════════════════════════
def try_load(path, label):
    if os.path.exists(path):
        return imread(path)
    else:
        print(f"⚠️  {label} not found ({path}). "
              f"Run the prerequisite script first, or the panel will be blank.")
        return None

gut_bg_img      = try_load('gut_bg.png',         'gut_bg.png')
plasmid_img     = try_load('plasmid_diagram.png', 'plasmid_diagram.png')

# ═════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 9), facecolor=BG_COLOR)

# ── Left selector strip (disease type + stage) ────────────────────────────────
ax_type  = fig.add_axes([0.00, 0.55, 0.085, 0.30], facecolor='#0c0c22')
ax_stage = fig.add_axes([0.00, 0.20, 0.085, 0.30], facecolor='#0c0c22')
ax_type.set_title('Disease\nType', color='#88aaff', fontsize=7,
                   fontfamily='monospace', pad=3)
ax_stage.set_title('Stage', color='#88aaff', fontsize=7,
                    fontfamily='monospace', pad=3)

for ax in [ax_type, ax_stage]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a4a')

rb_type  = RadioButtons(ax_type,  DISEASE_TYPES,  active=1,
                         activecolor='#ff9966')
rb_stage = RadioButtons(ax_stage, ["Mild", "Moderate", "Severe"], active=1,
                         activecolor='#ff9966')
for rb in [rb_type, rb_stage]:
    for label in rb.labels:
        label.set_fontsize(7.5)
        label.set_fontfamily('monospace')
        label.set_color('#c8d8ff')

def _set_stage_enabled(enabled: bool) -> None:
    """Grey-out stage radio buttons when Healthy is selected."""
    dim_color   = '#3a3a5a'   # muted colour when disabled
    label_color = '#c8d8ff'   # normal colour
    for label in rb_stage.labels:
        label.set_color(label_color if enabled else dim_color)
    ax_stage.set_title(
        'Stage', color=('#88aaff' if enabled else '#3a3a5a'),
        fontsize=7, fontfamily='monospace', pad=3
    )

# ── KB info text strip (citation / parameters) ───────────────────────────────
ax_kbinfo = fig.add_axes([0.00, 0.06, 0.085, 0.13], facecolor='#070712')
ax_kbinfo.set_xticks([]); ax_kbinfo.set_yticks([])
for spine in ax_kbinfo.spines.values():
    spine.set_edgecolor('#1e2a4a')
kb_info_text = ax_kbinfo.text(
    0.05, 0.95, '', transform=ax_kbinfo.transAxes,
    fontsize=5.5, color='#8899bb', fontfamily='monospace',
    va='top', wrap=True
)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[2, 2, 1.1],
    height_ratios=[1.6, 1],
    hspace=0.08,
    wspace=0.06,
    left=0.10, right=0.97,
    top=0.87, bottom=0.07,
)

ax_healthy   = fig.add_subplot(gs[0, 0])   # Healthy gut panel
ax_ibd       = fig.add_subplot(gs[0, 1])   # IBD + treatment panel
ax_plasmid   = fig.add_subplot(gs[0, 2])   # Plasmid diagram
ax_kinetics  = fig.add_subplot(gs[1, 0:2]) # ODE traces
ax_legend    = fig.add_subplot(gs[1, 2])   # Legend + status

for ax in [ax_healthy, ax_ibd, ax_plasmid, ax_kinetics, ax_legend]:
    ax.set_facecolor(PANEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a4a')
        spine.set_linewidth(0.8)

# ═════════════════════════════════════════════════════════════════════════════
# 5.  HELPER: draw gut background into an axes
# ═════════════════════════════════════════════════════════════════════════════
def draw_gut_bg(ax, img=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if img is not None:
        ax.imshow(img, extent=[0, 1, 0, 1], aspect='auto', zorder=0, alpha=0.55)
    else:
        # Fallback: draw simple concentric ellipses
        for (rx, ry, col) in [
            (0.48, 0.45, '#2d0a0a'), (0.44, 0.41, '#3d1212'),
            (0.40, 0.36, '#4a1a1a'), (0.37, 0.32, '#c0504a'),
        ]:
            ep = mpatches.Ellipse((0.5, 0.5), 2*rx, 2*ry,
                                  facecolor=col, edgecolor='#7a3030',
                                  linewidth=1.2, zorder=1)
            ax.add_patch(ep)
        lumen = mpatches.Ellipse((0.5, 0.5), 2*LUMEN_RX, 2*LUMEN_RY,
                                 facecolor='#12081a', edgecolor='#6a1f8a',
                                 linewidth=2, zorder=2)
        ax.add_patch(lumen)

# ─── Healthy gut panel (static) ──────────────────────────────────────────────
draw_gut_bg(ax_healthy, gut_bg_img)

# Scatter a few DOPAC dots to represent a healthy state
ax_healthy.scatter(
    healthy_pos[:, 0], healthy_pos[:, 1],
    s=18, c=C_DOPAC, alpha=0.65, zorder=5, linewidths=0
)

# Healthy bacteria (calm green, small)
ax_healthy.scatter(
    bact_pos[:, 0], bact_pos[:, 1],
    s=38, c=C_BACTERIA_OFF, alpha=0.80, zorder=6,
    edgecolors='#aaffaa', linewidths=0.5
)
ax_healthy.set_title('Healthy Gut  (Reference)',
                     color='#88ffcc', fontsize=10, fontweight='bold',
                     fontfamily='monospace', pad=4)
ax_healthy.text(0.5, 0.02, 'No inflammatory Quercetin  ·  Normal microbiome',
                ha='center', va='bottom', transform=ax_healthy.transAxes,
                fontsize=7.5, color='#8899cc', fontfamily='monospace')

# ─── IBD panel (animated) ────────────────────────────────────────────────────
draw_gut_bg(ax_ibd, gut_bg_img)
ax_ibd.set_title('IBD Gut  →  Engineered Bifidobacterium Treatment',
                  color='#ff9966', fontsize=10, fontweight='bold',
                  fontfamily='monospace', pad=4)

# Particle scatters (updated each frame)
scat_q  = ax_ibd.scatter([], [], s=22, c=C_QUERCETIN, alpha=0.75, zorder=5, linewidths=0)
scat_s  = ax_ibd.scatter([], [], s=26, c=C_SENSING,   alpha=0.85, zorder=6, linewidths=0)
scat_d  = ax_ibd.scatter([], [], s=18, c=C_DOPAC,     alpha=0.70, zorder=5, linewidths=0)
scat_n  = ax_ibd.scatter(noise_pos[:, 0], noise_pos[:, 1],
                          s=8, c=C_NOISE, alpha=0.10, zorder=3, linewidths=0)
scat_b  = ax_ibd.scatter(bact_pos[:, 0], bact_pos[:, 1],
                          s=55, c=C_BACTERIA_OFF,
                          alpha=0.85, zorder=7,
                          edgecolors='#aaffaa', linewidths=0.7)

# Inflammation heatmap (fades as treatment proceeds) – simple circle glow
inflam_circle = mpatches.Ellipse(
    (LUMEN_CX, LUMEN_CY), 2*LUMEN_RX*0.70, 2*LUMEN_RY*0.70,
    facecolor='#ff000020', edgecolor='none', zorder=1
)
ax_ibd.add_patch(inflam_circle)

# Time-step label inside IBD panel
time_label = ax_ibd.text(
    0.5, 0.96, 'T = 0.0 h',
    ha='center', va='top', transform=ax_ibd.transAxes,
    fontsize=9, color='#ffcc88', fontfamily='monospace', zorder=10
)

# Inflammation bar
inflam_bar_bg, = ax_ibd.plot([0.10, 0.90], [0.03, 0.03],
                              color='#330000', lw=7, alpha=0.6, zorder=8,
                              transform=ax_ibd.transAxes)
inflam_bar, = ax_ibd.plot([], [], lw=7, color='#ff3333',
                           zorder=9, transform=ax_ibd.transAxes, alpha=0.85)
inflam_label = ax_ibd.text(0.5, 0.055, 'Inflammation: 100%',
                             ha='center', va='bottom', transform=ax_ibd.transAxes,
                             fontsize=7.5, color='#ff9999', fontfamily='monospace', zorder=10)

# ─── Plasmid panel ───────────────────────────────────────────────────────────
ax_plasmid.set_xticks([])
ax_plasmid.set_yticks([])
if plasmid_img is not None:
    ax_plasmid.imshow(plasmid_img, aspect='auto', zorder=1)
else:
    ax_plasmid.text(0.5, 0.5, 'Run\nplasmid_diagram.py\nfirst',
                    ha='center', va='center', color='#9988aa',
                    fontsize=9, transform=ax_plasmid.transAxes)
ax_plasmid.set_title('Engineered Plasmid: pDOJH10S',
                      color='#cc99ff', fontsize=9, fontweight='bold',
                      fontfamily='monospace', pad=4)

# ─── Kinetics panel ──────────────────────────────────────────────────────────
ax_kinetics.set_facecolor(PANEL_COLOR)
ax_kinetics.tick_params(colors=TEXT_COLOR, labelsize=8)
ax_kinetics.set_xlim(0, 48)
ax_kinetics.set_ylim(-2, 55)
ax_kinetics.set_xlabel('Time (hours)', color=TEXT_COLOR, fontsize=9, fontfamily='monospace')
ax_kinetics.set_ylabel('Concentration (µM)', color=TEXT_COLOR, fontsize=9, fontfamily='monospace')
ax_kinetics.set_title('ODE Kinetic Traces  –  5-Phase Biological Timeline',
                       color=TEXT_COLOR, fontsize=9, fontweight='bold', fontfamily='monospace')
ax_kinetics.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle='--')
for spine in ax_kinetics.spines.values():
    spine.set_edgecolor('#1e2a4a')

# Static full traces (faint)
ax_kinetics.plot(T, Q_trace, color=C_QUERCETIN, lw=0.8, alpha=0.2)
ax_kinetics.plot(T, E_trace, color='#e67e22',   lw=0.8, alpha=0.2)
ax_kinetics.plot(T, D_trace, color=C_DOPAC,     lw=0.8, alpha=0.2)

# Dynamic animated lines
line_q, = ax_kinetics.plot([], [], color=C_QUERCETIN, lw=2.0, label='Quercetin (inflammatory)')
line_e, = ax_kinetics.plot([], [], color='#e67e22',   lw=1.6, label='Quercetin Dioxygenase (enzyme)')
line_d, = ax_kinetics.plot([], [], color=C_DOPAC,     lw=2.0, label='DOPAC (therapeutic)')

kin_marker_q, = ax_kinetics.plot([], [], 'o', color=C_QUERCETIN, ms=6, zorder=10)
kin_marker_e, = ax_kinetics.plot([], [], 'o', color='#e67e22',   ms=6, zorder=10)
kin_marker_d, = ax_kinetics.plot([], [], 'o', color=C_DOPAC,     ms=6, zorder=10)

ax_kinetics.legend(loc='upper right', fontsize=8, framealpha=0.2,
                    labelcolor='white', facecolor='#0a0a1a', edgecolor='#333355')

# Current time vline
vline = ax_kinetics.axvline(x=0, color='#ffffff', lw=0.8, alpha=0.4, linestyle=':')

# ─── Legend / status panel ───────────────────────────────────────────────────
ax_legend.set_xticks([])
ax_legend.set_yticks([])
ax_legend.set_title('Molecule Key', color='#aabbdd', fontsize=9,
                     fontweight='bold', fontfamily='monospace')

legend_entries = [
    (C_QUERCETIN, '●', 'Quercetin  (IBD flare)'),
    (C_SENSING,   '●', 'Quercetin–MarR complex\n(circuit sensing)'),
    (C_DOPAC,     '●', 'DOPAC  (therapeutic\nresolution)'),
    (C_BACTERIA_ON,'●','Bifidobacterium  (enzyme\nproducing)'),
    (C_BACTERIA_OFF,'●','Bifidobacterium  (idle)'),
    ('#ffffff',   '●', 'Neutral luminal\nproteins / noise'),
]
for i, (col, sym, lbl) in enumerate(legend_entries):
    y = 0.90 - i * 0.145
    ax_legend.text(0.10, y, sym, color=col, fontsize=15,
                    transform=ax_legend.transAxes, va='center')
    ax_legend.text(0.22, y, lbl, color=TEXT_COLOR, fontsize=7.8,
                    transform=ax_legend.transAxes, va='center',
                    fontfamily='monospace')

# Status text
status_text = ax_legend.text(0.5, 0.03, 'Status: IBD Flare Active',
                               ha='center', va='bottom',
                               transform=ax_legend.transAxes,
                               fontsize=8, color='#ff6666',
                               fontweight='bold', fontfamily='monospace')

# ─── Figure suptitle ─────────────────────────────────────────────────────────
fig.suptitle(
    'Synthetic Bifidobacterium  ·  IDP Gut Simulation Dashboard\n'
    'MarR–PhlF Double-Inverter Circuit  |  Quercetin → DOPAC Therapeutic Conversion',
    color='#c8d8ff', fontsize=10, fontweight='bold',
    fontfamily='monospace', y=0.98
)

# ═════════════════════════════════════════════════════════════════════════════
# 6.  ANIMATION UPDATE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════
def update(frame):
    global q_pos, q_state, bact_enzyme_level

    # Map animation frame → ODE time index (300 ODE pts, 120 frames)
    ode_idx = min(int(frame / N_FRAMES * len(T)), len(T) - 1)
    t_now   = T[ode_idx]

    # ── A. Particle physics ─────────────────────────────────────────────────
    for i in range(N_QUERCETIN):
        if q_state[i] < 3:
            # Brownian motion (slightly slower for visual clarity)
            step = np.random.randn(2) * 0.008
            q_pos[i] += step

            # Clamp to lumen ellipse
            dx = (q_pos[i, 0] - LUMEN_CX) / LUMEN_RX
            dy = (q_pos[i, 1] - LUMEN_CY) / LUMEN_RY
            if dx**2 + dy**2 > 0.88:   # slightly inside wall
                norm = np.sqrt(dx**2 + dy**2)
                q_pos[i, 0] = LUMEN_CX + LUMEN_RX * 0.88 * dx / norm
                q_pos[i, 1] = LUMEN_CY + LUMEN_RY * 0.88 * dy / norm

            # Check proximity to any bacterium
            dists  = np.linalg.norm(bact_pos - q_pos[i], axis=1)
            nearest = np.argmin(dists)
            if dists[nearest] < INTERACT_R:
                if q_state[i] == 1:
                    q_state[i] = 2          # enter sensing state

                # Bacteria upregulate enzyme (capped at 5)
                bact_enzyme_level[nearest] = min(
                    bact_enzyme_level[nearest] + 0.08, 5.0
                )

                # Conversion probability scales with enzyme level AND ODE time
                # (uses real enzyme level from ODE to stay honest)
                enz_now = E_trace[ode_idx]
                p_conv  = K_CAT_PARTICLE * bact_enzyme_level[nearest] * (enz_now / 30.0)
                if np.random.rand() < p_conv:
                    q_state[i] = 3          # converted to DOPAC

    # ── A2. Bulk enzymatic clearance (diffuse enzyme, no proximity needed) ───
    # Gated by transit + ODE enzyme level so phases 1-2 stay visually quiescent.
    phi_now        = 1.0 / (1.0 + np.exp(-TRANSIT_SLOPE * (t_now - TRANSIT_T0)))
    enz_now_global = E_trace[ode_idx]
    if phi_now > 0.15 and enz_now_global > 0.05:
        # Scale p_bulk by ODE enzyme (peaks ~8-10 µM at phase 4)
        p_bulk = 0.005 * (enz_now_global / 10.0) * phi_now
        unconverted = np.where(q_state < 3)[0]
        converts = np.random.rand(len(unconverted)) < p_bulk
        q_state[unconverted[converts]] = 3

    # ── B. Inflammation level (fraction of still-active Quercetin) ──────────
    n_active  = np.sum(q_state == 1)
    n_sensing = np.sum(q_state == 2)
    n_dopac   = np.sum(q_state == 3)
    inflam_frac = (n_active + n_sensing * 0.4) / N_QUERCETIN

    # ── C. Update IBD particle scatters ─────────────────────────────────────
    mask1 = q_state == 1
    mask2 = q_state == 2
    mask3 = q_state == 3

    scat_q.set_offsets(q_pos[mask1] if np.any(mask1) else np.empty((0, 2)))
    scat_s.set_offsets(q_pos[mask2] if np.any(mask2) else np.empty((0, 2)))
    scat_d.set_offsets(q_pos[mask3] if np.any(mask3) else np.empty((0, 2)))

    # Bacteria colour (greener as enzyme level rises)
    bact_colors = [
        (0.10, 0.42 + (lev / 5.0) * 0.58, 0.20 + (lev / 5.0) * 0.25)
        for lev in bact_enzyme_level
    ]
    scat_b.set_color(bact_colors)

    # Inflammation glow alpha
    inflam_circle.set_alpha(inflam_frac * 0.35)
    inflam_circle.set_facecolor(f'#ff0000{int(inflam_frac*60):02x}')

    # Inflammation bar (spans 0.10 → 0.10 + 0.80*inflam_frac in axes coords)
    bar_xmax = 0.10 + 0.80 * inflam_frac
    inflam_bar.set_data([0.10, bar_xmax], [0.03, 0.03])

    pct = int(inflam_frac * 100)
    inflam_label.set_text(f'Inflammation: {pct}%')
    col_lerp = (inflam_frac, 0.15, 0.15)
    inflam_label.set_color(col_lerp)

    # Time label
    time_label.set_text(f'T = {t_now:.1f} h')

    # ── D. Kinetics traces ──────────────────────────────────────────────────
    line_q.set_data(T[:ode_idx+1], Q_trace[:ode_idx+1])
    line_e.set_data(T[:ode_idx+1], E_trace[:ode_idx+1])
    line_d.set_data(T[:ode_idx+1], D_trace[:ode_idx+1])

    kin_marker_q.set_data([t_now], [Q_trace[ode_idx]])
    kin_marker_e.set_data([t_now], [E_trace[ode_idx]])
    kin_marker_d.set_data([t_now], [D_trace[ode_idx]])

    vline.set_xdata([t_now, t_now])

    # ── E. Status text  (inflam_frac – particle visual state)
    if inflam_frac > 0.75:
        status_text.set_text('Status: IBD Flare Active')
        status_text.set_color('#ff4444')
    elif inflam_frac > 0.50:
        status_text.set_text('Status: Treatment Engaging')
        status_text.set_color('#ffaa33')
    elif inflam_frac > 0.15:
        status_text.set_text('Status: Inflammation Receding')
        status_text.set_color('#ffdd44')
    else:
        status_text.set_text('Status: Flare Resolved')
        status_text.set_color('#44ff88')

    return (scat_q, scat_s, scat_d, scat_b,
            inflam_circle, inflam_bar, inflam_label,
            time_label, line_q, line_e, line_d,
            kin_marker_q, kin_marker_e, kin_marker_d,
            vline, status_text)

# ═════════════════════════════════════════════════════════════════════════════
# 7.  RUN ANIMATION
# ═════════════════════════════════════════════════════════════════════════════
anim = FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    interval=160,          # ms between frames  (~6 fps – clear to watch)
    blit=True,
    repeat=True,
)

# ═════════════════════════════════════════════════════════════════════════════
# 8.  KNOWLEDGE BASE SELECTOR CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════
def _refresh_kb_info(params: dict) -> None:
    """Update the KB info text box with the active parameter set."""
    txt = (
        f"Q\u2080 = {params['quercetin_uM']:.0f} \u00b5M\n"
        f"TNF-\u03b1 = {params['TNF_alpha_pgmL']:.1f} pg/mL\n"
        f"IL-6  = {params['IL6_pgmL']:.1f} pg/mL\n"
        f"Barrier: {params['barrier_integrity']:.2f}\n"
        f"Dysbios: {params['dysbiosis_score']:.2f}\n"
        f"Coloniz: {params['colonization_efficiency_pct']:.0f}%\n"
        f"Bact: {params['n_bacteria']}\n"
        f"PMID: {params['pmid']}"
    )
    kb_info_text.set_text(txt)
    fig.canvas.draw_idle()


def _on_selection_changed(_: str) -> None:
    """Called when either RadioButton changes; re-solves ODE + resets sim."""
    global Q_trace, E_trace, D_trace, q_pos, q_state, bact_pos, bact_enzyme_level

    dtype  = rb_type.value_selected
    is_healthy = (dtype == "Healthy")

    # Healthy has no stages – map to its only stage and lock the stage widget
    _set_stage_enabled(not is_healthy)
    stage = "Healthy" if is_healthy else rb_stage.value_selected

    _ACTIVE["disease_type"]  = dtype
    _ACTIVE["disease_stage"] = stage

    try:
        params = _KB.get_params(dtype, stage)
    except KeyError:
        return

    # Re-solve ODE
    new_sol    = _solve_ode(params)
    Q_trace[:] = new_sol[:, 0]
    E_trace[:] = new_sol[:, 2]
    D_trace[:] = new_sol[:, 3]

    # Reset particle states
    q_pos[:] = random_in_ellipse(N_QUERCETIN, LUMEN_CX, LUMEN_CY,
                                  LUMEN_RX * 0.85, LUMEN_RY * 0.85)
    if is_healthy:
        # Healthy gut: mostly DOPAC but a small residual quercetin fraction
        # reflects the low-but-nonzero Q₀ visible in the ODE trace (~15 % of particles)
        HEALTHY_Q_FRAC = 0.15
        n_quercetin_healthy = max(1, int(N_QUERCETIN * HEALTHY_Q_FRAC))
        q_state[:] = 3                               # default all to DOPAC
        q_state[:n_quercetin_healthy] = 1            # first slice → residual quercetin
        np.random.shuffle(q_state)                   # scatter them randomly
        n_bact = params["n_bacteria"]
        bact_pos, _ = bacteria_on_wall(n_bact)
        bact_enzyme_level = np.full(n_bact, 5.0)    # bacteria fully active, no flare
    else:
        q_state[:] = 1                              # IBD: start with full inflammation
        n_bact = params["n_bacteria"]
        bact_pos, _ = bacteria_on_wall(n_bact)
        bact_enzyme_level = np.zeros(n_bact)

    # Update static traces (faint guide lines)
    ax_kinetics.lines[0].set_data(T, Q_trace)
    ax_kinetics.lines[1].set_data(T, E_trace)
    ax_kinetics.lines[2].set_data(T, D_trace)

    # Update bacteria scatter
    scat_b.set_offsets(bact_pos)
    scat_b.set_sizes([55] * n_bact)

    # Update IBD panel title to reflect mode
    if is_healthy:
        ax_ibd.set_title('Healthy Gut  →  No Inflammatory Load',
                          color='#88ffcc', fontsize=10, fontweight='bold',
                          fontfamily='monospace', pad=4)
    else:
        ax_ibd.set_title('IBD Gut  →  Engineered Bifidobacterium Treatment',
                          color='#ff9966', fontsize=10, fontweight='bold',
                          fontfamily='monospace', pad=4)

    # Update suptitle
    fig.suptitle(
        f'Synthetic Bifidobacterium  \u00b7  IDP Gut Simulation Dashboard\n'
        f'MarR\u2013PhlF Double-Inverter Circuit  |  '
        f'{dtype} \u2013 {stage}  |  Q\u2080 = {params["quercetin_uM"]:.0f} \u00b5M',
        color='#c8d8ff', fontsize=10, fontweight='bold',
        fontfamily='monospace', y=0.98
    )

    _refresh_kb_info(params)
    anim.frame_seq = anim.new_saved_frame_seq()  # restart frame counter
    fig.canvas.draw_idle()


# Wire up callbacks
rb_type.on_clicked(_on_selection_changed)
rb_stage.on_clicked(_on_selection_changed)

# Initial KB info display
_refresh_kb_info(_kb_params)

# Initial suptitle with KB info
fig.suptitle(
    f'Synthetic Bifidobacterium  \u00b7  IDP Gut Simulation Dashboard\n'
    f'MarR\u2013PhlF Double-Inverter Circuit  |  '
    f'{_ACTIVE["disease_type"]} \u2013 {_ACTIVE["disease_stage"]}  |  '
    f'Q\u2080 = {_kb_params["quercetin_uM"]:.0f} \u00b5M',
    color='#c8d8ff', fontsize=10, fontweight='bold',
    fontfamily='monospace', y=0.98
)

plt.show()
print("Done.")
