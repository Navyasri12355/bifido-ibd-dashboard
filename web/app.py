"""
app.py — Flask backend for the IDP Gut Digital Twin web dashboard.

Routes:
  GET /                       → serves index.html
  GET /api/params             → lists all (disease_type, stage) entries
  GET /api/simulate?disease_type=X&stage=Y
                              → returns KB params + ODE time-series for the frontend
  GET /assets/<path>          → static assets (plasmid_diagram.png, gut_bg.png)
"""

import csv, os, json
from flask import Flask, jsonify, request, send_from_directory, send_file
from scipy.integrate import odeint
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)           # IDP/
KB_FILE   = os.path.join(ROOT_DIR, "knowledge_base.csv")
STATIC    = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC, template_folder=BASE_DIR)

# ── Load KB ──────────────────────────────────────────────────────────────────
_FLOAT_FIELDS = [
    "quercetin_uM", "TNF_alpha_pgmL", "IL6_pgmL", "IL1beta_pgmL",
    "barrier_integrity", "gut_permeability", "dysbiosis_score",
    "colonization_efficiency_pct",
]
_INT_FIELDS = ["n_bacteria"]

KB: dict[tuple, dict] = {}

def _load_kb():
    with open(KB_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["disease_type"].strip(), row["disease_stage"].strip())
            entry = dict(row)
            for fld in _FLOAT_FIELDS:
                entry[fld] = float(entry[fld])
            for fld in _INT_FIELDS:
                entry[fld] = int(entry[fld])
            KB[key] = entry

_load_kb()

# ── ODE system (same as gut_simulation.py) ───────────────────────────────────
TRANSIT_T0    = 5.0
TRANSIT_SLOPE = 1.8
alpha_Enz = 2.6;  beta_t = 0.9;   deg_mRNA = 1.4;  deg_prot = 0.22
K_Q_sense = 8.0;  n_hill  = 2;    K_m_enz  = 12.0; k_cat_h  = 0.55
T_HOURS   = np.linspace(0, 48, 300)

def ode_system(y, t):
    Q, mE, E, D = y
    Q  = max(Q, 0.0)
    phi     = 1.0 / (1.0 + np.exp(-TRANSIT_SLOPE * (t - TRANSIT_T0)))
    marr_off = (Q / K_Q_sense)**n_hill / (1.0 + (Q / K_Q_sense)**n_hill)
    dmE = phi * alpha_Enz * marr_off - deg_mRNA * mE
    dE  = beta_t * mE - deg_prot * E
    v   = k_cat_h * max(E, 0.0) * Q / (K_m_enz + Q)
    return [-v, dmE, dE, v]

def solve_ode(q0: float) -> dict:
    sol = odeint(ode_system, [q0, 0.0, 0.0, 0.0], T_HOURS)
    return {
        "t":     T_HOURS.tolist(),
        "Q":     np.clip(sol[:, 0], 0, None).tolist(),
        "E":     np.clip(sol[:, 2], 0, None).tolist(),
        "D":     np.clip(sol[:, 3], 0, None).tolist(),
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/api/params")
def api_params():
    result = []
    for (dt, ds), p in KB.items():
        result.append({"disease_type": dt, "stage": ds})
    return jsonify(result)

@app.route("/api/simulate")
def api_simulate():
    dtype = request.args.get("disease_type", "Healthy")
    stage = request.args.get("stage", "Healthy")
    key   = (dtype.strip(), stage.strip())
    if key not in KB:
        return jsonify({"error": f"Unknown combination: {key}"}), 404
    params = dict(KB[key])
    ode    = solve_ode(params["quercetin_uM"])
    return jsonify({"params": params, "ode": ode})

@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ROOT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5050)
