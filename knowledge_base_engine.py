"""
knowledge_base_engine.py
========================
Knowledge Base Engine for the IDP Digital Twin.

Loads literature-derived biological parameters from knowledge_base.csv
and provides them to the simulation as initial conditions.

Usage:
    from knowledge_base_engine import KnowledgeBaseEngine
    kb = KnowledgeBaseEngine()
    params = kb.get_params("Ulcerative Colitis", "Moderate")
"""

import csv
import os

KB_FILE = os.path.join(os.path.dirname(__file__), "knowledge_base.csv")

# Canonical ordering for UI dropdowns
DISEASE_TYPES  = ["Healthy", "Ulcerative Colitis", "Crohn's Disease"]
DISEASE_STAGES = {
    "Healthy":            ["Healthy"],
    "Ulcerative Colitis": ["Mild", "Moderate", "Severe"],
    "Crohn's Disease":    ["Mild", "Moderate", "Severe"],
}

_FLOAT_FIELDS = [
    "quercetin_uM", "TNF_alpha_pgmL", "IL6_pgmL", "IL1beta_pgmL",
    "barrier_integrity", "gut_permeability", "dysbiosis_score",
    "colonization_efficiency_pct",
]
_INT_FIELDS = ["n_bacteria"]


class KnowledgeBaseEngine:
    """Loads and queries the biological parameter knowledge base."""

    def __init__(self, kb_file: str = KB_FILE):
        self._db: dict[tuple, dict] = {}
        self._load(kb_file)

    # ── private ──────────────────────────────────────────────────────────────
    def _load(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["disease_type"].strip(), row["disease_stage"].strip())
                entry = dict(row)
                for field in _FLOAT_FIELDS:
                    entry[field] = float(entry[field])
                for field in _INT_FIELDS:
                    entry[field] = int(entry[field])
                self._db[key] = entry

    # ── public ────────────────────────────────────────────────────────────────
    def get_params(self, disease_type: str, disease_stage: str) -> dict:
        """
        Return the parameter dict for (disease_type, disease_stage).
        Raises KeyError if not found.
        """
        key = (disease_type.strip(), disease_stage.strip())
        if key not in self._db:
            available = list(self._db.keys())
            raise KeyError(
                f"No entry for {key!r}. Available entries: {available}"
            )
        return self._db[key]

    def list_entries(self) -> list[tuple]:
        """Return all (disease_type, disease_stage) keys in the KB."""
        return list(self._db.keys())

    def summary(self) -> str:
        """Human-readable summary of the KB."""
        lines = [f"{'Disease Type':<25} {'Stage':<12} {'Q (µM)':>8} "
                 f"{'TNF-α':>8} {'IL-6':>8} {'Barrier':>8} {'Coloniz%':>9}"]
        lines.append("-" * 75)
        for (dt, ds), p in self._db.items():
            lines.append(
                f"{dt:<25} {ds:<12} {p['quercetin_uM']:>8.0f} "
                f"{p['TNF_alpha_pgmL']:>8.1f} {p['IL6_pgmL']:>8.1f} "
                f"{p['barrier_integrity']:>8.2f} {p['colonization_efficiency_pct']:>9.0f}"
            )
        return "\n".join(lines)


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    kb = KnowledgeBaseEngine()
    print(kb.summary())
    print()
    p = kb.get_params("Ulcerative Colitis", "Moderate")
    print(f"UC Moderate -> Q={p['quercetin_uM']} uM, "
          f"n_bacteria={p['n_bacteria']}, "
          f"colonization={p['colonization_efficiency_pct']}%")
    print(f"  Reference DOI: {p['reference_doi']}")
    print(f"  PMID: {p['pmid']}")
