# 4_Calibration.py
# Calibration — raking to admin margins (counts-first, CSV/Excel targets)
# Streamlit-compatible across versions (uses use_container_width=True)

import io
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Calibration — raking to admin margins", layout="wide")
st.title("4) Calibration — raking to admin margins (counts-first)")

# ---------------------------
# Compat helpers
# ---------------------------
def show_df(df: pd.DataFrame, head: int | None = None):
    """Version-safe dataframe display."""
    d = df.head(head) if head is not None else df
    # Older/newer Streamlit both accept use_container_width=True
    st.dataframe(d, use_container_width=True)

# ---------------------------
# Tiny IPF (raking) utility
# ---------------------------
def rake_ipf(
    df: pd.DataFrame,
    margins_props: Dict[str, Dict[str, float]],
    weight_col: str = "weight",
    max_iter: int = 200,
    tol: float = 1e-8,
    cap: Optional[float] = None,
) -> np.ndarray:
    """Iterative proportional fitting on categorical margins."""
    w = df[weight_col].astype(float).to_numpy().copy()
    w = np.maximum(w, 0.0)
    if not np.isfinite(w).all():
        raise ValueError("Weight column contains non-finite values.")

    # Precompute masks
    masks: Dict[Tuple[str, str], np.ndarray] = {}
    for col, cats in margins_props.items():
        if col not in df.columns:
            raise ValueError(f"Margin column '{col}' not found in data.")
        s = df[col].astype(str)
        for cat in cats.keys():
            masks[(col, str(cat))] = (s.values == str(cat))

    # IPF loop
    for _ in range(max_iter):
        w_old = w.copy()
        total = w.sum()
        if total <= 0:
            raise ValueError("Total weight is zero during raking; check inputs.")

        for col, cats in margins_props.items():
            total = w.sum()
            for cat, p_target in cats.items():
                mask = masks[(col, str(cat))]
                if mask.sum() == 0:
                    continue
                current = w[mask].sum()
                desired = p_target * total
                factor = 1.0 if current == 0 else desired / current
                if cap is not None:
                    factor = max(min(factor, cap), 1.0 / cap)
                w[mask] *= factor

        denom = np.abs(w_old).sum()
        delta = np.abs(w - w_old).sum() / (denom if denom > 0 else 1.0)
        if delta < tol:
            break
    return w

# ---------------------------
# Helpers
# ---------------------------
def infer_categoricals(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("category")]

def build_example_targets(df: pd.DataFrame, sel_cols: List[str]) -> str:
    """Create CSV-like example targets (COUNTS)."""
    lines = ["margin,category,target"]
    n = len(df)
    for col in sel_cols:
        cats = sorted(df[col].astype(str).unique())
        if not cats:
            continue
        base = n / len(cats)
        cum = 0.0
        for i, cat in enumerate(cats):
            if i == len(cats) - 1:
                lines.append(f"{col},{cat},{round(n - cum, 0)}")
            else:
                val = round(base, 0)
                lines.append(f"{col},{cat},{val}")
                cum += val
    return "\n".join(lines)

def parse_targets_block(text: str) -> Dict[str, Dict[str, float]]:
    """Parse CSV target block into normalized proportions (handles counts or %)."""
    if not text.strip():
        raise ValueError("Targets text is empty.")
    df_t = pd.read_csv(io.StringIO(text.strip()), sep=",", header=0)
    df_t.columns = [c.strip().lower() for c in df_t.columns]

    col_margin = next((c for c in df_t.columns if c in ("margin", "column", "col")), None)
    col_cat = next((c for c in df_t.columns if c in ("category", "cat", "value")), None)
    col_target = next((c for c in df_t.columns if c in ("target", "target_%", "percent", "pct", "count", "counts", "weight")), None)
    if not (col_margin and col_cat and col_target):
        raise ValueError("Targets must include columns: margin, category, target.")

    t = df_t[[col_margin, col_cat, col_target]].copy()
    # Allow "50%" or numbers
    t[col_target] = pd.to_numeric(t[col_target].astype(str).str.replace("%", ""), errors="coerce")
    if t[col_target].isna().any():
        raise ValueError("Some target values are non-numeric.")

    props: Dict[str, Dict[str, float]] = {}
    for margin, grp in t.groupby(col_margin):
        vals = grp[col_target].to_numpy(float)
        s = np.nansum(vals)
        denom = 100.0 if abs(s - 100) <= 0.5 else (s if s > 0 else 1.0)
        cats = grp[col_cat].astype(str).tolist()
        p = [float(v) / denom for v in vals]
        s2 = sum(p) or 1.0
        p = [pi / s2 for pi in p]
        props[margin] = dict(zip(cats, p))
    return props

def make_diagnostics(df: pd.DataFrame, margins_props: Dict[str, Dict[str, float]], weight_before: str, weight_after: str) -> pd.DataFrame:
    rows = []
    for col, cats in margins_props.items():
        bsum = df.groupby(col, dropna=False)[weight_before].sum()
        asum = df.groupby(col, dropna=False)[weight_after].sum()
        tb, ta = bsum.sum(), asum.sum()
        for cat, p_t in cats.items():
            b, a = float(bsum.get(cat, 0)), float(asum.get(cat, 0))
            tpct, apct = p_t * 100, (a / ta) * 100 if ta > 0 else 0
            rows.append({
                "margin": col,
                "category": cat,
                "target_n": round(p_t * ta, 3),
                "actual_n": round(a, 3),
                "target_%": round(tpct, 2),
                "actual_%": round(apct, 2),
                "rel_error_%": round(abs(apct - tpct), 2),
                "before_total": round(b, 3),
                "after_total": round(a, 3),
            })
    return pd.DataFrame(rows)

# ---------------------------
# Input
# ---------------------------
up = st.file_uploader("Survey file (CSV or Excel)", type=["csv", "xlsx", "xls"])
if up:
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(up)
    else:
        df = pd.read_excel(up)
else:
    st.info("Using toy data. Upload to replace.")
    df = pd.DataFrame({
        "school_id": range(1, 21),
        "urban_rural": ["Urban"] * 8 + ["Rural"] * 12,
        "gender": (["Girls", "Boys"] * 10)[:20],
        "weight": 1.0,
    })

st.caption("Preview of input data")
show_df(df, head=30)

cats = infer_categoricals(df)
sel = st.multiselect("Categorical columns (margins to calibrate)", cats, default=[c for c in ["urban_rural", "gender"] if c in cats] or cats[:2])
wcol = st.selectbox("Weight column", ["<none>"] + list(df.columns), index=(1 if "weight" in df.columns else 0))

# Example targets (counts)
example = build_example_targets(df, sel)
st.write("**Targets table (CSV)** — columns: `margin,category,target` where **target is a NUMBER (count)** or a percentage (auto-detected).")
txt = st.text_area("Targets (CSV text)", example, height=180)

run = st.button("Run raking")

# ---------------------------
# Compute
# ---------------------------
if run:
    try:
        dfc = df.copy()
        if wcol == "<none>" or wcol not in dfc.columns:
            dfc["weight"] = 1.0
            wuse = "weight"
        else:
            wuse = wcol
            dfc[wuse] = pd.to_numeric(dfc[wuse], errors="coerce").fillna(0.0)

        margins_props = parse_targets_block(txt)

        # Run IPF
        w_before = dfc[wuse].to_numpy(float)
        w_new = rake_ipf(dfc, margins_props, weight_col=wuse)
        dfc["weight_before"] = w_before
        dfc["weight_calibrated"] = w_new
        dfc["weight_change"] = dfc["weight_calibrated"] - dfc["weight_before"]
        with np.errstate(divide="ignore", invalid="ignore"):
            dfc["weight_ratio"] = np.where(dfc["weight_before"] > 0, dfc["weight_calibrated"] / dfc["weight_before"], np.nan)

        diag = make_diagnostics(dfc, margins_props, "weight_before", "weight_calibrated")

        st.success("Calibration complete.")

        st.subheader("Diagnostics — numbers first (plus %)")
        show_df(diag)

        st.subheader("Calibrated data (explicit weight changes)")
        show_df(dfc.head(100))

        st.download_button("Download calibrated CSV", dfc.to_csv(index=False), "survey_calibrated.csv")
        st.download_button("Download diagnostics CSV", diag.to_csv(index=False), "calibration_diagnostics.csv")

        with st.expander("What to paste in the Targets box?"):
            st.write("Use a simple CSV with headers **margin,category,target** (numbers preferred).")
            st.write("**Example (CSV, counts):**")
            st.code(
                "margin,category,target\n"
                "urban_rural,Urban,800\n"
                "urban_rural,Rural,1200\n"
                "gender,Girls,1000\n"
                "gender,Boys,1000",
                language="csv"
            )

    except Exception as ex:
        st.error(f"Calibration error: {ex}")
