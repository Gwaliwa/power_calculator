# neyman_allocation_app.py
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Neyman Allocation", layout="wide")
st.title("3) Allocation — Neyman (when strata variances differ)")

st.markdown("""
**What this does:** Distributes a fixed total across strata to maximize precision.
You can provide either **baseline p** per stratum (the app computes SD = sqrt(p(1-p))) or provide **SD** directly.
Then we **split each stratum into Treatment/Control** using an allocation ratio r.
""")

with st.sidebar:
    st.header("Global settings")
    total = st.number_input("Total to allocate (clusters or individuals)", min_value=1, max_value=1000000, value=120, step=1)
    r = st.number_input("Allocation ratio r = n_T / n_C", min_value=0.01, max_value=100.0, value=1.0, step=0.01, format="%.2f")
    unit = st.selectbox("Units", ["Clusters", "Individuals"])

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Stratum,Count  (N_h)")
    txtN = st.text_area("Provide one per line", "Urban,4000\nRural,6000", height=150)

with col2:
    st.subheader("EITHER: Stratum,p  (baseline proportion)")
    txtP = st.text_area("Optional — if provided, app computes SD = sqrt(p(1-p))",
                        "Urban,0.80\nRural,0.50", height=150)

with col3:
    st.subheader("OR: Stratum,SD  (use directly)")
    txtS = st.text_area("Optional — used only if p is not provided for that stratum",
                        "Urban,0.40\nRural,0.50", height=150)

def parse_kv_lines(txt):
    rows = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    kv = {}
    for ln in rows:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Line must be 'Name,Value': {ln}")
        name, val = parts[0], parts[1]
        try:
            kv[name] = float(val)
        except ValueError:
            raise ValueError(f"Value must be numeric in line: {ln}")
    return kv

def largest_remainder_round(values, total_target):
    """Round a vector of nonnegative floats to integers so they sum to total_target."""
    base = np.floor(values).astype(int)
    remainder = values - base
    short = int(total_target - base.sum())
    if short < 0:
        # if we somehow overshot, remove from largest bases (rare for this pipeline)
        idx = np.argsort(base)[::-1]
        for i in range(-short):
            base[idx[i % len(base)]] -= 1
        return base
    order = np.argsort(-remainder)
    for i in range(short):
        base[order[i % len(base)]] += 1
    return base

def compute_sd_from_p(p):
    return np.sqrt(p * (1 - p))

def neyman_allocation(total, counts, sds):
    """Return integer allocations that sum to total using weights counts * sds."""
    weights = counts * sds
    if np.any(weights < 0):
        raise ValueError("Counts and SDs must be nonnegative")
    if weights.sum() == 0:
        # degenerate: split equally
        alloc_float = np.ones_like(weights) * (total / len(weights))
    else:
        alloc_float = total * (weights / weights.sum())
    return largest_remainder_round(alloc_float, total)

try:
    # Parse inputs
    counts_kv = parse_kv_lines(txtN)  # N_h
    names = list(counts_kv.keys())
    counts = np.array([counts_kv[n] for n in names], dtype=float)

    # Try p first; if missing for some stratum, fallback to SD line
    p_kv = parse_kv_lines(txtP) if txtP.strip() else {}
    s_kv = parse_kv_lines(txtS) if txtS.strip() else {}

    sds = []
    missing = []
    for n in names:
        if n in p_kv:
            p = p_kv[n]
            if not (0 <= p <= 1):
                raise ValueError(f"Baseline p must be in [0,1] for {n}")
            sds.append(compute_sd_from_p(p))
        elif n in s_kv:
            sds.append(s_kv[n])
        else:
            missing.append(n)

    if missing:
        raise ValueError("Missing p or SD for strata: " + ", ".join(missing))

    sds = np.array(sds, dtype=float)

    # Main Neyman allocation for total (clusters or individuals)
    alloc = neyman_allocation(int(total), counts, sds)

    # Split each stratum into Treatment/Control with ratio r
    t_share = r / (1 + r)
    c_share = 1 / (1 + r)
    alloc_T_float = alloc * t_share
    alloc_C_float = alloc * c_share
    # Round within each stratum to keep n_h fixed
    alloc_T = np.floor(alloc_T_float).astype(int)
    alloc_C = alloc - alloc_T  # ensures per-stratum sums match n_h
    # Fix any off-by-one due to rounding by moving one unit from C to T where remainder was largest
    remainder_T = alloc_T_float - alloc_T
    diff = int(alloc_T.sum() - np.round(alloc_T_float).sum())
    if diff < 0:
        # need to add |diff| to T (take from C) at largest remainders
        order = np.argsort(-remainder_T)
        for i in range(-diff):
            idx = order[i % len(order)]
            if alloc_C[idx] > 0:
                alloc_T[idx] += 1
                alloc_C[idx] -= 1

    df = pd.DataFrame({
        "Stratum": names,
        "AdminCount (N_h)": counts.astype(int),
        "SD (S_h)": np.round(sds, 4),
        f"{unit} total (n_h)": alloc.astype(int),
        f"{unit} — Treatment (T)": alloc_T.astype(int),
        f"{unit} — Control (C)": alloc_C.astype(int),
    })

    st.success("Neyman allocation computed.")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        f"Download Neyman allocation ({unit}, with T/C)",
        df.to_csv(index=False),
        file_name=f"allocation_neyman_{unit.lower()}_with_TC.csv",
        mime="text/csv"
    )

    # Show shares for transparency
    shares = (counts * sds) / (counts * sds).sum() if (counts * sds).sum() > 0 else np.ones_like(counts)/len(counts)
    st.caption("Shares (weights normalized): " + ", ".join([f"{n}: {s:.3f}" for n, s in zip(names, shares)]))

except Exception as ex:
    st.warning(f"Neyman allocation error: {ex}")
