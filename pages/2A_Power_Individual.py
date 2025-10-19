
import math, streamlit as st

st.title('2A) Power/MDE — individual (two-arm, proportions)')

st.markdown('Two-sample proportions (independent): n_per_arm(SRS) ≈ [ (z_{1-α/2} * sqrt(2 p̄ (1-p̄)) + z_{power} * sqrt(p0(1-p0) + p1(1-p1)) )^2 ] / Δ^2. Supports unequal allocation ratio r = n_T / n_C.')

# Z table
Z = {0.80:1.2815515655446004,0.85:1.4395314709384563,0.90:1.6448536269514722,0.95:1.959963984540054,0.975:2.241402727604947,0.99:2.5758293035489004}

c1, c2, c3 = st.columns(3)
with c1:
    p0 = st.number_input('Baseline p0', 0.0, 1.0, 0.85, 0.01)
    delta = st.number_input('Target ITT effect Δ (abs.)', 0.001, 0.5, 0.05, 0.001, format='%.3f')
    alpha = st.selectbox('Alpha', [0.10,0.05,0.01], index=1)
    one_sided = st.checkbox('One-sided test', value=False)
with c2:
    power = st.selectbox('Power (1-β)', [0.80,0.85,0.90], index=0)
    r = st.number_input('Allocation ratio r = n_T / n_C', 0.1, 10.0, 1.0, 0.1)
    r2 = st.number_input('Baseline R^2 (ANCOVA gain)', 0.0, 0.95, 0.0, 0.05)
with c3:
    attr_t = st.number_input('Attrition in T', 0.0, 0.9, 0.0, 0.05)
    attr_c = st.number_input('Attrition in C', 0.0, 0.9, 0.0, 0.05)
    take_t = st.number_input('Take-up in T', 0.0, 1.0, 1.0, 0.05)
    take_c = st.number_input('Take-up in C', 0.0, 1.0, 0.0, 0.05)

eff_delta = delta * (take_t - take_c)
if eff_delta <= 0: eff_delta = 1e-9

z_alpha = Z[1 - (alpha if one_sided else alpha/2)]
z_power = Z[power]
p1 = max(0.0, min(1.0, p0 + eff_delta))
pbar = (p0 + p1)/2.0

# Unequal allocation adjustment (approximate): inflate by (1 + 1/r)^2 / 4 when p0≈p1; a more direct way is to compute per-arm using variance with differing n.
term1 = z_alpha * math.sqrt(2 * pbar * (1 - pbar))
term2 = z_power * math.sqrt(p0*(1-p0) + p1*(1-p1))
n_equal = ((term1 + term2)**2) / (eff_delta**2)

# Unequal allocation inflation
infl = (1 + 1/r)**2 / 4.0
n_total = n_equal * (1 - r2) * infl

n_c = math.ceil(n_total / (1 + r))
n_t = math.ceil(r * n_c)

# Attrition inflation
keep_t = max(1e-6, 1 - attr_t); keep_c = max(1e-6, 1 - attr_c)
n_t_adj = math.ceil(n_t / keep_t)
n_c_adj = math.ceil(n_c / keep_c)

st.success(f'Individuals needed: Treatment ≈ {n_t_adj}, Control ≈ {n_c_adj} (after attrition & ANCOVA adjustments). Allocation ratio r={r}.')
