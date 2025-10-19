
import math
import numpy as np
import pandas as pd

Z = {
    0.80: 1.2815515655446004,
    0.85: 1.4395314709384563,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.975: 2.241402727604947,
    0.99: 2.5758293035489004,
    0.995: 2.807033768343811,
}

def deff_cluster(m: float, icc: float, cov_m: float = 0.0) -> float:
    return 1.0 + ((1.0 + cov_m**2) * (m - 1.0)) * icc

def n_precision(p: float, e: float, conf: float, m: float, icc: float, cov_m: float = 0.0, N: int | None = None):
    z = Z.get(conf, Z[0.95])
    n_srs = (z**2) * p * (1-p) / (e**2)
    if N is not None and N > 0:
        n_srs = n_srs / (1 + (n_srs - 1) / N)
    deff = deff_cluster(m, icc, cov_m)
    n_indiv = math.ceil(n_srs * deff)
    clusters = math.ceil(n_indiv / m)
    return clusters, n_indiv, deff

def clusters_per_arm_power(p0: float, delta: float, alpha: float, power: float, m: float, icc: float, cov_m: float=0.0,
                           r2_baseline: float=0.0, one_sided: bool=False,
                           attrit_t: float=0.0, attrit_c: float=0.0,
                           takeup_t: float=1.0, takeup_c: float=0.0):
    eff_delta = delta * (takeup_t - takeup_c)
    if eff_delta <= 0:
        eff_delta = 1e-9
    z_alpha = Z.get(1 - (alpha if one_sided else alpha/2), Z[0.95])
    z_power = Z.get(power, Z[0.80])
    p1 = min(1.0, max(0.0, p0 + eff_delta))
    pbar = (p0 + p1) / 2
    term1 = z_alpha * math.sqrt(2 * pbar * (1 - pbar))
    term2 = z_power * math.sqrt(p0*(1-p0) + p1*(1-p1))
    n_srs = ((term1 + term2)**2) / (eff_delta**2)
    n_srs = n_srs * (1.0 - r2_baseline)
    keep_t = max(1e-6, 1 - attrit_t); keep_c = max(1e-6, 1 - attrit_c)
    n_srs = n_srs / min(keep_t, keep_c)
    deff = deff_cluster(m, icc, cov_m)
    n_indiv_arm = math.ceil(n_srs * deff)
    clusters_arm = math.ceil(n_indiv_arm / m)
    return clusters_arm, n_indiv_arm, deff

def neyman_allocation(total_clusters: int, N_h, S_h):
    import numpy as np
    N_h = np.array(N_h, dtype=float); S_h = np.array(S_h, dtype=float)
    w = N_h * S_h
    if w.sum() == 0: 
        return np.full_like(N_h, total_clusters // len(N_h), dtype=int)
    raw = total_clusters * w / w.sum()
    flo = np.floor(raw).astype(int)
    rem = total_clusters - flo.sum()
    order = np.argsort(-(raw - flo))
    for i in range(rem):
        flo[order[i % len(flo)]] += 1
    return flo

def rake_weights(df: pd.DataFrame, targets: dict, weight_col: str="weight", max_iter: int=100, tol: float=1e-6) -> pd.Series:
    w = df.get(weight_col, pd.Series(1.0, index=df.index)).astype(float).copy()
    for _ in range(max_iter):
        maxdiff = 0.0
        for col, tar in targets.items():
            cats = df[col].astype(str)
            cur = df.groupby(cats).apply(lambda g: w.loc[g.index].sum())
            tarS = pd.Series(tar, dtype=float)
            factors = cats.map(lambda x: tarS.get(x, float("nan")))
            denom = cats.map(lambda x: cur.get(x, 0.0))
            f = np.where((~np.isnan(factors)) & (denom > 0), factors / denom, 1.0)
            maxdiff = max(maxdiff, float(np.max(np.abs(f - 1.0))))
            w = w * f
        if maxdiff < tol:
            break
    return w

def composite_ivw(theta_s: float, se_s: float, theta_a: float, se_a: float):
    v_s, v_a = se_s**2, se_a**2
    if v_s <= 0 and v_a <= 0:
        return float("nan"), float("nan")
    if v_s <= 0:
        return theta_a, se_a
    if v_a <= 0:
        return theta_s, se_s
    alpha = v_a / (v_s + v_a)
    theta = alpha * theta_s + (1 - alpha) * theta_a
    se = math.sqrt((alpha**2) * v_s + ((1 - alpha)**2) * v_a)
    return theta, se

def regress_calibrate(admin, survey):
    import numpy as np
    X = np.vstack([np.ones_like(admin), admin]).T
    beta, *_ = np.linalg.lstsq(X, survey, rcond=None)
    return float(beta[0]), float(beta[1])
