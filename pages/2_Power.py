
import pandas as pd, streamlit as st, math
from utils import clusters_per_arm_power

st.title('2) Power / MDE — two-arm cluster RCT')
c1, c2, c3 = st.columns(3)
with c1:
    p0 = st.number_input('Baseline p0', 0.0, 1.0, 0.85, 0.01)
    delta = st.number_input('Target ITT effect Δ (abs.)', 0.001, 0.5, 0.05, 0.001, format='%.3f')
    alpha = st.selectbox('Alpha', [0.10,0.05,0.01], index=1)
    one_sided = st.checkbox('One-sided test', value=False)
    power = st.selectbox('Power (1-β)', [0.80,0.85,0.90], index=0)
with c2:
    m = st.number_input('Cluster size m', 2, 1000, 20)
    icc = st.number_input('ICC', 0.0, 1.0, 0.05, 0.01)
    cov_m = st.number_input('Cluster size CoV', 0.0, 2.0, 0.0, 0.05)
    r2 = st.number_input('Baseline R² (ANCOVA)', 0.0, 0.95, 0.0, 0.05)
with c3:
    attr_t = st.number_input('Attrition in T', 0.0, 0.9, 0.0, 0.05)
    attr_c = st.number_input('Attrition in C', 0.0, 0.9, 0.0, 0.05)
    take_t = st.number_input('Take-up in T', 0.0, 1.0, 1.0, 0.05)
    take_c = st.number_input('Take-up in C', 0.0, 1.0, 0.0, 0.05)

cl_arm, n_indiv_arm, deff = clusters_per_arm_power(p0,delta,alpha,power,m,icc,cov_m,r2,one_sided,attr_t,attr_c,take_t,take_c)
st.success(f'Clusters per arm ≈ {cl_arm}; individuals per arm ≈ {n_indiv_arm}; DEFF ≈ {deff:.2f}')

st.markdown('---')
st.markdown('**Stratified randomization targets using admin shares**')
txt = st.text_area("'Stratum,Count' per line", 'Urban,4000\nRural,6000', key='alloc2')
try:
    parts = [ln.split(',') for ln in txt.splitlines() if ln.strip()]
    rows=[]
    tot = sum(float(p[1]) for p in parts) or 1.0
    for name,count in parts:
        share=float(count)/tot
        rows.append((name.strip(), int(float(count)), round(100*share,2), math.floor(share*cl_arm), math.floor(share*cl_arm)))
    def fix(idx, target):
        miss = target - sum(r[idx] for r in rows)
        i=0
        while miss>0 and rows:
            r=list(rows[i%len(rows)]); r[idx]+=1; rows[i%len(rows)]=tuple(r); miss-=1; i+=1
    fix(3, cl_arm); fix(4, cl_arm)
    df = pd.DataFrame(rows, columns=['Stratum','AdminCount','Share%','Clusters_T','Clusters_C'])
    st.dataframe(df, use_container_width=True)
    st.download_button('Download stratified targets', df.to_csv(index=False), file_name='stratified_targets.csv')
except Exception as ex:
    st.warning(f'Parse error: {ex}')
