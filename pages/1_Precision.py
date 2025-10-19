
import math, pandas as pd, streamlit as st
from utils import n_precision

st.title('1) Precision — cluster-adjusted sample size')
col1, col2, col3 = st.columns(3)
with col1:
    p = st.number_input('Expected proportion p', 0.0, 1.0, 0.85, 0.01)
    e = st.number_input('Margin ±e', 0.001, 0.2, 0.02, 0.001, format='%.3f')
    conf = st.selectbox('Confidence', [0.80,0.85,0.90,0.95,0.975,0.99,0.995], index=3)
with col2:
    m = st.number_input('Cluster size m', 2, 1000, 20)
    icc = st.number_input('ICC', 0.0, 1.0, 0.05, 0.01)
    cov_m = st.number_input('Cluster size CoV', 0.0, 2.0, 0.0, 0.05)
with col3:
    N = st.number_input('Population size N (FPC optional)', 0, 10000000, 10000)
    N = int(N) if N>0 else None

clusters, n_indiv, deff = n_precision(p,e,conf,m,icc,cov_m,N)
st.success(f'Clusters ≈ {clusters}; individuals ≈ {n_indiv}; DEFF ≈ {deff:.2f}')

st.markdown('---')
st.markdown('**Admin shares -> proportional allocation**')
txt = st.text_area('Paste "Stratum,Count" per line', 'Urban,4000\nRural,6000')
try:
    parts = [ln.split(',') for ln in txt.splitlines() if ln.strip()]
    rows = []
    total = sum(float(p[1]) for p in parts) or 1.0
    for name,count in parts:
        share = float(count)/total
        rows.append((name.strip(), int(float(count)), round(100*share,2), math.floor(share*clusters)))
    miss = clusters - sum(r[3] for r in rows)
    i=0
    while miss>0 and rows:
        r=list(rows[i%len(rows)]); r[3]+=1; rows[i%len(rows)]=tuple(r); miss-=1; i+=1
    df = pd.DataFrame(rows, columns=['Stratum','AdminCount','Share%','Clusters'])
    st.dataframe(df, use_container_width=True)
    st.download_button('Download allocation CSV', df.to_csv(index=False), file_name='allocation_proportional.csv')
except Exception as ex:
    st.warning(f'Parse error: {ex}')
