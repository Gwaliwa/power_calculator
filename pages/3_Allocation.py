
import numpy as np, pandas as pd, streamlit as st
from utils import neyman_allocation

st.title('3) Allocation — Neyman (when strata variances differ)')
total = st.number_input('Total clusters to allocate', 1, 10000, 120)
c1, c2 = st.columns(2)
with c1:
    txtN = st.text_area('Stratum,Count', 'Urban,4000\nRural,6000')
with c2:
    txtS = st.text_area('Stratum,SD (same order)', 'Urban,0.10\nRural,0.20')

try:
    a = [ln.split(',') for ln in txtN.splitlines() if ln.strip()]
    b = [ln.split(',') for ln in txtS.splitlines() if ln.strip()]
    namesN = [x[0].strip() for x in a]; counts = np.array([float(x[1]) for x in a])
    namesS = [x[0].strip() for x in b]; sds = np.array([float(x[1]) for x in b])
    assert namesN == namesS, 'Stratum names/order must match'
    alloc = neyman_allocation(int(total), counts, sds)
    df = pd.DataFrame({'Stratum':namesN, 'AdminCount':counts.astype(int), 'SD':sds, 'Clusters':alloc})
    st.dataframe(df, use_container_width=True)
    st.download_button('Download Neyman allocation', df.to_csv(index=False), file_name='allocation_neyman.csv')
except Exception as ex:
    st.warning(f'Neyman allocation error: {ex}')
