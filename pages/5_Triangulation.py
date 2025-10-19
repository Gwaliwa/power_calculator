
import pandas as pd, numpy as np, streamlit as st
from utils import composite_ivw, regress_calibrate

st.title('5) Triangulation — combine survey & admin outcomes')

c1, c2 = st.columns(2)
with c1:
    theta_s = st.number_input('Survey estimate', 0.0, 1.0, 0.86, 0.01)
    se_s = st.number_input('Survey SE', 0.000, 1.0, 0.020, 0.001)
with c2:
    theta_a = st.number_input('Admin estimate', 0.0, 1.0, 0.83, 0.01)
    se_a = st.number_input('Admin SE', 0.000, 1.0, 0.010, 0.001)

if st.button('Inverse-variance blend'):
    t, se = composite_ivw(theta_s, se_s, theta_a, se_a)
    st.success(f'Triangulated estimate ≈ {t:.3f}  (SE ≈ {se:.3f})')
    st.json({'inputs':{'theta_s':theta_s,'se_s':se_s,'theta_a':theta_a,'se_a':se_a},
             'output':{'theta':round(t,4),'se':round(se,4)}})

st.markdown('---')
st.caption('Regression calibration from overlap (teach admin to mimic survey). Upload CSV with columns "admin" and "survey".')
up = st.file_uploader('Overlap CSV', type=['csv'], key='ov')
if up:
    ov = pd.read_csv(up).dropna()
    if set(['admin','survey']).issubset(ov.columns):
        b0, b1 = regress_calibrate(ov['admin'].values, ov['survey'].values)
        st.success(f'Estimated calibration: survey ≈ {b0:.3f} + {b1:.3f} × admin')
        ov['admin_calibrated'] = b0 + b1 * ov['admin'].values
        st.dataframe(ov.head(), use_container_width=True)
        st.download_button('Download calibrated overlap', ov.to_csv(index=False), file_name='overlap_calibrated.csv')
    else:
        st.error('CSV must contain "admin" and "survey" columns.')
