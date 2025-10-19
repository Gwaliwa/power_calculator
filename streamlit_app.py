
import streamlit as st
from pathlib import Path

st.set_page_config(page_title='UNICEF Triangulator', layout='wide')
st.title('UNICEF Triangulator — Sampling, Power & Triangulation')
st.caption('Admin -> Survey -> Triangulation • Design with admin data, power your RCTs, calibrate to margins, and combine outcomes.')

logo_path = Path('assets/unicef_logo.png')
if logo_path.exists():
    st.image(str(logo_path), width=180)

st.markdown('''
**Whats inside**
1. **Precision**: sample size for a proportion (cluster-adjusted; optional FPC).
2. **Power/MDE**: two-arm cluster RCT with ANCOVA, attrition, noncompliance, unequal cluster sizes.
3. **Allocation**: proportional & Neyman across strata using administrative counts and SDs.
4. **Calibration**: raking survey weights to admin margins, with diagnostics and exports.
5. **Triangulation**: inverse-variance blend and regression calibration from overlap.

Tip: Use Admin data to build the frame & shares -> collect Survey data -> calibrate to Admin -> triangulate outcomes.
''')
