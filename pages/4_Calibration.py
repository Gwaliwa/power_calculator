
import json, numpy as np, pandas as pd, streamlit as st
from utils import rake_weights

st.title('4) Calibration — raking to admin margins')
up = st.file_uploader('Survey CSV (include categorical cols and optional "weight")', type=['csv'])
if up:
    df = pd.read_csv(up)
else:
    st.info('Using toy data. Upload to replace.')
    df = pd.DataFrame({
        'school_id': range(1,21),
        'urban_rural': ['Urban']*8 + ['Rural']*12,
        'gender': (['Girls','Boys']*10)[:20],
        'attend': [1,1,0,1,1,0,1,1, 1,0,1,1, 1,0,1,1, 0,1,1,1],
        'weight': 1.0,
    })
st.dataframe(df.head(), use_container_width=True)

cats = [c for c in df.columns if df[c].dtype==object or str(df[c].dtype).startswith('category')]
sel = st.multiselect('Categorical columns', options=cats, default=[c for c in ['urban_rural','gender'] if c in cats])
wcol = st.selectbox('Weight column', options=['<none>']+list(df.columns), index=(list(df.columns).index('weight')+1 if 'weight' in df.columns else 0))

example = {c: {str(k):100 for k in sorted(df[c].astype(str).unique())} for c in sel}
txt = st.text_area('Admin targets JSON', json.dumps(example, indent=2))

if st.button('Run raking'):
    try:
        targets = json.loads(txt)
        dfc = df.copy()
        if wcol == '<none>' or wcol not in dfc.columns:
            dfc['weight'] = 1.0; wuse='weight'
        else:
            wuse = wcol
        w_new = rake_weights(dfc, targets, weight_col=wuse)
        dfc['weight_calibrated'] = w_new
        st.success('Calibration complete.')
        rows=[]; 
        for col, tar in targets.items():
            before = dfc.groupby(col)[wuse].sum()
            after = dfc.groupby(col)['weight_calibrated'].sum()
            for k,V in tar.items():
                rows.append({'Margin':col,'Category':k,'AdminTarget':V,'SurveyBefore':float(before.get(k,0.0)),'SurveyAfter':float(after.get(k,0.0))})
        diag = pd.DataFrame(rows)
        st.dataframe(diag, use_container_width=True)
        st.download_button('Download calibrated CSV', dfc.to_csv(index=False), file_name='survey_calibrated.csv')
        st.download_button('Download diagnostics', diag.to_csv(index=False), file_name='calibration_diagnostics.csv')
    except Exception as ex:
        st.error(f'Calibration error: {ex}')
