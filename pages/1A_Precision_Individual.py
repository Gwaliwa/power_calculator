
import math, streamlit as st

st.title('1A) Precision — individual (SRS) sample size for a proportion')

st.markdown('SRS formula: n = z^2 * p(1-p) / e^2; optional finite population correction (FPC).')

conf = st.selectbox('Confidence', [0.80,0.85,0.90,0.95,0.975,0.99,0.995], index=3)
Z = {0.80:1.2815515655446004,0.85:1.4395314709384563,0.90:1.6448536269514722,0.95:1.959963984540054,0.975:2.241402727604947,0.99:2.5758293035489004,0.995:2.807033768343811}
p = st.number_input('Expected proportion p', 0.0, 1.0, 0.85, 0.01)
e = st.number_input('Margin ±e', 0.001, 0.2, 0.02, 0.001, format='%.3f')
N = st.number_input('Population size N (optional FPC)', 0, 100000000, 0)

z = Z[conf]
n_srs = (z**2) * p * (1-p) / (e**2)
n_fpc = n_srs / (1 + (n_srs - 1)/N) if N>0 else n_srs

st.success(f'Individuals needed (no clustering): n ≈ {math.ceil(n_fpc)}')
