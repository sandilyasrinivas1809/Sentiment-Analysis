import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file,encoding = 'unicode_escape')
st.dataframe(df1.columns)
#select = st.selectbox()

