import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup,compare_models,pull,save_model


with st.sidebar:
    st.image("deep-learning.png")
    st.title("Auto Machine learning")
    choice = st.radio("Navigation", ["Upload", "Profile", "ML", "Download"])
    st.info("This app allows you to build an autoomated machine learning pipline")


if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv",index_col=None)

if choice == "Upload":
    st.title("Upload your data for modeling")
    file = st.file_uploader("Upload Your dataset here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("source.csv",index=None)
        st.dataframe(df)

if choice == "Profile":
    st.title("Automated Data analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine learning trainer")
    target = st.selectbox("select your target", df.columns)
    setup(df,target=target)
    setup_df = pull()
    st.info("this is setup")
    st.dataframe(setup_df)
    best_modal = compare_models()
    compare_df = pull()
    st.info("this is Modal")
    st.dataframe(compare_df)
    best_modal
    save_model(best_modal,"best_modal")


if choice == "Download":
    with open("best_modal.pkl",'rb') as f:
        st.download_button("Download the modal",f,"trained_modal.pkl")
