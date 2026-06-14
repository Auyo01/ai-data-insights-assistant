import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("AI Data Insights Assistant")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Column Names")
    st.write(list(df.columns))

    st.subheader("Dataset Information")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(df.describe())
    st.subheader("Automated Insights")

rows, cols = df.shape

st.write(f"The dataset contains {rows} rows and {cols} columns.")

missing = df.isnull().sum().sum()

st.write(f"There are {missing} missing values in the dataset.")

numeric_cols = df.select_dtypes(include=['number']).columns

if len(numeric_cols) > 0:
    st.write(
        f"The dataset contains {len(numeric_cols)} numeric columns available for analysis."
    )
  

    numeric_columns = df.select_dtypes(include=['number']).columns

    if len(numeric_columns) > 0:
        st.subheader("Data Visualization")

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_columns
        )

        fig, ax = plt.subplots()
        df[selected_column].hist(ax=ax)
        st.pyplot(fig)
