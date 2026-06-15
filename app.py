import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("AI Data Insights Assistant")

st.write(
    "Upload a CSV file to generate summaries, visualizations, and insights."
)

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

    summary_csv = df.describe().to_csv()

    st.download_button(
        label="📥 Download Summary Report",
        data=summary_csv,
        file_name="summary_report.csv",
        mime="text/csv"
    )

    st.subheader("Automated Insights")

    rows, cols = df.shape
    st.write(f"The dataset contains {rows} rows and {cols} columns.")

    missing = df.isnull().sum().sum()
    st.write(f"There are {missing} missing values in the dataset.")

    numeric_cols = df.select_dtypes(include=["number"]).columns

    if len(numeric_cols) > 0:
        st.write(
            f"The dataset contains {len(numeric_cols)} numeric columns available for analysis."
        )

    if len(numeric_cols) > 1:
        st.subheader("Correlation Analysis")

        correlation_matrix = df[numeric_cols].corr()
        st.dataframe(correlation_matrix)

    if len(numeric_cols) > 0:
        st.subheader("Data Visualization")

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_cols
        )

        chart_type = st.selectbox(
            "Select chart type",
            ["Histogram", "Line Chart", "Bar Chart"]
        )

        fig, ax = plt.subplots()

        if chart_type == "Histogram":
            df[selected_column].hist(ax=ax)

        elif chart_type == "Line Chart":
            ax.plot(df[selected_column])

        elif chart_type == "Bar Chart":
            ax.bar(
                range(len(df[selected_column])),
                df[selected_column]
            )

        st.pyplot(fig)
