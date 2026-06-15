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

    st.subheader("AI Insights")

    rows, cols = df.shape
    missing = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=["number"]).columns

    insights = []

    insights.append(
        f"This dataset contains {rows} rows and {cols} columns."
    )

    if missing > 0:
        insights.append(
            f"The dataset contains {missing} missing values that may affect analysis quality."
        )
    else:
        insights.append(
            "No missing values were detected in the dataset."
        )

    if len(numeric_cols) > 0:
        insights.append(
            f"There are {len(numeric_cols)} numeric columns available for statistical analysis."
        )

        for col in numeric_cols:
            mean_value = df[col].mean()
            max_value = df[col].max()
            min_value = df[col].min()

            insights.append(
                f"{col}: average={mean_value:.2f}, minimum={min_value:.2f}, maximum={max_value:.2f}"
            )

    for insight in insights:
        st.write("• " + insight)

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
