import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq

st.set_page_config(
    page_title="AI Data Insights Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Insights Assistant")

st.sidebar.title("Navigation")
st.sidebar.info(
    "Upload a CSV dataset to generate insights, visualizations, and reports."
)

st.sidebar.markdown("### Features")
st.sidebar.markdown("""
- Dataset Preview
- Statistical Summary
- AI Analysis (Llama 3)
- Correlation Analysis
- Data Visualization
- Download Reports
""")

st.info(
    "Upload a CSV file to explore your data, generate insights, and visualize trends."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

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

    numeric_cols = df.select_dtypes(include=["number"]).columns

    st.subheader("🤖 AI Analysis (Llama 3)")

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        dataset_summary = f"""
        Dataset Shape: {df.shape}

        Columns:
        {list(df.columns)}

        Missing Values:
        {df.isnull().sum().to_dict()}

        Statistical Summary:
        {df.describe().to_string()}
        """

        prompt = f"""
        You are an expert data analyst.

        Analyze the dataset and provide:

        1. Key observations
        2. Data quality issues
        3. Important trends
        4. Actionable recommendations

        Dataset Information:

        {dataset_summary}
        """

        with st.spinner("Generating AI insights..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            ai_response = response.choices[0].message.content

        st.write(ai_response)

    except Exception as e:
        st.error(f"AI analysis unavailable: {e}")

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
