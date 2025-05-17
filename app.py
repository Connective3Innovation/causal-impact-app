import streamlit as st
import pandas as pd
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Causal Impact Analysis Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        # Read and clean the file
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file, skiprows=6)
        df = df_raw[['Date', 'Sessions', 'Transactions', 'Purchase revenue']].dropna()

        # Convert date and sort
        df['Date'] = pd.to_datetime(df['Date'].astype(float).astype(int).astype(str), format='%Y%m%d', errors='coerce')
        df.set_index('Date', inplace=True)
        df = df.sort_index()

        st.subheader("Preview of Uploaded Data")
        st.write(df.head())

        # Column selection
        response_col = st.selectbox("Select the response (dependent) variable", df.columns, index=2)
        control_cols = st.multiselect("Select control (independent) variables", df.columns.difference([response_col]), default=['Sessions'])

        # Intervention date selection
        intervention_date = st.date_input("Intervention Date", df.index[int(len(df) * 0.7)].date())

        # Pre and post periods around intervention
        pre_start_date = st.date_input("Pre-period start", df.index.min().date())
        pre_end_date = st.date_input("Pre-period end", (pd.Timestamp(intervention_date) - pd.Timedelta(days=1)).date())
        post_start_date = st.date_input("Post-period start", intervention_date)
        post_end_date = st.date_input("Post-period end", df.index.max().date())

        if st.button("Run Causal Impact Analysis"):
            # Subset the data
            data = df[[response_col] + control_cols]

            # Use timestamps directly
            pre_period = [pd.to_datetime(pre_start_date), pd.to_datetime(pre_end_date)]
            post_period = [pd.to_datetime(post_start_date), pd.to_datetime(post_end_date)]

            # Run the model
            ci = CausalImpact(data, pre_period, post_period)

            # Output results
            st.subheader("Summary")
            st.text(ci.summary())
            st.subheader("Detailed Report")
            st.text(ci.summary(output='report'))

            st.subheader("Impact Plot")
            
            plt.figure(figsize=(15, 6))
            ci.plot()
            plt.xticks(rotation=90)
            fig = plt.gcf()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
