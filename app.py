import streamlit as st
import pandas as pd
from causalimpact import CausalImpact
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 Causal Impact Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        sheet = st.selectbox("Select the sheet to use", sheet_names)
        raw_df = pd.read_excel(uploaded_file, sheet_name=sheet)

    # Let user pick the date column
    st.subheader("Select the Date Column")
    date_col = st.selectbox("Which column contains the date?", raw_df.columns.tolist())

    # Drop unnamed or empty columns
    df = raw_df.drop(columns=[col for col in raw_df.columns if "Unnamed" in col], errors="ignore")
    df = df.dropna(subset=[date_col])

    # Convert and set date index
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df.set_index(date_col, inplace=True)
    df = df.sort_index()

    st.subheader("Preview of Cleaned Data")
    st.dataframe(df.head())

    target_metric = st.selectbox("Select target metric to evaluate", df.columns.tolist())
    control_metrics = st.multiselect("Optional: Add control variables", [col for col in df.columns if col != target_metric])

    all_dates = df.index
    min_date = all_dates.min()
    max_date = all_dates.max()

    st.markdown("### Define Analysis Periods")
    intervention_date = st.date_input("Intervention Date (e.g. campaign start)", min_value=min_date, max_value=max_date)
    pre_start_date = st.date_input("Pre-period start date", value=min_date)
    pre_end_date = st.date_input("Pre-period end date", value=intervention_date - pd.Timedelta(days=1))
    post_start_date = st.date_input("Post-period start date", value=intervention_date)
    post_end_date = st.date_input("Post-period end date", value=max_date)

    # Convert all selected dates to pandas Timestamps for accurate comparison
    pre_start_date = pd.Timestamp(pre_start_date)
    pre_end_date = pd.Timestamp(pre_end_date)
    post_start_date = pd.Timestamp(post_start_date)
    post_end_date = pd.Timestamp(post_end_date)
    intervention_date = pd.Timestamp(intervention_date)

    fit_method = st.sidebar.selectbox(
        "Select model fitting method",
        options=["vi", "hmc"],
        index=0,
        help="VI is faster; HMC is more accurate but slower."
    )

    if st.button("Run Causal Impact Analysis"):
        try:
            columns_to_use = [target_metric] + control_metrics if control_metrics else [target_metric]
            model_data = df[columns_to_use]

            # Check for any missing dates before dropping NaNs
            required_dates = [pre_start_date, pre_end_date, post_start_date, post_end_date]
            missing = [date for date in required_dates if date not in model_data.index]
            if missing:
                st.error(f"The following selected dates are missing in the data: {missing}")
            else:
                model_data = model_data.dropna()
                missing_after_drop = [date for date in required_dates if date not in model_data.index]
                if missing_after_drop:
                    st.error(f"The following dates were removed due to missing values: {missing_after_drop}")
                else:
                    pre_period = [pre_start_date, pre_end_date]
                    post_period = [post_start_date, post_end_date]

                    impact = CausalImpact(
                        model_data,
                        pre_period,
                        post_period,
                        model_args={"fit_method": fit_method}
                    )

                    st.subheader("📊 Summary")
                    st.text(impact.summary())

                    st.subheader("📝 Detailed Report")
                    st.text(impact.summary(output='report'))

                    st.subheader("📉 Impact Plot")
                    fig = impact.plot(figsize=(15, 6))
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("Plot could not be rendered.")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
