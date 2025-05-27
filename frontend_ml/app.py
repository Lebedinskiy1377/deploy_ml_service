import streamlit as st
import pandas as pd
import requests
import tempfile
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

FASTAPI_INVOCATION_ENDPOINT = "http://fastapi:8005/invocation"
FASTAPI_OPTIMIZE_ENDPOINT = "http://fastapi:8005/optimize_price"

st.title("Price Prediction and Optimization Web Application")
st.write("Upload a CSV file containing SKU and related information to get predictions or optimize prices.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data)

    action = st.radio("Select action:", ["Predict Demand", "Optimize Price"])

    if st.button("Submit"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name

            try:
                if action == "Predict Demand":
                    with open(tmp_file_path, "rb") as file:
                        response = requests.post(
                            FASTAPI_INVOCATION_ENDPOINT,
                            files={"file": (uploaded_file.name, file, "text/csv")}
                        )

                    if response.status_code == 200:
                        result = response.json()
                        result_df = pd.DataFrame(result)

                        st.write("Predictions with Original Data:")
                        st.dataframe(result_df)

                        csv = result_df.to_csv(index=False).encode()
                        st.download_button(
                            label="Download Predicted Data as CSV",
                            data=csv,
                            file_name="predicted_demand.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"Error in prediction: {response.text}")

                elif action == "Optimize Price":
                    with open(tmp_file_path, "rb") as file:
                        response = requests.post(
                            FASTAPI_OPTIMIZE_ENDPOINT,
                            files={"file": (uploaded_file.name, file, "text/csv")}
                        )

                    if response.status_code == 200:
                        result = response.json()
                        results_df = pd.DataFrame(result)

                        results_df['base_gmv'] = results_df['price_per_sku'] * results_df['base_demand']
                        results_df['base_margin'] = (results_df['price_per_sku'] - results_df['cost']) / results_df['price_per_sku']

                        results_df['gmv_increase_%'] = ((results_df['gmv'] - results_df['base_gmv']) / results_df['base_gmv'] * 100).round(2)
                        results_df['margin_increase_%'] = ((results_df['margin'] - results_df['base_margin']) / results_df['base_margin'] * 100).round(2)

                        st.subheader("Optimization Results")
                        styled_df = results_df.style.format({
                            'optimal_price': "{:.2f}", 'expected_demand': "{:.2f}", 'gmv': "{:.2f}",
                            'margin': "{:.2f}", 'score': "{:.2f}", 'price_per_sku': "{:.2f}",
                            'cost': "{:.2f}", 'base_gmv': "{:.2f}", 'base_margin': "{:.2f}",
                            'gmv_increase_%': "{:.2f}%", 'margin_increase_%': "{:.2f}%"
                        })
                        if MATPLOTLIB_AVAILABLE:
                            styled_df = styled_df.background_gradient(subset=['gmv_increase_%', 'margin_increase_%'], cmap='RdYlGn')
                        st.write(styled_df, unsafe_allow_html=True)


                        st.subheader("Distribution of GMV and Margin Increase")
                        col1, col2 = st.columns(2)

                        with col1:
                            fig_gmv = px.histogram(
                                results_df, x='gmv_increase_%', nbins=30,
                                title="Distribution of GMV Increase (%)",
                                labels={'gmv_increase_%': 'GMV Increase (%)'},
                                color_discrete_sequence=['#636EFA']
                            )
                            fig_gmv.update_layout(bargap=0.1)
                            st.plotly_chart(fig_gmv, use_container_width=True)

                        with col2:
                            fig_margin = px.histogram(
                                results_df, x='margin_increase_%', nbins=30,
                                title="Distribution of Margin Increase (%)",
                                labels={'margin_increase_%': 'Margin Increase (%)'},
                                color_discrete_sequence=['#EF553B']
                            )
                            fig_margin.update_layout(bargap=0.1)
                            st.plotly_chart(fig_margin, use_container_width=True)

                        st.subheader("Average GMV: Base vs Optimal")
                        avg_gmv = pd.DataFrame({
                            'Type': ['Base GMV', 'Optimal GMV'],
                            'Value': [results_df['base_gmv'].mean(), results_df['gmv'].mean()]
                        })
                        fig_avg_gmv = px.bar(
                            avg_gmv, x='Type', y='Value',
                            title="Average GMV Comparison",
                            color='Type',
                            color_discrete_map={'Base GMV': '#636EFA', 'Optimal GMV': '#00CC96'}
                        )
                        fig_avg_gmv.update_layout(showlegend=True, yaxis_title="GMV (units)")
                        st.plotly_chart(fig_avg_gmv, use_container_width=True)

                        st.subheader("Average Margin: Base vs Optimal")
                        avg_margin = pd.DataFrame({
                            'Type': ['Base Margin', 'Optimal Margin'],
                            'Value': [results_df['base_margin'].mean(), results_df['margin'].mean()]
                        })
                        fig_avg_margin = px.bar(
                            avg_margin, x='Type', y='Value',
                            title="Average Margin Comparison",
                            color='Type',
                            color_discrete_map={'Base Margin': '#EF553B', 'Optimal Margin': '#AB63FA'}
                        )
                        fig_avg_margin.update_layout(showlegend=True, yaxis_title="Margin (fraction)")
                        st.plotly_chart(fig_avg_margin, use_container_width=True)

                        csv = results_df.to_csv(index=False).encode()
                        st.download_button(
                            label="Download Optimized Data as CSV",
                            data=csv,
                            file_name="optimized_prices.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"Error in optimization: {response.text}")

            finally:
                os.unlink(tmp_file_path)