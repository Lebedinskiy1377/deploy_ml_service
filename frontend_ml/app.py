"""Streamlit UI for the dynamic pricing API."""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8005").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120

NUMBER_FORMATS = {
    "optimal_price": "{:.2f}",
    "expected_demand": "{:.2f}",
    "gmv": "{:.2f}",
    "margin": "{:.2f}",
    "score": "{:.2f}",
    "price_per_sku": "{:.2f}",
    "cost": "{:.2f}",
    "base_demand": "{:.2f}",
    "base_gmv": "{:.2f}",
    "base_margin": "{:.2f}",
    "gmv_increase_%": "{:.2f}%",
    "margin_increase_%": "{:.2f}%",
}


def post_csv(endpoint: str, data: pd.DataFrame, filename: str) -> pd.DataFrame:
    response = requests.post(
        f"{API_BASE_URL}{endpoint}",
        files={"file": (filename, data.to_csv(index=False).encode("utf-8"), "text/csv")},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())


def percent_change(current: pd.Series, base: pd.Series) -> pd.Series:
    current_values = current.to_numpy(dtype=float)
    base_values = base.to_numpy(dtype=float)
    change = np.divide(
        current_values - base_values,
        base_values,
        out=np.zeros_like(current_values),
        where=base_values != 0,
    )
    return pd.Series(change * 100, index=current.index).round(2)


def api_error_message(exc: requests.HTTPError) -> str:
    if exc.response is None:
        return str(exc)
    try:
        return str(exc.response.json().get("detail", exc.response.text))
    except ValueError:
        return exc.response.text


def download_button(frame: pd.DataFrame, label: str, file_name: str) -> None:
    st.download_button(label=label, data=frame.to_csv(index=False).encode(), file_name=file_name, mime="text/csv")


def comparison_chart(title: str, labels: tuple[str, str], values: tuple[float, float], colors: tuple[str, str]):
    frame = pd.DataFrame({"Type": list(labels), "Value": list(values)})
    colors_by_label = dict(zip(labels, colors, strict=True))
    return px.bar(frame, x="Type", y="Value", title=title, color="Type", color_discrete_map=colors_by_label)


def show_demand(result: pd.DataFrame) -> None:
    st.write("Predictions with Original Data:")
    st.dataframe(result)
    download_button(result, "Download Predicted Data as CSV", "predicted_demand.csv")


def show_optimization(results: pd.DataFrame) -> None:
    # The API returns the base scenario (uploaded price) next to the optimum.
    results["gmv_increase_%"] = percent_change(results["gmv"], results["base_gmv"])
    results["margin_increase_%"] = percent_change(results["margin"], results["base_margin"])

    st.subheader("Optimization Results")
    styled = results.style.format(NUMBER_FORMATS).background_gradient(
        subset=["gmv_increase_%", "margin_increase_%"],
        cmap="RdYlGn",
    )
    st.dataframe(styled)

    st.subheader("Distribution of GMV and Margin Increase")
    gmv_column, margin_column = st.columns(2)
    with gmv_column:
        figure = px.histogram(
            results,
            x="gmv_increase_%",
            nbins=30,
            title="Distribution of GMV Increase (%)",
            labels={"gmv_increase_%": "GMV Increase (%)"},
            color_discrete_sequence=["#636EFA"],
        )
        figure.update_layout(bargap=0.1)
        st.plotly_chart(figure, use_container_width=True)
    with margin_column:
        figure = px.histogram(
            results,
            x="margin_increase_%",
            nbins=30,
            title="Distribution of Margin Increase (%)",
            labels={"margin_increase_%": "Margin Increase (%)"},
            color_discrete_sequence=["#EF553B"],
        )
        figure.update_layout(bargap=0.1)
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Average GMV: Base vs Optimal")
    figure = comparison_chart(
        "Average GMV Comparison",
        ("Base GMV", "Optimal GMV"),
        (results["base_gmv"].mean(), results["gmv"].mean()),
        ("#636EFA", "#00CC96"),
    )
    figure.update_layout(yaxis_title="GMV (units)")
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Average Margin: Base vs Optimal")
    figure = comparison_chart(
        "Average Margin Comparison",
        ("Base Margin", "Optimal Margin"),
        (results["base_margin"].mean(), results["margin"].mean()),
        ("#EF553B", "#AB63FA"),
    )
    figure.update_layout(yaxis_title="Margin (fraction)")
    st.plotly_chart(figure, use_container_width=True)

    download_button(results, "Download Optimized Data as CSV", "optimized_prices.csv")


def main() -> None:
    st.title("Price Prediction and Optimization Web Application")
    st.write("Upload a CSV file with SKU prices to predict demand or find the optimal price.")
    st.caption("Required columns: dates, SKU, price_per_sku. Example: examples/request.csv in the repository.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is None:
        return

    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data)

    action = st.radio("Select action:", ["Predict Demand", "Optimize Price"])
    if not st.button("Submit"):
        return

    try:
        if action == "Predict Demand":
            show_demand(post_csv("/invocation", data, uploaded_file.name))
        else:
            show_optimization(post_csv("/optimize_price", data, uploaded_file.name))
    except requests.HTTPError as exc:
        st.error(f"API request failed: {api_error_message(exc)}")
    except requests.RequestException as exc:
        st.error(f"Could not connect to API at {API_BASE_URL}: {exc}")


if __name__ == "__main__":
    main()
