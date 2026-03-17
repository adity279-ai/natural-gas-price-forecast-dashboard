import streamlit as st
import matplotlib.pyplot as plt
from model import load_data, train_model, build_forecast, estimate_gas_price

st.set_page_config(page_title="Natural Gas Forecast Dashboard", page_icon="📈", layout="wide")

df = load_data()
trend_model, seasonal_factors, df = train_model(df)
forecast_df = build_forecast(df, trend_model, seasonal_factors)

st.title("📈 Natural Gas Price Forecasting Dashboard")
st.write("Analyze historical natural gas prices and estimate future prices using a simple trend + seasonality model.")

c1, c2, c3 = st.columns(3)
c1.metric("Data Points", len(df))
c2.metric("Start Date", str(df["Date"].min().date()))
c3.metric("End Date", str(df["Date"].max().date()))

st.subheader("Dataset Preview")
st.dataframe(df[["Date", "Price"]], use_container_width=True)

st.subheader("Price Estimator")
selected_date = st.date_input("Select a date")
estimated_price = estimate_gas_price(selected_date, df, trend_model, seasonal_factors)
st.success(f"Estimated natural gas price for {selected_date}: {estimated_price}")

st.subheader("Historical Trend and Forecast")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df["Price"], marker="o", label="Historical Price")
ax.plot(df["Date"], df["Trend"], linestyle="--", label="Trend")
ax.plot(forecast_df["Date"], forecast_df["ForecastPrice"], marker="o", label="Forecast")
ax.set_title("Natural Gas Price Analysis and Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.subheader("Forecast Table")
st.dataframe(forecast_df, use_container_width=True)

st.caption("Built with Python, Streamlit, Pandas, Matplotlib, and Scikit-learn.")
