import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def load_data(file_path="Nat_Gas.csv"):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["MonthIndex"] = np.arange(len(df))
    df["Month"] = df["Date"].dt.month
    return df

def train_model(df):
    X = df[["MonthIndex"]]
    y = df["Price"]
    trend_model = LinearRegression()
    trend_model.fit(X, y)
    df["Trend"] = trend_model.predict(X)
    df["Seasonal"] = df["Price"] - df["Trend"]
    seasonal_factors = df.groupby("Month")["Seasonal"].mean().to_dict()
    return trend_model, seasonal_factors, df

def estimate_gas_price(input_date, df, trend_model, seasonal_factors):
    input_date = pd.to_datetime(input_date)
    start_date = df["Date"].iloc[0]
    months_since_start = ((input_date.year - start_date.year) * 12
                          + (input_date.month - start_date.month))
    trend_price = trend_model.predict([[months_since_start]])[0]
    seasonal_adjustment = seasonal_factors.get(input_date.month, 0)
    return round(float(trend_price + seasonal_adjustment), 2)

def build_forecast(df, trend_model, seasonal_factors, forecast_months=12):
    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.offsets.MonthEnd(1),
        periods=forecast_months,
        freq="ME"
    )
    forecast_prices = [
        estimate_gas_price(date, df, trend_model, seasonal_factors)
        for date in future_dates
    ]
    return pd.DataFrame({"Date": future_dates, "ForecastPrice": forecast_prices})
