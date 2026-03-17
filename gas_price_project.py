from model import load_data, train_model, build_forecast, estimate_gas_price
import matplotlib.pyplot as plt

def plot_results(df, forecast_df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Price"], marker="o", label="Historical Price")
    plt.plot(df["Date"], df["Trend"], linestyle="--", label="Trend")
    plt.plot(forecast_df["Date"], forecast_df["ForecastPrice"], marker="o", label="Forecast")
    plt.title("Natural Gas Price Analysis and Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    trend_model, seasonal_factors, df = train_model(df)
    forecast_df = build_forecast(df, trend_model, seasonal_factors)
    print("Natural Gas Price Analysis Project")
    print("-" * 40)
    user_date = input("Enter a date (YYYY-MM-DD): ").strip()
    try:
        estimated_price = estimate_gas_price(user_date, df, trend_model, seasonal_factors)
        print(f"Estimated natural gas price for {user_date}: {estimated_price}")
    except Exception as e:
        print("Invalid date format. Use YYYY-MM-DD.")
        print("Error:", e)
    plot_results(df, forecast_df)

if __name__ == "__main__":
    main()
