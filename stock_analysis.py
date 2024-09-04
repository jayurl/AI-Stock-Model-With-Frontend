import yfinance as yf


def calculate_metrics(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.financials
    balance_sheet = stock.balance_sheet

    metrics = {}

    # Print the available periods to confirm
    print("Available Periods in Financials DataFrame:")
    print(financials.columns)

    try:
        # Revenue and Net Income Data
        revenue = financials.loc["Total Revenue"]
        net_income = financials.loc["Net Income"]

        # YoY Revenue Growth (Comparing latest year to the previous year)
        yoy_revenue_growth = (revenue.iloc[0] - revenue.iloc[1]) / revenue.iloc[1]
        yoy_net_income_growth = (
            net_income.iloc[0] - net_income.iloc[1]
        ) / net_income.iloc[1]

        metrics["yoy_revenue_growth"] = round(yoy_revenue_growth, 4)
        metrics["yoy_net_income_growth"] = round(yoy_net_income_growth, 4)

    except Exception as e:
        metrics["yoy_revenue_growth"] = None
        metrics["yoy_net_income_growth"] = None
        print(f"YoY Growth Calculation Error: {e}")

    try:
        # Calculate ROE using "Stockholders Equity"
        shareholder_equity = balance_sheet.loc["Stockholders Equity"]
        roe = net_income.iloc[0] / shareholder_equity.iloc[0]

        metrics["roe"] = round(roe, 4)

    except Exception as e:
        metrics["roe"] = None
        print(f"ROE Calculation Error: {e}")

    return metrics


def analyze_stock(ticker):
    metrics = calculate_metrics(ticker)

    print(f"\nFinancial Metrics for {ticker.upper()}:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    ticker = "AAPL"
    analyze_stock(ticker)
