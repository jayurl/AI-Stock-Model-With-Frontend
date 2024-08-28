import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# Function to fetch data and perform analysis on any company
def analyze_stock(ticker, start_date="2010-01-01", end_date="2023-01-01"):
    # Fetching data for the given ticker
    data = yf.download(ticker, start=start_date, end=end_date)

    # Drop rows with missing values
    data = data.dropna()

    # Calculate 50-day and 200-day moving averages
    data["50_MA"] = data["Close"].rolling(window=50).mean()
    data["200_MA"] = data["Close"].rolling(window=200).mean()

    # Calculate daily returns
    data["Daily_Return"] = data["Close"].pct_change()

    # Calculate 30-day rolling volatility
    data["Volatility"] = data["Daily_Return"].rolling(window=30).std()

    # Calculate RSI
    def calculate_rsi(data, window=14):
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    data["RSI"] = calculate_rsi(data)

    # Define the target variable 'Success' based on a 5% increase in the next 30 days
    data["Future_30D_Return"] = data["Close"].shift(-30) / data["Close"] - 1
    data["Success"] = (data["Future_30D_Return"] > 0.05).astype(int)

    # Drop rows with NaN values (due to the shift operation)
    data = data.dropna()

    # Selecting the features to normalize
    features_to_normalize = [
        "Close",
        "50_MA",
        "200_MA",
        "Daily_Return",
        "Volatility",
        "RSI",
    ]

    # Initialize and apply StandardScaler
    scaler = StandardScaler()
    data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

    # Define features and target
    X = data[features_to_normalize]
    y = data["Success"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {ticker}: {accuracy:.2f}")

    # Display classification report
    print(
        f"Classification Report for {ticker}:\n", classification_report(y_test, y_pred)
    )

    # Display confusion matrix
    print(f"Confusion Matrix for {ticker}:\n", confusion_matrix(y_test, y_pred))

    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"Cross-Validation Scores for {ticker}: {cv_scores}")
    print(f"Mean CV Accuracy for {ticker}: {cv_scores.mean():.2f}")

    # Get and display feature importances
    feature_importances = pd.Series(
        rf_model.feature_importances_, index=features_to_normalize
    )
    print(
        f"Feature Importances for {ticker}:\n",
        feature_importances.sort_values(ascending=False),
    )

    # Save the final prepared data
    data.to_csv(f"{ticker}_prepared_data_with_success.csv")


# Example usage
ticker = input("Enter the stock ticker symbol (e.g., AAPL, MSFT, TSLA): ")
analyze_stock(ticker)
