from flask import Flask, request, jsonify
from stock_analysis import analyze_stock

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    ticker = data.get("ticker")

    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400

    try:
        # Perform analysis on the given ticker
        analyze_stock(ticker)
        return (
            jsonify({"message": f"Analysis for {ticker} completed successfully"}),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
