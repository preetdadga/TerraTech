from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import random

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

DATA_PATH = os.path.join(MODEL_FOLDER, "data.csv")
LSTM_MODEL_PATH = os.path.join(MODEL_FOLDER, "lstm_model.h5")
XGB_MODEL_PATH = os.path.join(MODEL_FOLDER, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")
MODEL_TYPE_PATH = os.path.join(MODEL_FOLDER, "model_type.txt")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        days = int(request.form.get("days"))
        df = pd.read_csv(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])

        with open(MODEL_TYPE_PATH, 'r') as f:
            model_type = f.read().strip()

        future_dates = []
        future_values = []
        last_known_date = df['date'].max()

        if model_type in ['lstm', 'hybrid']:
            lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)

        if model_type in ['xgboost', 'hybrid']:
            with open(XGB_MODEL_PATH, 'rb') as f:
                xgb_model = pickle.load(f)

        for i in range(days):
            if model_type in ['lstm', 'hybrid']:
                last_window = scaler.transform(df['value'].values[-10:].reshape(-1, 1)).reshape(1, 10, 1)
                lstm_pred = lstm_model.predict(last_window, verbose=0)[0][0]
                lstm_pred_value = scaler.inverse_transform([[lstm_pred]])[0][0]

            if model_type in ['xgboost', 'hybrid']:
                new_date = last_known_date + timedelta(days=1)
                lag_1 = df['value'].iloc[-1]
                lag_2 = df['value'].iloc[-2] if len(df) > 1 else lag_1
                rolling_mean_3 = df['value'].iloc[-3:].mean()
                rolling_std_3 = df['value'].iloc[-3:].std()
                diff_1 = lag_1 - lag_2

                features = {
                    "dayofweek": new_date.dayofweek,
                    "month": new_date.month,
                    "dayofyear": new_date.timetuple().tm_yday,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "rolling_mean_3": rolling_mean_3,
                    "rolling_std_3": rolling_std_3,
                    "diff_1": diff_1
                }

                if model_type == "hybrid":
                    features["lstm_pred"] = lstm_pred_value

                xgb_features = pd.DataFrame([features])
                xgb_pred = xgb_model.predict(xgb_features)[0]

            if model_type == "lstm":
                final_pred = lstm_pred_value
            elif model_type == "xgboost":
                final_pred = xgb_pred
            else:
                final_pred = 0.5 * lstm_pred_value + 0.5 * xgb_pred

            final_pred += np.random.normal(loc=0, scale=0.8)
            final_pred = max(0, final_pred)

            new_date = last_known_date + timedelta(days=1)
            future_dates.append(new_date.strftime("%Y-%m-%d"))
            future_values.append(final_pred)

            df = pd.concat([df, pd.DataFrame({"date": [new_date], "value": [final_pred]})], ignore_index=True)
            last_known_date = new_date

        plt.figure(figsize=(10, 4))
        plt.plot(future_dates, future_values, marker='o', linestyle='-', color='teal')
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Predicted Price")
        plt.title("Forecasted Crop Prices")
        plt.tight_layout()
        graph_path = os.path.join("static", "forecast.png")
        plt.savefig(graph_path)
        plt.close()

        forecast_table = list(zip(future_dates, future_values))

        # 📌 Updated summary recommendation with smaller delta
        overall_change = future_values[-1] - future_values[0]
        if overall_change > 1:
            summary_recommendation = "📈 Strong upward trend expected. Good time to hold crops."
        elif overall_change < -1:
            summary_recommendation = "📉 Downward trend ahead. Consider early selling."
        else:
            summary_recommendation = "🔄 Prices expected to remain stable. Plan accordingly."

        farmer_seller_recommendations = []
        consumer_recommendations = []
        government_recommendations = []

        for i in range(len(forecast_table)):
            change = forecast_table[i][1] - forecast_table[i - 1][1] if i > 0 else 0

            if change > 0.8:
                farmer_seller_recommendations.append({"farmer_seller_action": "Hold your crop, price is increasing."})
                consumer_recommendations.append({"consumer_action": "Buy now before prices rise more."})
                government_recommendations.append({"government_action": "Monitor for inflation; consider price caps."})
            elif change < -0.8:
                farmer_seller_recommendations.append({"farmer_seller_action": "Consider selling before price drops further."})
                consumer_recommendations.append({"consumer_action": "Delay buying; prices may go down."})
                government_recommendations.append({"government_action": "No major action; ensure fair pricing."})
            else:
                farmer_seller_recommendations.append({"farmer_seller_action": "Stable prices; act as usual."})
                consumer_recommendations.append({"consumer_action": "Stable market; buy as needed."})
                government_recommendations.append({"government_action": "Maintain monitoring; no action required."})

        return render_template("index.html",
            forecast_table=forecast_table,
            graph_url=graph_path,
            summary_recommendation=summary_recommendation,
            farmer_seller_recommendations=farmer_seller_recommendations,
            consumer_recommendations=consumer_recommendations,
            government_recommendations=government_recommendations,
            current_year=datetime.now().year
        )

    except Exception as e:
        return render_template("index.html", error=f"Prediction failed: {str(e)}", current_year=datetime.now().year)

@app.route("/")
def home():
    return render_template("index.html", current_year=datetime.now().year)

@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("csv_file")
    model_type = request.form.get("model_type")
    if not file:
        return render_template("index.html", error="Please upload a CSV file.", current_year=datetime.now().year)

    try:
        df = pd.read_csv(file)
        expected_columns = {"date", "value"}
        if not expected_columns.issubset(df.columns):
            return render_template("index.html", error="CSV must contain 'date' and 'value' columns.", current_year=datetime.now().year)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        scaler = MinMaxScaler()
        df['scaled_value'] = scaler.fit_transform(df[['value']])

        window_size = 10

        if model_type == "lstm":
            X_lstm, y_lstm = [], []
            for i in range(len(df) - window_size):
                X_lstm.append(df['scaled_value'].values[i:i + window_size])
                y_lstm.append(df['scaled_value'].values[i + window_size])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(window_size, 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_lstm, y_lstm, epochs=30, verbose=0)
            lstm_model.save(LSTM_MODEL_PATH)

            with open(SCALER_PATH, "wb") as f:
                pickle.dump(scaler, f)

            with open(MODEL_TYPE_PATH, 'w') as f:
                f.write("lstm")

        elif model_type == "xgboost":
            df["dayofweek"] = df["date"].dt.dayofweek
            df["month"] = df["date"].dt.month
            df["dayofyear"] = df["date"].dt.dayofyear
            df["lag_1"] = df["value"].shift(1)
            df["lag_2"] = df["value"].shift(2)
            df["rolling_mean_3"] = df["value"].rolling(window=3).mean()
            df["rolling_std_3"] = df["value"].rolling(window=3).std()
            df["diff_1"] = df["value"].diff()
            df = df.dropna().reset_index(drop=True)

            features = ["dayofweek", "month", "dayofyear", "lag_1", "lag_2", "rolling_mean_3", "rolling_std_3", "diff_1"]
            X = df[features]
            y = df['value']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X_train, y_train)

            with open(XGB_MODEL_PATH, "wb") as f:
                pickle.dump(xgb_model, f)

            with open(MODEL_TYPE_PATH, 'w') as f:
                f.write("xgboost")

        else:  # hybrid
            X_lstm, y_lstm = [], []
            for i in range(len(df) - window_size):
                X_lstm.append(df['scaled_value'].values[i:i + window_size])
                y_lstm.append(df['scaled_value'].values[i + window_size])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

            lstm_model = Sequential([
                LSTM(50, activation='relu', input_shape=(window_size, 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_lstm, y_lstm, epochs=30, verbose=0)
            lstm_model.save(LSTM_MODEL_PATH)
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(scaler, f)

            preds = lstm_model.predict(X_lstm).flatten()
            df = df.iloc[window_size:].copy()
            df["lstm_pred"] = preds

            df["dayofweek"] = df["date"].dt.dayofweek
            df["month"] = df["date"].dt.month
            df["dayofyear"] = df["date"].dt.dayofyear
            df["lag_1"] = df["value"].shift(1)
            df["lag_2"] = df["value"].shift(2)
            df["rolling_mean_3"] = df["value"].rolling(window=3).mean()
            df["rolling_std_3"] = df["value"].rolling(window=3).std()
            df["diff_1"] = df["value"].diff()
            df = df.dropna().reset_index(drop=True)

            features = ["dayofweek", "month", "dayofyear", "lag_1", "lag_2", "rolling_mean_3", "rolling_std_3", "diff_1", "lstm_pred"]
            X = df[features]
            y = df['value']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X_train, y_train)
            with open(XGB_MODEL_PATH, "wb") as f:
                pickle.dump(xgb_model, f)

            with open(MODEL_TYPE_PATH, 'w') as f:
                f.write("hybrid")

        df.to_csv(DATA_PATH, index=False)
        return render_template("index.html", message=f"{model_type.upper()} model trained successfully!", current_year=datetime.now().year)

    except Exception as e:
        return render_template("index.html", error=f"Training failed: {str(e)}", current_year=datetime.now().year)

if __name__ == "__main__":
    app.run(debug=True)
