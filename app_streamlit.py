import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Set up folders
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

DATA_PATH = os.path.join(MODEL_FOLDER, "data.csv")
LSTM_MODEL_PATH = os.path.join(MODEL_FOLDER, "lstm_model.h5")
XGB_MODEL_PATH = os.path.join(MODEL_FOLDER, "xgb_model.pkl")
SCALER_PATH = os.path.join(MODEL_FOLDER, "scaler.pkl")
MODEL_TYPE_PATH = os.path.join(MODEL_FOLDER, "model_type.txt")

# Page config
st.set_page_config(page_title="TeraTech - Crop Price Predictor", layout="wide", page_icon="🌾")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .farmer-box {
        background-color: #E8F5E9;
        border-color: #4CAF50;
    }
    .consumer-box {
        background-color: #E3F2FD;
        border-color: #2196F3;
    }
    .government-box {
        background-color: #FFF3E0;
        border-color: #FF9800;
    }
    .summary-box {
        background-color: #F3E5F5;
        border-color: #9C27B0;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌾 TeraTech</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Crop Price Forecasting System</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio("Choose an action:", ["🏠 Home", "🎯 Train Model", "📊 Predict Future Prices"])

# Home page
if mode == "🏠 Home":
    st.markdown("## Welcome to TeraTech! 👋")
    st.write("""
    TeraTech uses advanced machine learning models to predict crop prices and provide actionable recommendations 
    for farmers, consumers, and policymakers.

    ### 🚀 Features:
    - **Multiple ML Models**: Choose from LSTM, XGBoost, or Hybrid models
    - **Custom Dataset Training**: Upload your own CSV data
    - **Future Price Forecasting**: Predict prices for up to 60 days
    - **Smart Recommendations**: Get tailored advice for different stakeholders

    ### 📋 How to Use:
    1. **Train Model**: Upload a CSV file with 'date' and 'value' columns
    2. **Select Model Type**: Choose LSTM, XGBoost, or Hybrid
    3. **Predict**: Forecast future prices and get recommendations

    ### 📁 CSV Format Example:
    ```
    date,value
    2024-01-01,45.5
    2024-01-02,46.2
    2024-01-03,45.8
    ```

    👈 **Get started by selecting an action from the sidebar!**
    """)

# Train Model page
elif mode == "🎯 Train Model":
    st.markdown("## 🎯 Train Your Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("📂 Upload your CSV file (with 'date' and 'value' columns)", type=['csv'])

        if uploaded_file:
            try:
                # Try reading with different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')

                # Check if dataframe is valid
                if df.empty or len(df.columns) == 0:
                    st.error("❌ CSV file appears to be empty or improperly formatted.")
                else:
                    st.success("✅ File uploaded successfully!")

                    # Show column names first
                    st.info(f"**Detected columns:** {', '.join(df.columns.tolist())}")

                    # Check if required columns exist
                    if 'date' in df.columns and 'value' in df.columns:
                        # Try to parse dates
                        try:
                            df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
                        except:
                            try:
                                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                            except:
                                df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

                        st.write("### 📊 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)

                        st.write("### 📈 Data Statistics")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Records", len(df))
                        col_b.metric("Average Value", f"{df['value'].mean():.2f}")
                        col_c.metric("Value Range", f"{df['value'].min():.2f} - {df['value'].max():.2f}")
                    else:
                        st.warning(f"⚠️ Required columns 'date' and 'value' not found. Please rename your columns.")
                        st.write("**Your current columns:**", df.columns.tolist())
                        st.write("**Sample data:**")
                        st.dataframe(df.head())

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.write(
                    "**Debug info:** Try saving your CSV with UTF-8 encoding or check if the file has any special characters.")

    with col2:
        st.write("### ⚙️ Model Configuration")
        model_type = st.selectbox(
            "Select Model Type",
            ["lstm", "xgboost", "hybrid"],
            help="LSTM: Good for time series patterns | XGBoost: Good for feature-based predictions | Hybrid: Combines both"
        )

        st.info(f"""
        **Selected: {model_type.upper()}**

        {'🔮 LSTM excels at capturing sequential patterns in time series data.' if model_type == 'lstm' else ''}
        {'⚡ XGBoost is fast and handles feature engineering well.' if model_type == 'xgboost' else ''}
        {'🎯 Hybrid combines the strengths of both models for better accuracy.' if model_type == 'hybrid' else ''}
        """)

    if uploaded_file and st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner(f"Training {model_type.upper()} model... This may take a few minutes ⏳"):
            try:
                # Try reading with different encodings and skip bad lines
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')

                # Check if dataframe is empty
                if df.empty or len(df.columns) == 0:
                    st.error("❌ CSV file appears to be empty or improperly formatted.")
                    st.stop()

                # Check columns
                if 'date' not in df.columns or 'value' not in df.columns:
                    st.error(f"❌ CSV must contain 'date' and 'value' columns. Found columns: {list(df.columns)}")
                    st.write("Your columns:", df.columns.tolist())
                    st.stop()

                # Handle different date formats including MM-DD-YYYY
                try:
                    df["date"] = pd.to_datetime(df["date"], format='%m-%d-%Y')
                except:
                    try:
                        df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d')
                    except:
                        try:
                            df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True)
                        except:
                            st.error(
                                "❌ Could not parse date column. Please ensure dates are in MM-DD-YYYY or YYYY-MM-DD format.")
                            st.stop()

                df = df.sort_values("date").reset_index(drop=True)

                scaler = MinMaxScaler()
                df['scaled_value'] = scaler.fit_transform(df[['value']])
                window_size = 10

                progress_bar = st.progress(0)

                # LSTM Training
                if model_type in ["lstm", "hybrid"]:
                    progress_bar.progress(20)
                    X_lstm, y_lstm = [], []
                    for i in range(len(df) - window_size):
                        X_lstm.append(df['scaled_value'].values[i:i + window_size])
                        y_lstm.append(df['scaled_value'].values[i + window_size])
                    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

                    progress_bar.progress(40)
                    lstm_model = Sequential([
                        LSTM(50, activation='relu', input_shape=(window_size, 1)),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss='mse')
                    lstm_model.fit(X_lstm, y_lstm, epochs=30, verbose=0)
                    lstm_model.save(LSTM_MODEL_PATH)

                    with open(SCALER_PATH, "wb") as f:
                        pickle.dump(scaler, f)
                    progress_bar.progress(60)

                # XGBoost Training
                if model_type in ["xgboost", "hybrid"]:
                    progress_bar.progress(40 if model_type == "xgboost" else 60)

                    df["dayofweek"] = df["date"].dt.dayofweek
                    df["month"] = df["date"].dt.month
                    df["dayofyear"] = df["date"].dt.dayofyear
                    df["lag_1"] = df["value"].shift(1)
                    df["lag_2"] = df["value"].shift(2)
                    df["rolling_mean_3"] = df["value"].rolling(window=3).mean()
                    df["rolling_std_3"] = df["value"].rolling(window=3).std()
                    df["diff_1"] = df["value"].diff()

                    if model_type == "hybrid":
                        preds = lstm_model.predict(X_lstm, verbose=0).flatten()
                        df = df.iloc[window_size:].copy()
                        df["lstm_pred"] = preds

                    df = df.dropna().reset_index(drop=True)

                    features = ["dayofweek", "month", "dayofyear", "lag_1", "lag_2",
                                "rolling_mean_3", "rolling_std_3", "diff_1"]
                    if model_type == "hybrid":
                        features.append("lstm_pred")

                    X = df[features]
                    y = df['value']

                    progress_bar.progress(70)
                    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                    xgb_model.fit(X, y)

                    with open(XGB_MODEL_PATH, "wb") as f:
                        pickle.dump(xgb_model, f)
                    progress_bar.progress(90)

                # Save metadata
                with open(MODEL_TYPE_PATH, 'w') as f:
                    f.write(model_type)

                # Save original data for prediction (reset file pointer first)
                uploaded_file.seek(0)
                try:
                    df_original = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    try:
                        df_original = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df_original = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')

                # Parse dates with multiple format support
                try:
                    df_original["date"] = pd.to_datetime(df_original["date"], format='%m-%d-%Y')
                except:
                    try:
                        df_original["date"] = pd.to_datetime(df_original["date"], format='%Y-%m-%d')
                    except:
                        df_original["date"] = pd.to_datetime(df_original["date"], infer_datetime_format=True)

                df_original.to_csv(DATA_PATH, index=False)

                progress_bar.progress(100)
                st.success(f"🎉 {model_type.upper()} model trained successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

# Predict page
elif mode == "📊 Predict Future Prices":
    st.markdown("## 📊 Forecast Future Crop Prices")

    # Check if model exists
    if not os.path.exists(MODEL_TYPE_PATH):
        st.warning("⚠️ No trained model found. Please train a model first!")
        st.stop()

    col1, col2 = st.columns([1, 3])

    with col1:
        days = st.slider("📅 Forecast Period (days)", min_value=1, max_value=60, value=7)
        st.info(f"Predicting prices for the next **{days} days**")

    with col2:
        with open(MODEL_TYPE_PATH, 'r') as f:
            current_model = f.read().strip()
        st.success(f"✅ Using **{current_model.upper()}** model")

    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating predictions... 🔄"):
            try:
                df = pd.read_csv(DATA_PATH)
                df["date"] = pd.to_datetime(df["date"])

                with open(MODEL_TYPE_PATH, 'r') as f:
                    model_type = f.read().strip()

                # Load models
                scaler = None
                if model_type in ["lstm", "hybrid"]:
                    lstm_model = load_model(LSTM_MODEL_PATH, compile=False)
                    with open(SCALER_PATH, "rb") as f:
                        scaler = pickle.load(f)

                if model_type in ["xgboost", "hybrid"]:
                    with open(XGB_MODEL_PATH, "rb") as f:
                        xgb_model = pickle.load(f)

                future_dates, future_values = [], []
                last_known_date = df["date"].max()

                # Generate predictions
                for i in range(days):
                    if model_type in ["lstm", "hybrid"]:
                        last_window = scaler.transform(df["value"].values[-10:].reshape(-1, 1)).reshape(1, 10, 1)
                        lstm_pred = lstm_model.predict(last_window, verbose=0)[0][0]
                        lstm_pred_value = scaler.inverse_transform([[lstm_pred]])[0][0]

                    if model_type in ["xgboost", "hybrid"]:
                        new_date = last_known_date + timedelta(days=1)
                        lag_1 = df["value"].iloc[-1]
                        lag_2 = df["value"].iloc[-2] if len(df) > 1 else lag_1
                        rolling_mean_3 = df["value"].iloc[-3:].mean()
                        rolling_std_3 = df["value"].iloc[-3:].std()
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

                        xgb_pred = xgb_model.predict(pd.DataFrame([features]))[0]

                    # Final prediction
                    if model_type == "lstm":
                        final_pred = lstm_pred_value
                    elif model_type == "xgboost":
                        final_pred = xgb_pred
                    else:  # hybrid
                        final_pred = 0.5 * lstm_pred_value + 0.5 * xgb_pred

                    # Add realistic noise
                    final_pred += np.random.normal(loc=0, scale=0.8)
                    final_pred = max(0, final_pred)

                    new_date = last_known_date + timedelta(days=1)
                    df = pd.concat([df, pd.DataFrame({"date": [new_date], "value": [final_pred]})], ignore_index=True)
                    last_known_date = new_date
                    future_dates.append(new_date.strftime("%Y-%m-%d"))
                    future_values.append(final_pred)

                # Display results
                st.markdown("---")
                st.markdown("## 📈 Forecast Results")

                # Overall trend summary
                overall_change = future_values[-1] - future_values[0]
                if overall_change > 1:
                    summary = "📈 **Strong upward trend expected.** Good time to hold crops."
                    summary_color = "#4CAF50"
                elif overall_change < -1:
                    summary = "📉 **Downward trend ahead.** Consider early selling."
                    summary_color = "#F44336"
                else:
                    summary = "🔄 **Prices expected to remain stable.** Plan accordingly."
                    summary_color = "#FF9800"

                st.markdown(f'<div class="summary-box" style="border-color: {summary_color};">{summary}</div>',
                            unsafe_allow_html=True)

                # Visualization
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(future_dates, future_values, marker='o', linestyle='-', color='teal', linewidth=2, markersize=6)
                ax.fill_between(range(len(future_dates)), future_values, alpha=0.3, color='teal')
                ax.set_xlabel("Date", fontsize=12, fontweight='bold')
                ax.set_ylabel("Predicted Price", fontsize=12, fontweight='bold')
                ax.set_title("Forecasted Crop Prices", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

                # Forecast table with recommendations
                st.markdown("### 📋 Detailed Day-by-Day Forecast")

                forecast_data = []
                for i in range(len(future_dates)):
                    change = future_values[i] - future_values[i - 1] if i > 0 else 0

                    # Determine recommendations based on price change
                    if change > 0.8:
                        farmer_rec = "🟢 Hold your crop, price is increasing"
                        consumer_rec = "🔴 Buy now before prices rise more"
                        gov_rec = "⚠️ Monitor for inflation; consider price caps"
                    elif change < -0.8:
                        farmer_rec = "🔴 Consider selling before price drops further"
                        consumer_rec = "🟢 Delay buying; prices may go down"
                        gov_rec = "✅ No major action; ensure fair pricing"
                    else:
                        farmer_rec = "🟡 Stable prices; act as usual"
                        consumer_rec = "🟡 Stable market; buy as needed"
                        gov_rec = "✅ Maintain monitoring; no action required"

                    forecast_data.append({
                        "Date": future_dates[i],
                        "Predicted Price": f"{future_values[i]:.2f}",
                        "Change": f"{change:+.2f}",
                        "Farmer/Seller": farmer_rec,
                        "Consumer": consumer_rec,
                        "Government": gov_rec
                    })

                df_forecast = pd.DataFrame(forecast_data)
                st.dataframe(df_forecast, use_container_width=True, hide_index=True)

                # Stakeholder-specific recommendations
                st.markdown("### 🎯 Stakeholder Recommendations")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="recommendation-box farmer-box">', unsafe_allow_html=True)
                    st.markdown("#### 👨‍🌾 For Farmers & Sellers")
                    if overall_change > 1:
                        st.write("✅ **Hold Strategy Recommended**")
                        st.write("- Wait for peak prices")
                        st.write("- Monitor daily trends")
                        st.write("- Prepare for harvest timing")
                    elif overall_change < -1:
                        st.write("⚠️ **Sell Early Recommended**")
                        st.write("- Sell before further decline")
                        st.write("- Lock in current prices")
                        st.write("- Consider forward contracts")
                    else:
                        st.write("📊 **Normal Market Conditions**")
                        st.write("- Follow regular selling pattern")
                        st.write("- No urgent action needed")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="recommendation-box consumer-box">', unsafe_allow_html=True)
                    st.markdown("#### 🛒 For Consumers")
                    if overall_change > 1:
                        st.write("🔴 **Buy Now**")
                        st.write("- Prices rising ahead")
                        st.write("- Stock up if possible")
                        st.write("- Expect higher costs soon")
                    elif overall_change < -1:
                        st.write("🟢 **Wait for Better Prices**")
                        st.write("- Prices declining")
                        st.write("- Delay purchases if possible")
                        st.write("- Better deals coming")
                    else:
                        st.write("📊 **Stable Pricing**")
                        st.write("- Buy as needed")
                        st.write("- Normal market conditions")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="recommendation-box government-box">', unsafe_allow_html=True)
                    st.markdown("#### 🏛️ For Government/Policy")
                    if overall_change > 1:
                        st.write("⚠️ **Action Required**")
                        st.write("- Monitor inflation risk")
                        st.write("- Consider price regulations")
                        st.write("- Ensure supply stability")
                    elif overall_change < -1:
                        st.write("✅ **Minimal Intervention**")
                        st.write("- Ensure fair pricing")
                        st.write("- Protect farmer interests")
                        st.write("- Monitor market health")
                    else:
                        st.write("✅ **Continue Monitoring**")
                        st.write("- Maintain current policies")
                        st.write("- No intervention needed")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Download option
                st.markdown("---")
                csv = df_forecast.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name=f"terratech_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🌾 TeraTech - AI-Powered Crop Price Forecasting | © {datetime.now().year}</p>
    <p style="font-size: 0.8rem;">Built with Streamlit, TensorFlow, and XGBoost</p>
</div>
""", unsafe_allow_html=True)