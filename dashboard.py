import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import zipfile
import os

# Set Page Config for Better Layout
st.set_page_config(page_title="Spare Parts Dashboard", layout="wide")

# Load dataset from ZIP file
@st.cache_data
def load_data():
    zip_path = "spare-parts.zip"
    extract_path = "extracted"
    
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
   
    csv_file = [f for f in os.listdir(extract_path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(extract_path, csv_file))

    # Convert Month column to datetime
    df["Month"] = pd.to_datetime(df["Month"], format="%Y%m")
    df["Promo_Flag"] = df["PromoID"].notna().astype(int)  # 1 if promo, 0 if no promo
    df["Month_Num"] = df["Month"].dt.month

    return df

df = load_data()

# Sidebar filters with Styled Sections
st.sidebar.header("🔍 Filters")
product_ids = df["ID"].unique()
selected_product = st.sidebar.selectbox("📌 Select Product ID", product_ids)
promo_filter = st.sidebar.radio("📊 Filter by Promotion", ["All", "Promoted", "Non-Promoted"])
forecast_model = st.sidebar.selectbox("🔮 Select Forecasting Model", ["Holt-Winters", "ARIMA", "XGBoost"])

# Apply Filters
df_filtered = df[df["ID"] == selected_product]
if promo_filter == "Promoted":
    df_filtered = df_filtered[df_filtered["Promo_Flag"] == 1]
elif promo_filter == "Non-Promoted":
    df_filtered = df_filtered[df_filtered["Promo_Flag"] == 0]

# Display Product Details with Better Formatting
st.markdown("## 📌 Product Details")
st.info(f"**Product ID:** {selected_product}\n\n**Promotion Status:** {'Yes' if df_filtered['Promo_Flag'].sum() > 0 else 'No'}")

# Sales trend visualization
st.markdown("## 📈 Sales Trend")
fig = px.line(df_filtered, x="Month", y="Quantity", title="📊 Sales Over Time", markers=True)
st.plotly_chart(fig, use_container_width=True)

# Seasonality Analysis
st.markdown("## ⏳ Seasonality Analysis")
df_filtered["Season"] = df_filtered["Month_Num"].apply(lambda x: "Summer" if x in [6,7,8] else ("Winter" if x in [12,1,2] else "Other"))
seasonality = df_filtered.groupby("Season")["Quantity"].mean().reset_index()
fig_seasonality = px.bar(seasonality, x="Season", y="Quantity", title="Seasonal Sales Trends")
st.plotly_chart(fig_seasonality, use_container_width=True)

# Forecasting Future Sales
st.markdown("## 🔮 Future Sales Prediction")
if len(df_filtered) > 3:
    if forecast_model == "Holt-Winters":
        model = ExponentialSmoothing(df_filtered["Quantity"], trend="add", seasonal="add", seasonal_periods=12).fit()
        forecast = model.forecast(6)
    elif forecast_model == "ARIMA":
        model = ARIMA(df_filtered["Quantity"], order=(5,1,0)).fit()
        forecast = model.forecast(steps=6)
    elif forecast_model == "XGBoost":
        X = np.arange(len(df_filtered)).reshape(-1,1)
        y = df_filtered["Quantity"].values
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X, y)
        forecast = model.predict(np.arange(len(df_filtered), len(df_filtered) + 6).reshape(-1,1))
    forecast_df = pd.DataFrame({"Month": pd.date_range(start=df_filtered["Month"].max(), periods=7, freq="M")[1:], "Forecast": forecast})
    fig_forecast = px.line(forecast_df, x="Month", y="Forecast", title=f"Future Sales Prediction ({forecast_model})")
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Not enough data to generate a forecast.")
