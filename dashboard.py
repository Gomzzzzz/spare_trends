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

# Set Page Config for Better Layout
st.set_page_config(page_title="Spare Parts Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("notebooks/spare-parts.csv")
    df["Month"] = pd.to_datetime(df["Month"], format="%Y%m")
    df["Promo_Flag"] = df["PromoID"].notna().astype(int)  # 1 if promo, 0 if no promo
    df["Month_Num"] = df["Month"].dt.month
    return df

df = load_data()

# Sidebar filters with Styled Sections
st.sidebar.image("https://via.placeholder.com/150", caption="Company Logo", use_column_width=True)
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

# Promotion Impact Analysis with Conditional Display
st.markdown("## 💰 True Impact of Promotions on Sales")
if df_filtered["Promo_Flag"].sum() > 0:
    non_promo_data = df_filtered[df_filtered["Promo_Flag"] == 0]
    if not non_promo_data.empty:
        X_train = non_promo_data[["Month_Num"]]
        y_train = non_promo_data["Quantity"]
        model = LinearRegression()
        model.fit(X_train, y_train)
        df_filtered["Expected_Sales"] = model.predict(df_filtered[["Month_Num"]])
        df_filtered["Promotion_Impact"] = df_filtered["Quantity"] - df_filtered["Expected_Sales"]
        fig_promo_impact = px.bar(df_filtered, x="Month", y="Promotion_Impact", title="Promotion Impact on Sales")
        st.plotly_chart(fig_promo_impact, use_container_width=True)
    else:
        st.warning("Not enough non-promotion data to estimate promotion impact.")
else:
    st.info("This product has never undergone promotion, so promotion impact analysis is not applicable.")

# Seasonality Analysis
st.markdown("## ⏳ Seasonality Analysis")
df_filtered["Season"] = df_filtered["Month_Num"].apply(lambda x: "Summer" if x in [6,7,8] else ("Winter" if x in [12,1,2] else "Other"))
seasonality = df_filtered.groupby("Season")["Quantity"].mean().reset_index()
fig_seasonality = px.bar(seasonality, x="Season", y="Quantity", title="Seasonal Sales Trends")
st.plotly_chart(fig_seasonality, use_container_width=True)

# AI-Based Dynamic Pricing Optimization
st.markdown("## 🏷️ AI-Based Dynamic Pricing Suggestion")
if "Price" in df_filtered.columns:
    df_filtered["Price_Elasticity"] = df_filtered["Quantity"].pct_change() / df_filtered["Price"].pct_change()
    optimal_discount = -1 / df_filtered["Price_Elasticity"].mean()
    st.success(f"📌 Suggested Discount Rate for Maximum Profit: {optimal_discount:.2f}%")
else:
    st.warning("Pricing data not available for this product.")

# Smart Inventory Optimization
st.markdown("## 🚚 Smart Inventory Optimization")
reorder_point = df_filtered["Quantity"].mean() * 1.2  # Example: 20% buffer
df_filtered["Reorder_Flag"] = df_filtered["Quantity"] < reorder_point
if df_filtered["Reorder_Flag"].any():
    st.error(f"⚠️ Inventory Alert: Consider restocking Product {selected_product} soon!")

# Run command: streamlit run dashboard.py
