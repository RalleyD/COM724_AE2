import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Set page config ---
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# --- Fake data for demo purposes ---


def generate_mock_market_data():
    coins = [f"Coin {i}" for i in range(1, 31)]
    data = {
        "Coin": coins,
        "24h Change (%)": np.random.normal(0, 5, 30).round(2),
        "7d Change (%)": np.random.normal(0, 10, 30).round(2),
        "30d Change (%)": np.random.normal(0, 20, 30).round(2),
        "Sparkline": [np.random.normal(1, 0.02, 20).cumsum() for _ in range(30)]
    }
    return pd.DataFrame(data)


def plot_sparkline(prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prices, mode='lines', line=dict(color='gray')))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=40)
    return fig


# --- UI Start ---
st.title("📊 Cryptocurrency Analysis Dashboard")

# --- Tab layout ---
tab1, tab2 = st.tabs(["🌐 Market Overview", "📈 Forecast & Insights"])

# ============ TAB 1 ============ #
with tab1:
    st.subheader("Top 30 Coins - Market Performance")

    market_df = generate_mock_market_data()

    # Show filters or summary stats (optional)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Global Market Change (24h)",
                  value=f"{market_df['24h Change (%)'].mean():.2f} %")

    # Display overview grid
    for i in range(0, len(market_df), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j >= len(market_df):
                break
            coin = market_df.iloc[i + j]
            with cols[j]:
                st.markdown(f"**{coin['Coin']}**")
                col_val = coin["24h Change (%)"]
                st.metric(
                    "24h", f"{col_val:+.2f}%", delta_color="inverse" if col_val < 0 else "normal")
                st.plotly_chart(plot_sparkline(
                    coin["Sparkline"]), use_container_width=True)

# ============ TAB 2 ============ #
with tab2:
    st.subheader("Coin Forecasting & Insights")

    # Sidebar-style controls
    st.markdown("### 🔍 Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_coin = st.selectbox(
            "Select Coin", [f"Coin {i}" for i in range(1, 31)])
    with col2:
        cluster = st.selectbox(
            "Cluster", ["All", "Cluster A", "Cluster B", "Cluster C", "Cluster D"])
    with col3:
        view_mode = st.radio("View", ["Historical", "Forecast"])

    st.markdown("### 📊 Prediction Graph")

    # Fake data for prediction chart
    dates = pd.date_range(datetime.today(), periods=30)
    price = np.cumsum(np.random.normal(0, 1, 30)) + 100
    lower = price - np.random.uniform(2, 4, 30)
    upper = price + np.random.uniform(2, 4, 30)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=price,
                  name="Price", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=dates, y=upper, name="Upper Bound",
                  line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=lower, name="Lower Bound", fill='tonexty',
                             fillcolor='rgba(0, 0, 255, 0.1)', line=dict(width=0), showlegend=False))

    fig.update_layout(
        title=f"{selected_coin} - {'Forecast' if view_mode == 'Forecast' else 'Historical'} View",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "💡 *Confidence intervals are based on model predictions and recent volatility.*")
