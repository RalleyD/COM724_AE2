import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import time
from datetime import datetime, timedelta
import requests
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from backend.app.fetch_data_coingecko import get_top_30_coins
import warnings

warnings.filterwarnings('ignore')

#### TODO ####
# get_market_state should use percentage change from pandas
# xgboost forecast continuity, use revised method from notebook
# Check best practices for 7 and 30 day moving average plots
# extend to all coins.
# confidence interval (look this up)
# add target profit parameter to buy recommendation - should not retrigger forecast

# Set page configuration
st.set_page_config(
    page_title="Cryptocurrency Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .positive {
        color: #10B981;
    }
    .negative {
        color: #EF4444;
    }
    .correlation-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #E5E7EB;
    }
    .disclaimer {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ----- Data Fetching Functions -----


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_crypto_data(symbol="BTC", interval="1d", limit=1000):
    """
    Fetch historical cryptocurrency data from Binance API
    """
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": limit
        }

        response = requests.get(url, params=params)
        data = response.json()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Set timestamp as index
        df = df.set_index('timestamp', drop=True)

        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        st.error(f"Error fetching data from Binance: {e}")
        # Return mock data if API fails
        return generate_mock_data(symbol, limit)


@st.cache_data
def generate_mock_data(symbol, days=365):
    """
    TODO remove
    Generate mock data if API is not available"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Set different base prices for different coins
    if symbol == "BTC":
        base_price = 30000
        volatility = 0.03
    elif symbol == "ETH":
        base_price = 2000
        volatility = 0.04
    elif symbol == "SOL":
        base_price = 100
        volatility = 0.05
    elif symbol == "ADA":
        base_price = 0.5
        volatility = 0.06
    else:
        base_price = 1.0
        volatility = 0.04

    price = base_price
    prices = []

    for i in range(len(date_range)):
        # Random price movement with some trend
        change = np.random.normal(0, volatility)
        # Add some cycles
        if i % 7 == 0:  # Weekly pattern
            change += 0.01
        if i % 30 == 0:  # Monthly pattern
            change -= 0.02

        price = price * (1 + change)
        prices.append(price)

    # Create dataframe
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [abs(np.random.normal(base_price * 1000000, base_price * 500000)) for _ in prices]
    }, index=date_range)

    return df


@st.cache_data(ttl=3600)
def get_crypto_news(coin="bitcoin"):
    """Fetch cryptocurrency news from CryptoCompare"""
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        if coin != "all":
            url += f"&categories={coin}"

        response = requests.get(url)
        data = response.json()

        news_items = []
        for item in data['Data'][:6]:  # Get top 6 news items
            news_items.append({
                'title': item['title'],
                'url': item['url'],
                'source': item['source'],
                'published_at': datetime.fromtimestamp(item['published_on']),
                'body': item['body'][:150] + "..." if len(item['body']) > 150 else item['body']
            })

        return news_items
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        # Return mock news if API fails
        return generate_mock_news(coin)


@st.cache_data
def generate_mock_news(coin="bitcoin"):
    """
    TODO remove
    Generate mock news if API is not available"""
    news_titles = [
        f"{coin.upper()} Surges Past Key Resistance Levels",
        f"Major Bank Announces {coin.capitalize()} Integration",
        f"New Regulatory Framework for {coin.capitalize()} Announced",
        f"{coin.upper()} Mining Difficulty Reaches All-Time High",
        f"Institutional Investors Add {coin.capitalize()} to Portfolios",
        f"Technical Analysis: {coin.capitalize()} Bull Run Imminent"
    ]

    sources = ["CryptoNews", "CoinDesk", "Bloomberg",
               "Financial Times", "Reuters", "CNBC"]

    news_items = []
    for i in range(6):
        days_ago = np.random.randint(0, 5)
        news_items.append({
            'title': news_titles[i],
            'url': "#",
            'source': sources[i],
            'published_at': datetime.now() - timedelta(days=days_ago),
            'body': f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisi vel consectetur euismod, nisi vel consectetur..."
        })

    return news_items


@st.cache_data
def get_correlations(base_symbol, all_symbols=None):
    """TODO Calculate correlations between cryptocurrencies"""
    if all_symbols is None:
        all_symbols = ["BTC", "ETH", "SOL", "ADA", "XRP",
                       "DOT", "DOGE", "AVAX", "LINK", "MATIC"]

    # Remove base symbol from the list if it's there
    if base_symbol in all_symbols:
        all_symbols.remove(base_symbol)

    # Generate mock correlations
    np.random.seed(42)  # For reproducibility
    correlations = []

    for symbol in all_symbols:
        # Generate a correlation value between -0.8 and 0.95
        if symbol in ["ETH", "SOL", "BNB"]:  # Usually positively correlated with BTC
            corr = np.random.uniform(0.6, 0.95)
            trend = "positive"
        elif symbol in ["XRP", "DOGE", "SHIB"]:  # Can be negatively correlated
            corr = np.random.uniform(-0.8, -0.2)
            trend = "negative"
        else:
            corr = np.random.uniform(-0.5, 0.8)
            trend = "positive" if corr > 0 else "negative"

        correlations.append({
            "name": symbol,
            "value": corr,
            "trend": trend
        })

    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x["value"]), reverse=True)

    return correlations

# ----- Data Processing Functions -----


@st.cache_data
def get_top_30() -> list:

    coins = get_top_30_coins(10)

    cryptos: list = [coin[1] for coin in coins]
    # binance kline format
    cryptos = list(map(lambda x: x.upper(), cryptos))

    return cryptos


def add_features(df):
    """Add technical indicators and features to the dataframe"""
    # Create a copy of the dataframe
    data = df.copy()

    # Add lagged features
    for lag in [1]:
        data[f'close_lag_{lag}'] = data['close'].shift(lag)

    # Add rolling stats
    data['ma7'] = data['close'].rolling(window=7).mean()
    data['ma30'] = data['close'].rolling(window=30).mean()
    data['std7'] = data['close'].rolling(window=7).std()

    # Add price changes
    data['price_change_1d'] = data['close'].pct_change(1)
    data['price_change_7d'] = data['close'].pct_change(7)

    # Calculate RSI (14-period)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values
    data = data.dropna()

    return data


@st.cache_data
def train_forecast_model(df, forecast_horizon=90, input_window=180):
    """Train an XGBoost model for multi-step forecasting"""
    # Prepare features
    target_col = 'close'
    
    df_features = add_features(df)

    # Define target and feature columns
    feature_cols = [col for col in df_features.columns if col != target_col]

    # Prepare X (input) and y (target) data for training
    X, y = [], []

    # For each possible start point
    for i in range(len(df_features) - input_window - forecast_horizon + 1):
        # Input window
        X.append(df_features.iloc[i:i+input_window]
                 [feature_cols].values.flatten())

        # Target: next forecast_horizon closing prices
        y.append(df_features.iloc[i+input_window:i +
                 input_window+forecast_horizon][target_col].values)

    X = np.array(X)
    y = np.array(y)

    # Train model
    base_model = XGBRegressor(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )

    model = MultiOutputRegressor(base_model)
    model.fit(X, y)

    # Get latest input window for forecasting
    latest_input = df_features.iloc[-input_window:
                                    ][feature_cols].values.flatten().reshape(1, -1)

    # Make forecast
    forecast = model.predict(latest_input)[0]
    
    # scale forcast values (if needed)
    last_historic_price = df['close'].iloc[-1]
    first_forecast_price = forecast[0]
    scaling_factor = last_historic_price / first_forecast_price
    
    if scaling_factor == 0:
        scaling_factor = 1
    elif scaling_factor > 1:
        if scaling_factor > 1.1:
            forecast = forecast * scaling_factor
    elif scaling_factor < 1:
        if scaling_factor < 0.9:
            forecast = forecast * scaling_factor
            
    print("scaling by a factor: ", scaling_factor)      
    
    # Create forecast dates
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted': forecast,
        'lower_bound': forecast * 0.95,  # Simplified confidence interval
        'upper_bound': forecast * 1.05,  # Simplified confidence interval
        # Decreasing confidence over time
        'confidence': np.linspace(0.9, 0.6, forecast_horizon)
    })

    return forecast_df


def get_market_state(data_dict):
    """Analyze market state based on multiple coins"""
    # Get the last 30 days of data for each coin
    changes = {}
    for symbol, df in data_dict.items():
        if len(df) >= 30:
            changes[symbol] = (df['close'].iloc[-1] /
                               df['close'].iloc[-30] - 1) * 100

    # Calculate overall market trend
    avg_change = sum(changes.values()) / len(changes)
    trend = "bullish" if avg_change > 0 else "bearish"

    # Calculate confidence based on consistency among coins
    positive_count = sum(1 for change in changes.values() if change > 0)
    negative_count = len(changes) - positive_count
    confidence = max(positive_count, negative_count) / len(changes)

    # Predict individual coin trends
    major_coins = []
    for symbol, change in changes.items():
        # Simple prediction based on recent trend
        major_coins.append({
            'name': symbol,
            'trend': 'up' if change > 0 else 'down',
            # Higher change = higher confidence
            'probability': min(0.5 + abs(change) / 50, 0.95)
        })

    # Sort by probability
    major_coins.sort(key=lambda x: x['probability'], reverse=True)

    return {
        'trend': trend,
        'confidence': confidence,
        'avg_change': avg_change,
        'major_coins': major_coins[:4]  # Top 4 coins with highest probability
    }


def calculate_kpis(df):
    """Calculate key performance indicators for a cryptocurrency"""
    # Get recent data (last 30 days)
    recent_df = df.iloc[-30:]

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = recent_df['close'].pct_change().dropna()
    volatility = daily_returns.std()

    # Estimate market cap (approximation: close price * average daily volume)
    market_cap = (recent_df['close'].iloc[-1] *
                  recent_df['volume'].mean()) / 1e9  # In billions

    return {
        'volatility': volatility,
        'market_cap': market_cap,
    }


def calculate_buy_recommendation(forecast_df: pd.DataFrame, target_profit):
    """Calculate buy recommendation based on forecast"""
    prices = forecast_df['predicted'].values
    dates = forecast_df['date'].dt.strftime('%Y-%m-%d').values
    confidence = forecast_df['confidence'].values

    best_buy_index = 0
    best_sell_index = 0
    best_roi = 0
    target_met = False

    # Find best buy/sell combination
    for buy_idx in range(len(prices)):
        for sell_idx in range(buy_idx + 1, len(prices)):
            buy_price = prices[buy_idx]
            sell_price = prices[sell_idx]
            roi = (sell_price - buy_price) / buy_price

            if roi > best_roi:
                best_roi = roi
                best_buy_index = buy_idx
                best_sell_index = sell_idx

    # If no good opportunity, use the lowest price point
    if best_buy_index == 0 and best_sell_index == 0:
        best_buy_index = np.argmin(prices)
        best_sell_index = min(best_buy_index + 7, len(prices) - 1)

    buy_date = dates[best_buy_index]
    buy_price = prices[best_buy_index]
    sell_price = prices[best_sell_index]
    buy_confidence = confidence[best_buy_index]

    # target profit met
    profit_pct = forecast_df.loc[best_buy_index:best_sell_index, [
        'predicted']].pct_change(periods=best_sell_index-best_buy_index).to_numpy()[-1] * 100
    target_met = profit_pct >= target_profit

    return {
        'buy_date': buy_date,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'confidence': buy_confidence,
        'target_met': target_met,
        'profit_pct': float(profit_pct)
    }


def calculate_moving_averages(df, windows=[7, 30]):
    """Calculate moving averages for specified windows"""
    ma_data = df.copy()

    for window in windows:
        ma_data[f'ma{window}'] = df['close'].rolling(window=window).mean()

    return ma_data


def calculate_profit_scenarios(df, amount=1, days_ago=30):
    """Calculate profit scenarios"""
    current_price = df['close'].iloc[-1]

    # If we have enough historical data
    if len(df) > days_ago:
        past_price = df['close'].iloc[-days_ago]
        profit = (current_price - past_price) * amount
        return profit
    else:
        return 0


def calculate_investment_profit(df, investment=1000, days_ago=7):
    """Calculate profit from an investment"""
    current_price = df['close'].iloc[-1]

    # If we have enough historical data
    if len(df) > days_ago:
        past_price = df['close'].iloc[-days_ago-1]
        coins_bought = investment / past_price
        current_value = coins_bought * current_price
        return current_value
    else:
        return investment


def load_base_model(coin: str):
    """load pre-trained model"""
    pass


def main():
    # Display header
    st.markdown('<div class="main-header">Cryptocurrency Forecasting Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown('Real-time analysis and predictions for cryptocurrency markets')

    with st.spinner('Getting top 30 market cap coins...'):
        top_market_cap = get_top_30()
    
    # Sidebar for controls
    with st.sidebar:
        st.header('Dashboard Controls')


        # Cryptocurrency selection
        selected_coin = st.selectbox(
            'Select Cryptocurrency',
            options=[coin.rstrip("USDT") for coin in top_market_cap],
            index=0
        )

        # Time interval selection
        time_interval = int(
            st.select_slider(
                'Time Interval (Days)',
                options=[7, 14, 30, 60, 90],
                value=30,
            )
        )

        # Target profit for recommendations
        target_profit = st.slider(
            'Target Profit (%)',
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )

        # Update button
        if st.button('🔄 Refresh Data'):
            st.cache_data.clear()
            st.experimental_rerun()

    # Fetch data for selected coin
    with st.spinner(f'Fetching {selected_coin} data...'):
        coin_data = get_crypto_data(selected_coin, limit=365)

        # Get data for additional coins for market analysis
        all_coins_data = {
            coin.rstrip("USDT"): get_crypto_data(coin.rstrip("USDT"), limit=90) for coin in top_market_cap
        }

    # Filter data based on selected time interval
    filtered_data = coin_data.iloc[-time_interval:]

    # Generate price forecast
    with st.spinner('Generating price forecast...'):
        forecast = train_forecast_model(coin_data)

    # Calculate metrics
    current_price = coin_data['close'].iloc[-1]
    yesterday_price = coin_data['close'].iloc[-2]
    price_change = current_price - yesterday_price
    price_change_pct = (price_change / yesterday_price) * 100

    # Display current price and metrics in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-label">{selected_coin} Current Price</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">${current_price:.2f}</div>', unsafe_allow_html=True)

        if price_change >= 0:
            st.markdown(
                f'<div class="positive">▲ ${price_change:.2f} ({price_change_pct:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="negative">▼ ${abs(price_change):.2f} ({price_change_pct:.2f}%)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Get KPIs
    kpis = calculate_kpis(coin_data)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">30-Day Volatility</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{kpis["volatility"]*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div>24h Volume: ${kpis["market_cap"]:.1f}B</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Get market state
    market_state = get_market_state(all_coins_data)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Overall Market State</div>',
                    unsafe_allow_html=True)

        if market_state['trend'] == 'bullish':
            st.markdown(
                f'<div class="metric-value positive">Bullish</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="metric-value negative">Bearish</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div>Confidence: {market_state["confidence"]*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Main charts
    st.markdown('<div class="sub-header">Price Analysis</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["Price History", "Moving Averages", "Price Forecast"])

    with tab1:
        # Price history chart
        fig = px.line(
            filtered_data.reset_index(),
            x='timestamp',
            y='close',
            title=f'{selected_coin} Price History ({time_interval} Days)',
            labels={'timestamp': 'Date', 'close': 'Price (USD)'}
        )
        fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Calculate moving averages
        ma_data = calculate_moving_averages(filtered_data)

        # Moving averages chart
        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=ma_data.index,
            y=ma_data['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1E40AF', width=2)
        ))

        # Add MA7 line
        fig.add_trace(go.Scatter(
            x=ma_data.index,
            y=ma_data['ma7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='#10B981', width=2)
        ))

        # Add MA30 line
        fig.add_trace(go.Scatter(
            x=ma_data.index,
            y=ma_data['ma30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='#F59E0B', width=2)
        ))

        fig.update_layout(
            title=f'{selected_coin} Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(orientation='h', yanchor='bottom',
                        y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Price forecast chart
        fig = go.Figure()

        # Add historical prices (last 30 days)
        historical = coin_data.iloc[-30:].reset_index()
        fig.add_trace(go.Scatter(
            x=historical['timestamp'],
            y=historical['close'],
            mode='lines',
            name='Historical',
            line=dict(color='#1E40AF', width=2)
        ))

        # Add forecast line
        # print(forecast.loc[forecast.index[:time_interval], 'date'].tail())
        fig.add_trace(go.Scatter(
            x=forecast.loc[forecast.index[:time_interval], 'date'],
            y=forecast.loc[forecast.index[:time_interval], 'predicted'],
            mode='lines',
            name='Forecast',
            line=dict(color='#7C3AED', width=2, dash='dash')
        ))

        # Add confidence interval
        forecast_interval = forecast[:time_interval]
        fig.add_trace(go.Scatter(
            x=forecast_interval['date'].tolist(
            ) + forecast_interval['date'].tolist()[::-1],
            y=forecast_interval['upper_bound'].tolist(
            ) + forecast_interval['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(124, 58, 237, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval'
        ))

        fig.update_layout(
            title=f'{selected_coin} {time_interval}-Day Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend=dict(orientation='h', yanchor='bottom',
                        y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Correlation and recommendations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="sub-header">Correlation Analysis</div>', unsafe_allow_html=True)

        # Get correlations
        correlations = get_correlations(selected_coin)

        # Split into positive and negative correlations
        positive_corr = [
            c for c in correlations if c['trend'] == 'positive'][:4]
        negative_corr = [
            c for c in correlations if c['trend'] == 'negative'][:4]

        # Create two columns for positive and negative correlations
        pos_col, neg_col = st.columns(2)

        with pos_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<h3 style="color: #10B981; font-weight: 600;">Positive Correlation</h3>', unsafe_allow_html=True)

            for coin in positive_corr:
                st.markdown(
                    f'<div class="correlation-item">'
                    f'<span>{coin["name"]}</span>'
                    f'<span style="color: #10B981; font-weight: 600;">{coin["value"]:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

        with neg_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<h3 style="color: #EF4444; font-weight: 600;">Negative Correlation</h3>', unsafe_allow_html=True)

            for coin in negative_corr:
                st.markdown(
                    f'<div class="correlation-item">'
                    f'<span>{coin["name"]}</span>'
                    f'<span style="color: #EF4444; font-weight: 600;">{coin["value"]:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="sub-header">Investment Calculator</div>', unsafe_allow_html=True)

        # Calculate buy recommendation
        buy_rec = calculate_buy_recommendation(forecast, target_profit)

        # Display buy recommendation
        st.markdown(
            '<div class="card" style="background-color: #EEF2FF;">', unsafe_allow_html=True)
        st.markdown(
            '<h3 style="color: #4F46E5; font-weight: 600;">Buy Recommendation</h3>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="correlation-item">'
            f'<span>Best time to buy:</span>'
            f'<span style="font-weight: 600;">{buy_rec["buy_date"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="correlation-item">'
            f'<span>Expected price:</span>'
            f'<span style="font-weight: 600;">${buy_rec["buy_price"]:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="correlation-item">'
            f'<span>Sell target:</span>'
            f'<span style="font-weight: 600; color: {"#10B981" if buy_rec["target_met"] else "#EF4444"}">${buy_rec["sell_price"]:.2f}</span >'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="correlation-item">'
            f'<span>Profit:</span>'
            f'<span style="font-weight: 600; color: {"#10B981" if buy_rec["target_met"] else "#EF4444"}">{buy_rec["profit_pct"]:.2f}%</span >'
            f'</div>',
            unsafe_allow_html=True
        )

        # Confidence progress bar
        st.markdown(
            f'<div class="correlation-item">'
            f'<span>Confidence:</span>'
            f'<span style="font-weight: 600;">{buy_rec["confidence"]*100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.progress(buy_rec["confidence"])

        st.markdown('</div>', unsafe_allow_html=True)

        # What-if scenarios
        st.markdown(
            '<div class="card" style="background-color: #ECFDF5;">', unsafe_allow_html=True)
        st.markdown(
            '<h3 style="color: #059669; font-weight: 600;">What-If Analysis</h3>', unsafe_allow_html=True)

        # Calculate profit scenarios
        profit_one_month = calculate_profit_scenarios(
            coin_data, amount=1, days_ago=30)
        profit_investment = calculate_investment_profit(
            coin_data, investment=1000, days_ago=7)

        st.markdown(
            f'<div class="correlation-item">'
            f'<span>If you bought 1 {selected_coin} a month ago:</span>'
            f'<span style="font-weight: 600; color: {"#10B981" if profit_one_month >= 0 else "#EF4444"}">'
            f'${profit_one_month:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="correlation-item">'
            f'<span>If you invested $1,000 a week ago:</span>'
            f'<span style="font-weight: 600; color: {"#10B981" if profit_investment >= 1000 else "#EF4444"}">'
            f'${profit_investment:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # News and KPIs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Recent News</div>',
                    unsafe_allow_html=True)

        # Get news
        news = get_crypto_news(selected_coin.lower())

        # Display news items
        st.markdown('<div class="card">', unsafe_allow_html=True)

        for item in news:
            st.markdown(
                f'<div style="padding: 10px 0; border-bottom: 1px solid #E5E7EB;">'
                f'<h4 style="margin: 0; font-weight: 600;">{item["title"]}</h4>'
                f'<p style="margin: 5px 0; font-size: 0.9rem; color: #6B7280;">{item["body"]}</p>'
                f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6B7280;">'
                f'<span>{item["source"]}</span>'
                f'<span>{item["published_at"].strftime("%Y-%m-%d")}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="sub-header">Key Performance Indicators</div>', unsafe_allow_html=True)

        # Create KPI metrics
        kpi1, kpi2 = st.columns(2)

        with kpi1:
            st.markdown(
                '<div class="card" style="background-color: #F3F4F6;">', unsafe_allow_html=True)
            st.metric(
                label="Market Cap",
                value=f"{kpis['volatility']*100:.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with kpi2:
            st.markdown(
                '<div class="card" style="background-color: #F3F4F6;">', unsafe_allow_html=True)
            st.metric(
                label="24h Volume",
                value=f"${kpis['market_cap']:.1f}B"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            # st.markdown(
            #     '<div class="card" style="background-color: #F3F4F6;">', unsafe_allow_html=True)
            # st.metric(
            #     label="BTC Dominance",
            #     value=f"{kpis['dominance']:.1f}%"
            # )
            # st.markdown('</div>', unsafe_allow_html=True)

            # st.markdown(
            #     '<div class="card" style="background-color: #F3F4F6;">', unsafe_allow_html=True)
            # st.metric(
            #     label="Volatility (30d)",
            #     value=f"{kpis['volatility']*100:.1f}%"
            # )
            # st.markdown('</div>', unsafe_allow_html=True)

            # Market prediction
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h3 style="font-weight: 600;">Major Coins Trend Prediction</h3>', unsafe_allow_html=True)

        # Display trend predictions in a grid
        # cols = st.columns(2)
        # for i, coin in enumerate(market_state['major_coins']):
        #     col_idx = i % 2
        #     with cols[col_idx]:
        #         colour = "#10B981" if coin['trend'] == 'up' else "#EF4444"
        #         arrow = "↑" if coin['trend'] == 'up' else "↓"

        #         st.markdown(
        #             f'<div style="text-align: center; padding: 10px; background-color: #F9FAFB; border-radius: 8px; margin-bottom: 10px;">'
        #             f'<div style="color: #EF4444; font-weight: 600;">{coin["name"]}</div>'
        #             f'<div style="color: {colour}; font-weight: 600;">{arrow} {coin["probability"]*100:.0f}%</div>'
        #             f'</div>',
        #             unsafe_allow_html=True
        #         )

        # st.markdown('</div>', unsafe_allow_html=True)

        cols = st.container(border=False)
        with cols:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # st.markdown(
            # '<h3 style="font-weight: 600;">Major Coins Trend Prediction</h3>', unsafe_allow_html=True)

            for coin in market_state['major_coins']:
                colour = "#10B981" if coin['trend'] == 'up' else "#EF4444"
                arrow = "↑" if coin['trend'] == 'up' else "↓"
                st.markdown(
                    f'<div class="correlation-item">'
                    f'<span>{coin["name"]}</span>'
                    f'<span style="color: {colour}; font-weight: 600;">{arrow} {coin["probability"]*100:.0f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown('<div class="disclaimer">', unsafe_allow_html=True)
    st.markdown(
        '<h3 style="font-weight: 600; color: #92400E;">Important Disclaimer</h3>'
        '<p style="color: #92400E;">'
        'Cryptocurrency investments are subject to high market risk. Past performance is not indicative of future results. '
        'Forecasts are based on historical data and modeling, and actual results may vary. '
        'Please consult with a financial advisor before making investment decisions.'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
