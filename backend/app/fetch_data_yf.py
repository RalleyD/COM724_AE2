import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from requests.exceptions import HTTPError, RequestException
import yfinance as yf
from .fetch_data_coingecko import get_top_30_coins
import os

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", 'data')
os.makedirs(data_dir, exist_ok=True)

CSV = f"{data_dir}/top_30_cryptos_past_year.csv"


def get_historical_data(coins: list, end_date: datetime = None, days=365, output_file=None):
    """
    Get the historical data for a specific coin

    Args:
        coin_id(str): the coin ID returned from the API
        days(int): specified days to go back, default 365
    Returns:
        pandas.DataFrame: of the historical price data.
    """
    cryptos: list = [coin[1] for coin in coins]

    cryptos = list(map(lambda x: x.upper() + '-USD', cryptos))

    print("cryptos: \n ", cryptos)

    # Initialize an empty DataFrame to hold all the data
    data: pd.DataFrame = None
    if not end_date:
        # Define the date range for the past year
        end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days-1))
    print(f"getting data for time range. {start_date} -> {end_date}")
    data = yf.download(cryptos, start=start_date, end=end_date)

    if data.empty:
        return None

    df = data.loc[:, ['Close']]
    df.dropna(inplace=True, axis=1)
    df = df.droplevel('Price', axis=1)

    if output_file:
        output_file = f"{data_dir}/{output_file}"
    else:
        output_file = CSV
    df.to_csv(output_file, sep=',', encoding='utf8')

    print("#----------------#")
    print(df.info(verbose=True))
    print("#----------------#")
    print("successfully obtained %d days of data for coin %s" %
          (len(df), str(df.columns)))
    print("#----------------#")

    return data.loc[:, ['Close']]


def fetch_historical_coin_data(coin_limit=30, days_limit=365, end_date: datetime = None, output_file=None):
    coins = get_top_30_coins(coin_limit)
    get_historical_data(coins, end_date, days_limit, output_file)


def plot_price_trends(data: pd.DataFrame, top_n=5):
    """
    Plot price trends for the top N cryptocurrencies

    Args:
        data (pandas.DataFrame): DataFrame containing historical price data
        top_n (int): Number of top cryptocurrencies to plot
    """
    if data is None:
        print("No data available to plot")
        return

    # Calculate average price for each cryptocurrency
    avg_prices = data.mean().to_dict()

    # Get top N cryptocurrencies by average price
    top_coins = sorted(avg_prices.items(),
                       key=lambda x: x[1], reverse=True)[:top_n]
    top_coin_cols = [coin[0] for coin in top_coins]

    # Plot
    plt.figure(figsize=(15, 8))
    for col in top_coin_cols:
        coin_name = col.split('-')[0].capitalize()
        plt.plot(data.index, data[col], label=coin_name)

        plt.plot(data.index, data[col].rolling(
            window=7).mean(), linestyle='--', label=f"{coin_name} 7-day moving avg")

    plt.title(f'Close price for Top {top_n} Cryptocurrencies')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    crypto_data = fetch_historical_coin_data()
    if crypto_data is not None:
        plot_price_trends(crypto_data)
