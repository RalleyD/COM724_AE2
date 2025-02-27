import urllib.error
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from requests.exceptions import HTTPError, RequestException
import sys
import os

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", 'data')
os.makedirs(data_dir, exist_ok=True)

URL = "https://api.coingecko.com/api/v3"
CSV = f"{data_dir}/top_30_cryptos_past_year.csv"


def _request_data(url, params):
    headers = {"x-cg-demo-api-key": "CG-ZdHmm9QY3wvHARGokUmkZZT9"}
    # request the data
    response = requests.get(url=url, params=params, headers=headers)
    # raise any HTTP status errors
    response.raise_for_status()
    # get the response data
    data = response.json()
    return data


def _handle_req_err(coin_id, e):
    print(f"Problem obtaining historical data for {coin_id=}: {e}")
    return None


def get_top_30_coins(limit=30):
    """
    fetch the top 30 cryptos by market cap

    Args:
        limit (int): limit the specified number of cryptos to fetch

    Returns:
        list of coin IDs
    """
    # define the request URL
    url = f"{URL}/coins/markets"
    # define the API params
    params = {
        "vs_currency": "USD",
        "order": "market_cap",
        "per_page": limit,
        "page": 1,
        "sparkline": False
    }
    # try request
    try:
        data = _request_data(url, params)
        # extract the coin ids
        coin_ids = [(coin['id'], coin['symbol']) for coin in data]
        # print number of coins obtained
        print("successfully obtained top %d coins by market cap" % len(coin_ids))
        return coin_ids
    except RequestException as e:
        return _handle_req_err("", e)
    except HTTPError as e:
        return _handle_req_err("", e)


def get_historical_data(coin_id: str, days=365):
    """
    Get the historical data for a specific coin

    Args:
        coin_id(str): the coin ID returned from the API
        days(int): specified days to go back, default 365
    Returns:
        pandas.DataFrame: of the historical price data.
    """
    # define the URL
    # url = f"{URL}/coins/{coin_id}/market_chart"
    url = f"{URL}/coins/{coin_id}/ohlc"
    # define the params
    params = {
        "vs_currency": "usd",
        "days": days,
        # "interval": "daily"
    }

    # try - request the data
    try:
        data = _request_data(url, params)
        # return None if the price data is not there
        # prices = data.get('prices', [])
        prices = [[d[0], d[4]] for d in data]
        if not prices:
            return None
        # convert the price data to a DataFrame
        prices_df = pd.DataFrame(prices,
                                 columns=['timestamp', f'{coin_id}_price'])
        # convert the timestamp to a datetime object
        prices_df['timestamp'] = pd.to_datetime(
            prices_df["timestamp"], unit='ms')

        prices_df.set_index("timestamp", inplace=True)

        # print success message
        print("successfully obtained %d days of data for coin %s" %
              (len(prices_df), coin_id))
        # return the df
        return prices_df
    # handle exceptions
    except RequestException as e:
        return _handle_req_err(coin_id, e)
    except HTTPError as e:
        return _handle_req_err(coin_id, e)


def fecth_historical_coin_data(coin_limit=30, days_limit=365):
    """
    """
    # get coin IDs
    coin_ids: list = get_top_30_coins()
    # init dataframe to store all data
    all_data: pd.DataFrame = None
    # fetch historical data for each coin
    for coin_id in coin_ids:
        coin_id = coin_id[0]
        coin_data = get_historical_data(coin_id, days_limit)
        if coin_data is not None:
            # build the data
            if all_data is None:
                all_data = coin_data
            else:
                all_data = all_data.join(coin_data, on='timestamp')

        # add a sleep to reduce frequency of API requests within a restricted time frame
        # CoinGecko demo has a limit of 30 calls per minute
        time.sleep(0.5)
    if all_data is not None:
        # forward fill, then back fill any missing data
        all_data = all_data.ffill()
        all_data = all_data.bfill()
        print(
            f"Successfuly obtained {all_data.shape[1]} coins for the last {all_data.shape[0]} days")

    return all_data


if __name__ == '__main__':
    coin_data: pd.DataFrame = fecth_historical_coin_data(
        coin_limit=30, days_limit=365)
    if coin_data is not None:
        coin_data.to_csv(CSV,
                         sep=',', encoding='utf8')
        print("historical data saved to %s" % CSV)
