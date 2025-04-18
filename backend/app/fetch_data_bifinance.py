import requests
import time
import os
import pandas as pd
from datetime import datetime, timedelta


class Preprocess():
    def __init__(self):
        pass

    def save_to_csv(self, df, filename):
        """
        Save the DataFrame to a CSV file

        Parameters:
        df (pandas.DataFrame): DataFrame to save
        filename (str): Name of the file to save to
        """
        if df is not None:
            df.to_csv(filename)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")

    def missing_values(self, df: pd.DataFrame):
        # First interpolate linearly where possible
        df = df.interpolate(method='time')

        # Then forward-fill any remaining NAs (typically at the beginning of series)
        df = df.fillna(method='ffill')

        # Finally, drop any columns that still have NAs at the beginning
        df = df.dropna(axis=1, how='any')


class BinanceDataFetcher(Preprocess):
    def __init__(self, data_dir="crypto_data"):
        """
        Initialize the BinanceDataFetcher

        Parameters:
        data_dir (str): Directory to save data to
        """
        self.data_dir = data_dir

        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Binance API base URL
        self.base_url = "https://api.binance.com/api/v3"

        # Common trading pairs in Binance
        self.trading_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "SOLUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
            "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "BCHUSDT",
            "ALGOUSDT", "XLMUSDT", "FILUSDT", "VETUSDT", "ETCUSDT",
            "THETAUSDT", "TRXUSDT", "MANAUSDT", "AXSUSDT", "FTMUSDT",
            "EGLDUSDT", "NEARUSDT", "XTZUSDT", "ICPUSDT", "CAKEUSDT"
        ]

    def get_available_pairs(self):
        """
        Return list of available trading pairs

        Returns:
        list: List of available trading pairs
        """
        return self.trading_pairs

    def convert_timestamp_to_datetime(self, timestamp):
        """
        Convert timestamp to datetime

        Parameters:
        timestamp (int): Timestamp in milliseconds

        Returns:
        datetime: Converted datetime
        """
        return datetime.fromtimestamp(timestamp / 1000)

    def fetch_klines(self, symbol, interval="1d", start_time=None, end_time=None, limit=1000):
        """
        Fetch klines (candlestick data) for a trading pair

        Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        start_time (int): Start time in milliseconds
        end_time (int): End time in milliseconds
        limit (int): Maximum number of klines to return (max 1000)

        Returns:
        list: List of klines
        """
        url = f"{self.base_url}/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        if start_time:
            params["startTime"] = start_time

        if end_time:
            params["endTime"] = end_time

        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                print(
                    f"Error: API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def fetch_historical_data(self, symbol, years=5, interval="1d"):
        """
        Fetch historical data for a trading pair over multiple years

        Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        years (int): Number of years of data to fetch
        interval (str): Kline interval

        Returns:
        pandas.DataFrame: DataFrame containing OHLC data
        """
        print(f"Fetching {years} years of {interval} data for {symbol}...")

        # Calculate start and end times
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int(
            (datetime.now() - timedelta(days=years*365)).timestamp() * 1000)

        all_klines = []
        current_start_time = start_time

        # Binance API can return max 1000 candles per request, so we need to make multiple requests
        while current_start_time < end_time:
            klines = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start_time,
                limit=1000
            )

            if not klines or len(klines) == 0:
                break

            all_klines.extend(klines)

            # Update start time for next request
            # The last candle's open time + 1
            current_start_time = klines[-1][0] + 1

            # Respect API rate limits
            time.sleep(1.5)

            print(f"Fetched {len(klines)} candles. Total: {len(all_klines)}")

        if not all_klines:
            print("No data fetched")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert string values to float for price and volume data
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                           'quote_asset_volume', 'taker_buy_base_asset_volume',
                           'taker_buy_quote_asset_volume']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Add symbol column
        df['symbol'] = symbol

        # Extract just OHLC
        df = df[['symbol', 'open', 'high', 'low', 'close', 'volume']]

        # preprocess
        df = self.missing_values(df)

        print(f"Successfully fetched {len(df)} data points for {symbol}")
        return df

    def fetch_multiple_pairs(self, symbols=None, years=5, interval="1d"):
        """
        Fetch historical data for multiple trading pairs

        Parameters:
        symbols (list): List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        years (int): Number of years of data to fetch
        interval (str): Kline interval

        Returns:
        dict: Dictionary of DataFrames containing OHLC data for each trading pair
        """
        if symbols is None:
            # Default to first 5 pairs if none specified
            symbols = self.trading_pairs[:5]

        data = {}

        for symbol in symbols:
            df = self.fetch_historical_data(symbol, years, interval)

            if df is not None:
                data[symbol] = df

                # Save individual data to CSV
                csv_filename = f"{self.data_dir}/{symbol.lower()}_{interval}.csv"
                self.save_to_csv(df, csv_filename)

                # Respect API rate limits between different symbols
                time.sleep(1)

        return data

# Initialize the fetcher
# fetcher = BinanceDataFetcher()

# Fetch BTC/USDT data for 5 years
# btc_data = fetcher.fetch_historical_data("BTCUSDT")

# fetcher.save_to_csv(btc_data, "../backend/data/BTC-past-5y-ohlc.csv")

# To fetch multiple pairs, uncomment and modify:
# Selected trading pairs to fetch
# selected_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
# multi_data = fetcher.fetch_multiple_pairs(selected_pairs)
