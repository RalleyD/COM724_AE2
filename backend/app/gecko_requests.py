from flask import Flask, jsonify
import requests
import pandas as pd

app = Flask(__name__)

@app.route('/api/top30')
def get_top_30():
    response = requests.get('https://api.coingecko.com/api/v3/coins/markets', params={
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 30,
        'page': 1,
        'sparkline': False
    })
    data = response.json()
    df = pd.DataFrame(data)
    # Perform additional data processing here
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
