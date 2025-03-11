from typing import Dict, Optional, Any
import requests
import time

# API key
api_key = 'b6f7526fc92d4753a5cc6d68f04dd486'

# Openexchangerate URL
url = f'https://openexchangerates.org/api/latest.json?app_id={api_key}&symbols=EUR,GBP,CHF,CAD'

# Cache to store exchange rates
cache: Dict[str, Optional[Any]] = {
    'data': None,
    'timestamp': 0.0  # Set initial timestamp to 0
}

# Cache expiry time in seconds
cache_expiry_time = 3600  # 1 hour

# Fetch exchange rates from the API and update the cache
def fetch_exchange_rates():
    global cache
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        data = response.json()
        cache.update({
            'data': data,
            'timestamp': data.get('timestamp')  # Use API timestamp
        })
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

# Retrieve the exchange rate for the selected currency
def exchange_rate_request(selected_currency: str) -> float:
    global cache
    current_time = time.time()

    # Refresh data if cache is empty or expired
    if not cache['data'] or current_time - cache['timestamp'] > cache_expiry_time:
        fetch_exchange_rates()

    # Access the cached data
    data = cache['data']

    # Calculate the exchange rate
    eur_rate = data['rates']['EUR']
    if selected_currency == 'USD':
        return 1 / eur_rate
    return data['rates'].get(selected_currency) / eur_rate

