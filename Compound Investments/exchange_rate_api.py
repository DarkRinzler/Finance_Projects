"""
exchange_rate_api.py
-----------------------------------------------------------------------
Script to fetch currency conversion data for compound_investment_calculator.py

Dependencies
-----------------------------------------------------------------------
- os
- requests

Usage
-----------------------------------------------------------------------
$ python exchange_rate_api.py

Author
-----------------------------------------------------------------------
Riccardo Nicolò Iorio

Date
-----------------------------------------------------------------------
12-04-2024
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# ---------------
# v1.0  12-04-2024  Initial version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
import os                                       # Operating system interface
import requests                                 # Time-related functions
import time                                     # HTTP requests
from typing import Dict, Optional, Any          # Type annotation

# Third-party Libraries
from dotenv import load_dotenv                  # Environment variable management

# -------------------------------------------------------- #
#                      DATA REQUEST
# -------------------------------------------------------- #

# Load the .env file
load_dotenv()

# Get API key from the .env file
api_key = os.getenv('CURRENCY_APY_KEY')

# Openexchangerate URL
url = f'https://openexchangerates.org/api/latest.json?app_id={api_key}&symbols=EUR,GBP,CHF,CAD'

# Cache to store exchange rates
cache: Dict[str, Optional[Any]] = {
    'data': None,
    'timestamp': 0.0  # Set initial timestamp to 0
}

# Cache expiry time in seconds
cache_expiry_time = 3600  # 1 hour

def fetch_exchange_rates() -> None:
    """
        The function fetches exchange rates for the requested currency from an external API and updates the cache

        Arguments:
            No arguments

        Returns:
            None
    """

    global cache                                            # Making the cache a global variable
    try:
        response = requests.get(url)
        response.raise_for_status()                         # Fetch API request status
        data = response.json()
        cache.update({                                      # Update cache with data and data timestamp
            'data': data,
            'timestamp': data.get('timestamp')              # Use API timestamp
        })
    except requests.exceptions.RequestException as error:       # Raise an error for bad HTTP responses
        print(f"Error making API request: {error}")
    except ValueError as value_error:
        print(f"Error parsing response: {value_error}")

def exchange_rate_request(selected_currency: str) -> float:
    """
        Retrieves the exchange rate for the specified currency relative to the Euro (EUR)

        Arguments:
            selected_currency (str) : The currency ticker for which to fetch the exchange rate

        Returns:
            float: The exchange rate between the Euro and the specified currency
    """

    # Global cache containing the selected currency exchange rate
    global cache
    current_time = time.time()

    # Refresh data if the cache is empty or older than the expiry time
    if not cache['data'] or current_time - cache['timestamp'] > cache_expiry_time:
        fetch_exchange_rates()

    # Access the cached data
    data = cache['data']

    # Get EUR base rate and compute the target rate
    eur_rate = data['rates']['EUR']
    if selected_currency == 'USD':
        return 1 / eur_rate

    return data['rates'].get(selected_currency) / eur_rate
