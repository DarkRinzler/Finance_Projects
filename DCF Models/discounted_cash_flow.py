import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, init

# Set display options
pd.set_option('display.width', 500)  # Increase the display width
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

#Constants
rDCF_years: int = 10            # Years for free cash flow projection
r_equity: float = 4.41          # Cost of equity

# Initialize colorama
init(autoreset=True)

# Currency type for user selection
currency_types: Dict[str, str] = {                # Available currencies from API
    'USD': 'United States Dollar',                # US Dollar
    'EUR': 'Euro',                                # Euro
    'GBP': 'British Pound Sterling',              # British Pound
    'CHF': 'Swiss Franc',                         # Swiss Franc
    'CAD': 'Canadian Dollar',                     # Canadian Dollar
}

# Prompts for user selection
prompts: Dict[str, str] = {
    'company_stock': 'Enter the stock ticker symbol of the company you want to analyze using the DCF model: ',                # Company ticker
    'currency': 'Choose the currency of the company’s stock price from the options provided below: ',                         # Currency of the traded stock
    'stock_price': 'Enter the current stock price of the company: ',                                                          # Current stock price
    'interest_expense': 'Enter the interest expense of the company (in billions, to two decimal places): ',                   # Interest expense
    'lt_debt': 'Enter the long-term debt of the company (in billions, to two decimal places): ',                              # Long-term debt
    'cp_lt_debt': 'Enter the current portion of long-term debt of the company (in billions, to two decimal places): ',        # Current portion of long-term debt
    'st_borrow': 'Enter the short-term borrowing of the company (in billions, to two decimal places): ',                      # Short-term borrowing
    'corporate_tax': 'Enter the effective tax rate of the company (%): ',                                                     # Effective tax rate
    'outstanding_shares': 'Enter the number of outstanding shares of the company (in billions, to two decimal places): ',     # Total outstanding shares
    'ltm_fcf': 'Enter the Free Cash Flow of the last twelve months (in billions, to two decimal places): ',                   # Free Cash Flow of the last 12 months
    '5y_fcf': 'Enter the Free Cash flow of 5 years ago (in billions, to two decimal places): '                                # Free Cash Flow of 5 years ago
}

# Time wrapper to measure function execution time
def timer(func:Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        results: Any = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Time for {func.__name__}: {end_time - start_time:.4f} seconds")

        return results
    return wrapper

# Dictionary of user input parameters (e.g. {'company stock': 'Apple', 'currency': 'Euro', 'stock price': 200, 'outstanding shares': 1.2, 'LTM FCF': 3.4})
#@timer
def input_stock_parameters() -> Dict[str, Any]:

    def input_choice(key_prompt:str, curr_dict : Dict[Any, Any], param: Dict[str, Any]) -> None:
        curr_keys = list(curr_dict.keys())
        for i, (sel_key, sel_type) in enumerate(curr_dict.items(), start=1):
            print(f'{i}. {Fore.GREEN}{sel_key} {Fore.RESET}({sel_type})')

        while key_prompt not in param:
            try:
                choice = int(input(f"Please enter the number corresponding to your selected {key_prompt}: ").strip())
                if 1 <= choice <= len(curr_keys):
                    param[key_prompt] = curr_keys[choice - 1]
                else:
                    raise ValueError("Invalid selection. Please choose a valid option.")
            except ValueError as err:
                print(f"Error: {err}")

    def validation_numeric_input(key_prompt: str, value_prompt) -> None:
        if key_prompt == 'stock_price' and value_prompt <= 0:
            raise ValueError("The stock price of the company can not be less or equal to 0")
        if key_prompt == 'outstanding_shares' and value_prompt <= 0:
            raise ValueError("The number of outstanding shares can not be less or equal to 0")
        if key_prompt == 'ltm_fcf' and value_prompt <= 0:
            raise ValueError("The Free Cash Flow of the last twelve months can not be less or equal to 0")

    parameters: Dict[str, Any] = {}

    # Inputs validation for remaining prompts, namely company stock ticker,
    for key, prompt in prompts.items():
        while key not in parameters:
            try:
                if key == 'company_stock':
                    value = input(prompt).strip()
                    parameters[key] = value
                elif key == 'currency':
                    print(prompts[key])
                    input_choice(key, currency_types, parameters)
                else:
                    value = float(input(prompt).strip())
                    validation_numeric_input(key, value)
                    parameters[key] = value
            except ValueError as e:
                print(f"Error: {e}")
    return parameters

# Function to calculate the historical average growth rate of the company from the free cash flow values of 5 years ago and of the last twelve months
#@timer
def avg_growth_rate(stock_parameters: Dict[str, Any]) -> float:

    annual_growth_rate = (stock_parameters['ltm_fcf'] / stock_parameters['5y_fcf']) ** (1 / 5) - 1
    return annual_growth_rate

# Function to calculate the projected free cash flow of the company based on the period selected (e.g. 10 years)
#@timer
def projected_free_cash_flow(stock_parameters: Dict[str, Any], compound_annual_growth_rate:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    growth_rate = compound_annual_growth_rate
    time_period = np.arange(1, rDCF_years + 1).reshape(1, -1)
    fcf_t = stock_parameters['ltm_fcf'] * (1 + growth_rate) ** time_period
    return time_period, fcf_t

# Function to calculate the discount rate using the weighted averaged cost of capital (WACC)
# @timer
def discount_rate(stock_parameters: Dict[str, Any]) -> float:

    # Market capitalisation
    t_equity = stock_parameters['share_price'] * stock_parameters['outstanding_shares']

    # Market debt
    t_debt = stock_parameters['lt_debt'] + stock_parameters['cp_lt_debt'] + stock_parameters['st_borrow']

    # Cost of debt
    r_debt = (stock_parameters['interest_expense'] / t_debt) * (1 - stock_parameters['corporate_tax'])

    # Calculating the wacc
    wacc = (t_equity / (t_equity + t_debt)) * r_equity + (t_debt / (t_equity + t_debt)) * r_debt
    return wacc

# Function to calculate
# @timer
# def present_value_free_cash_flow
#
#
# # @timer
# def discounted_terminal_value
#
#
# # @timer
#def intrinsic_value_share

def main() -> None:

    print(input_stock_parameters())


if __name__ == "__main__":
    main()