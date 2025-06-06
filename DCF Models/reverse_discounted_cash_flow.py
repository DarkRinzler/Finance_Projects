"""
Script Name: reverse_discounted_cash_flow.py

Description:
This script develops machine learning models to predict how likely a particular person is in developing cardiovascular diseases.
It reads data from a CSV file and trains three models: Linear Regression, Decision Tree, and a Neural Network.
For each model, it computes evaluation metrics including accuracy, precision, and recall.
After selecting the best-performing model, the script analyzes the most important features contributing to the prediction.

Dependencies:
- pandas
- numpy

Usage:
$ python3 reverse_discounted_cash_flow.py

Author: Riccardo Nicolò Iorio
Date:
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import utils
from colorama import Fore, init
from tabulate import tabulate

#

# Set display options
pd.set_option('display.width', 500)  # Increase the display width
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

#Constants
rDCF_years: int = 10
num_simulations: int = int(1e6)

# Initialize colorama
init(autoreset=True)

# Terminal growth and discount rate assumptions
terminal_growth_rates: Dict[int, str] = {          # Terminal growth rates assumptions
        1: 'Minimal Growth',                       # 1% - low growth
        2: 'Moderate Growth',                      # 2% - moderate growth
        3: 'Accelerated Growth',                   # 3% - high growth
        4: 'Exponential Growth'                    # 4% - very high growth
    }

discount_rates: Dict[int, str] = {                # Discount rates assumption
        6: 'Capital Protection',                  # 6% - very low risk
        8: 'Low Volatility',                      # 8% - low risk
        10: 'Moderate Risk',                      # 10% - normal risk
        12: 'Growth Oriented',                    # 12% - high risk
        15: 'Aggressive Growth'                   # 15% -  very high risk
    }

currency_types: Dict[str, str] = {                # Available currency from API
        'USD': 'United States Dollar',            # US dollar
        'EUR': 'Euro',                            # Euro
        'GBP': 'British Pound Sterling',          # British Pound
        'CHF': 'Swiss Franc',                     # Swiss Franc
        'CAD': 'Canadian Dollar'                  # Canadian Dollar
    }

# Prompts for user parameter selection
prompts: Dict[str, str] = {
        'company_stock': 'Enter the stock ticker symbol of the company you want to analyze using the rDCF model: ',               # Company Ticker
        'currency': 'Choose the currency of the company’s stock price from the options provided below: ',                         # Currency of the traded stock
        'stock_price': 'Enter the current stock price of the company: ',                                                          # Current stock price
        'outstanding_shares': 'Enter the number of outstanding shares of the company (in billions, to two decimal places): ',     # Total outstanding shares
        'ltm_fcf': 'Enter the Free Cash Flow of the last twelve months (in billions, to two decimal places): ',                   # Free Cash Flow of the last 12 months
    }

# Define numpy arrays for the discount and terminal growth rates (simple case)
ds_rate = np.array(list(discount_rates.keys()))
tg_rate = np.array(list(terminal_growth_rates.keys()))

# Define reshaped numpy arrays of the discount and terminal growth rates for broadcasting (simple case)
dds_rate = (ds_rate / 100).reshape(1, 1, len(ds_rate), 1)
dtgr_rate = (tg_rate / 100).reshape(1, 1, 1, len(tg_rate))

# Define numpy arrays for the discount and terminal growth rates (extended case)
ext_ds_rate = np.linspace(ds_rate[0], ds_rate[-1], 10, endpoint = True)
ext_tg_rate = np.linspace(tg_rate[0], tg_rate[-1], 7, endpoint = True)

# Define reshaped numpy arrays of the discount and terminal growth rates for broadcasting (extended case)
cds_rate = (ext_ds_rate / 100).reshape(1, 1, len(ext_ds_rate), 1)
ctg_rate = (ext_tg_rate / 100).reshape(1, 1, 1, len(ext_tg_rate))

# Define a dictionary with entries containing discount and terminal growth rate for both simple and extended case
simulation_rates = {
    'simple': {'ds_rates': dds_rate, 'tgr_rates': dtgr_rate},
    'extended': {'ds_rates': cds_rate, 'tgr_rates': ctg_rate}
}

# Time wrapper to measure function execution time
def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = time.perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = time.perf_counter()
        print(f"Time for {func.__name__}: {end_time - start_time:.4f} seconds")

        return result
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

# Function to calculate the projected free cash flows of the company for the time period selected (e.g. 10 years)
#@timer
def projected_free_cash_flow(stock_parameters: Dict[str, Any], growth_rate: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:

    time_period = np.arange(1, rDCF_years + 1).reshape(1, rDCF_years, 1, 1)
    decimal_growth_rate = growth_rate.reshape(-1, 1, 1, 1) / 100
    fcf_t = stock_parameters['ltm_fcf'] * ((1 + decimal_growth_rate) ** time_period)
    return time_period, fcf_t

# Function to calculate the present value of the free cash flows for the time period selected (e.g. 10 years)
#@timer
def present_value_free_cash_flow(discount_rate: np.ndarray, projected_fcf: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    time_period, fcf_t = projected_fcf
    pv_tfcf = fcf_t / ((1 + discount_rate) ** time_period)
    return np.sum(pv_tfcf, axis = 1)

# Function to calculate the discounted terminal value of the company
#@timer
def discounted_terminal_value(discount_rate:np.ndarray, terminal_growth:np.ndarray, projected_fcf: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    _, fcf_t = projected_fcf
    terminal_value_fcf_t = fcf_t[:, -1, :, :] * (1 + terminal_growth[:, -1, :, :]) / (discount_rate[:, -1, :, :] - terminal_growth[:, -1, :, :])
    pv_terminal_value = terminal_value_fcf_t / ((1 + discount_rate[:, -1, :, :]) ** rDCF_years)
    return pv_terminal_value

# Function to calculate the total present value of cash flows and the intrinsic value per share of the company
#@timer
def intrinsic_value_share(stock_parameters: Dict[str, Any], discounted_fcf: np.ndarray, discounted_tv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    total_present_value = discounted_fcf + discounted_tv
    intrinsic_value_per_share = total_present_value / stock_parameters['outstanding_shares']
    return total_present_value, intrinsic_value_per_share

# Function to calculate the intrinsic value per share (adjusted to use random growth rate)
#@timer
def calculate_intrinsic_value(discount_rate:np.ndarray, terminal_growth:np.ndarray, stock_parameters: Dict[str, Any], growth_rate: np.ndarray) -> np.ndarray:

    # Calculate projected FCF for each year
    projections = projected_free_cash_flow(stock_parameters, growth_rate)

    # Calculate discounted FCF
    discounted_fcf = present_value_free_cash_flow(discount_rate, projections)

    # Calculate discounted terminal value (TV)
    discounted_tv = discounted_terminal_value(discount_rate, terminal_growth, projections)

    # Calculate intrinsic value per share
    total_value, intrinsic_value_per_share = intrinsic_value_share(stock_parameters, discounted_fcf, discounted_tv)

    return intrinsic_value_per_share

#@timer
# Monte Carlo simulation to estimate the range of intrinsic values
def monte_carlo_simulation(stock_parameters: Dict[str, Any], discount_rate:np.ndarray, terminal_growth:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # Simulate a random growth rate within the interval [-120, 120]
    simulated_growth_rate = np.random.uniform(-120, 120, num_simulations)

    # Calculate intrinsic value for the simulated growth rate
    intrinsic_value = calculate_intrinsic_value(discount_rate, terminal_growth, stock_parameters, simulated_growth_rate)

    # Error from current market price
    error = np.abs(intrinsic_value - stock_parameters['stock_price'])

    # Track the best growth rate with the minimum error across the number of simulations axis
    min_error_index = np.argmin(error, axis = 0)
    min_error = error[min_error_index].astype(float)
    best_growth_rate = simulated_growth_rate[min_error_index].astype(float)

    return best_growth_rate, min_error

# Creating a DataFrame summary table of the implied growth rates for each value of terminal growth and discount rate
#timer
def implied_growth_rates_pd(stock_parameters: Dict[str, Any], discount_rate:np.ndarray, terminal_growth:np.ndarray, case: str) -> pd.DataFrame:

    # Vectorized computation by applying monte_carlo_simulation across all discount and terminal growth rate pairs
    implied_growth_rates = monte_carlo_simulation(stock_parameters, discount_rate, terminal_growth)[0]

    if case == 'simple':
        # Numpy array for the discount and terminal growth rates (simple case)
        ds_rates_labels = [f"{value:<20} {key}%".rjust(15) for key, value in discount_rates.items()]
        tgr_rates_labels = [f"{value} {key}%" for key,value in terminal_growth_rates.items()]

        # Create DataFrame with index and columns matching the simple case
        growth_rate_dataframe = pd.DataFrame(index = ds_rates_labels, columns = tgr_rates_labels)

        # Assign implied growth rates for all discount and terminal growth rate pairs to a new DataFrame
        growth_rate_dataframe[:] = implied_growth_rates
    else:
        # Numpy array for the discount and terminal growth rates (extended case)
        ext_ds_rates_labels = [f"{rates:.0f}%" for rates in ext_ds_rate]
        ext_tgr_rates_labels = [f"{rates:.1f}%" for rates in ext_tg_rate]

        # Create DataFrame with index and columns matching the extended case
        growth_rate_dataframe = pd.DataFrame(index = ext_ds_rates_labels, columns = ext_tgr_rates_labels)

        # Assign implied growth rates for all discount and terminal growth rate pairs to a new DataFrame for the extended case
        growth_rate_dataframe[:] = pd.DataFrame(implied_growth_rates)

    growth_rate_dataframe.index.name = f"{Fore.GREEN}{stock_parameters['company_stock'].upper()}{Fore.RESET}"

    return growth_rate_dataframe.astype(float)

# Display of DataFrame summary table with formatted output
#@timer
def formatted_output(stock_parameters: Dict[str, Any], df_matrix: pd.DataFrame) -> None:

    # List of columns to format
    formatted_matrix_df = df_matrix.map(lambda x: f"{x:.2f}%")

    # Calculate the width based on the column names only
    width = sum(len(str(col)) for col in terminal_growth_rates.items()) + len(df_matrix.columns) + len(stock_parameters['company_stock']) + 20
    print(f'\n{f' Reverse Discounted Cash Flow - {stock_parameters['company_stock']} ':-^{width}}')

    # Print table of the dataframe
    print(f'\n{tabulate(formatted_matrix_df, headers ='keys', tablefmt ='grid', stralign = 'center', showindex = True)}')

def main() -> None:

    parameters = input_stock_parameters()
    for label, rates in simulation_rates.items():
        dataframe = implied_growth_rates_pd(parameters, rates['ds_rates'], rates['tgr_rates'], label)
        formatted_output(parameters, dataframe)
        if label == 'simple':
            utils.plt_igr(parameters, dataframe)
        else:
            utils.ext_plt_igr(parameters, dataframe)


if __name__ == '__main__':
    main()
