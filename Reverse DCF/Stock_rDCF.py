import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, init

#Constants
rDCF_years: int = 10
num_simulations: int = int(5e5)

# Initialize colorama
init(autoreset=True)


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = time.perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = time.perf_counter()
        print(f"Time for {func.__name__}: {end_time - start_time:.4f} seconds")

        return result
    return wrapper

# Parameters needed for calculations
#@timer
def input_stock_parameters() -> Dict[str, Any]:

    def input_choice(key_prompt:str, stock_dict : Dict[Any, Any], stock_param: Dict[str, Any]) -> None:
        stock_keys = list(stock_dict.keys())
        for i, (sel_key, sel_type) in enumerate(stock_dict.items(), start=1):
            if stock_dict == currency_types:
                print(f'{i}. {Fore.GREEN}{sel_key} {Fore.RESET}({sel_type})')
            else:
                print(f'{i}. {Fore.GREEN}{sel_key:2d}%{Fore.RESET} - {sel_type}')

        while key_prompt not in stock_param:
            try:
                choice = int(input(f'Please enter the number corresponding to your selected {key_prompt}: ').strip())
                if 1 <= choice <= len(stock_keys):
                    stock_param[key_prompt] = stock_keys[choice - 1]
                else:
                    raise ValueError('Invalid selection. Please choose a valid option.')
            except ValueError as e:
                print(f'Error: {e}')

    def validation_numeric_input(key_prompt: str, value_prompt) -> None:
        if key_prompt == 'stock price' and value_prompt <= 0:
            raise ValueError('The stock price of the company can not be less or equal to 0')
        if key_prompt == 'outstanding shares' and value_prompt <= 0:
            raise ValueError('The number of outstanding shares can not be less or equal to 0')
        if key_prompt == 'LTM FCF' and value_prompt <= 0:
            raise ValueError('The Free Cash Flow of the last twelve months can not be less or equal to 0')

    prompts: Dict[str, str] = {
        'company stock': 'Enter the stock ticker symbol of the company you want to analyze using the rDCF model: ',               # Company Ticker
        'currency': 'Choose the currency of the company’s stock price from the options provided below: ',                         # Currency of the traded stock
        'stock price': 'Enter the current stock price of the company: ',                                                          # Current stock price
        'outstanding shares': 'Enter the number of outstanding shares of the company (in billions, to two decimal places): ',     # Total oustanding shares
        'LTM FCF': 'Enter the Free Cash Flow of the last twelve months (in billions, to two decimal places): ',                   # Free Cash Flow of the last 12 months
        'terminal growth rate': 'Select the desired growth rate to apply in the model from the options below: ',                  # Terminal growth rate assumption
        'discount rate': 'Select the desired discount rate to apply in the model from the options below: '                        # Discount rate assumption
    }

    currency_types: Dict[str, str] = {
        'USD': 'United States Dollar',
        'EUR': 'Euro',
        'GBP': 'British Pound Sterling',
        'CHF': 'Swiss Franc',
        'CAD': 'Canadian Dollar'
    }

    terminal_growth_rates: Dict[int, str] = {          # Terminal growth rates assumptions
        1: 'Minimal Growth',                           # 1% - low growth
        2: 'Moderate Growth',                          # 2% - moderate growth
        3: 'Accelerated Growth',                       # 3% - high growth
        4: 'Exponential Growth'                        # 4% - very high growth
    }

    discount_rates: Dict[int, str] = {                # Discount rates assumption
        6: 'Capital Preservation',                    # 6% - very low risk
        8: 'Low Volatility',                          # 8% - low risk
        10: 'Moderate Risk',                          # 10% - normal risk
        12: 'Growth Oriented',                        # 12% - high risk
        15: 'Aggressive Growth'                       # 15% -  very high risk
    }

    user_parameters: Dict[str, Dict[Any, str]] = {
        'currency': currency_types,
        'terminal growth rate': terminal_growth_rates,
        'discount rate': discount_rates
    }

    parameters: Dict[str, Any] = {}

    # Inputs validation for remaining prompts
    for key, prompt in prompts.items():
        while key not in parameters:
            try:
                if key in ['currency', 'terminal growth rate', 'discount rate']:
                    print(prompts[key])
                    input_choice(key, user_parameters[key], parameters)
                elif key == 'company stock':
                    value = input(prompt).strip()
                    parameters[key] = value
                else:
                    value = float(input(prompt).strip())
                    validation_numeric_input(key, value)
                    parameters[key] = value
            except ValueError as e:
                print(f'Error: {e}')
    return parameters

# Function to calculate the projected cash flows of the company for the time period selected
#@timer
def projected_FCF(stock_parameters: Dict[str, Any], growth_rate: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:

    time_periods = np.arange(1, rDCF_years + 1).reshape(1, -1)
    decimal_growth_rate = growth_rate.reshape(-1, 1) / 100
    tFCF = stock_parameters['LTM FCF'] * ((1 + decimal_growth_rate) ** time_periods)
    return time_periods, tFCF

# Function to calculate the present value of cash flows for the time period selected
#@timer
def present_value_FCF(stock_parameters: Dict[str, Any], projected_FCF: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    time_periods, tFCF = projected_FCF
    discount_rate = np.full((num_simulations, 1), stock_parameters['discount rate'] / 100)
    pv_tFCF = tFCF / ((1 + discount_rate) ** time_periods)
    return np.sum(pv_tFCF, axis=1)

# Function to calculate the discounted terminal value of the company
#@timer
def discounted_terminal_value(stock_parameters: Dict[str, Any], projected_FCF: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    _, tFCF = projected_FCF
    terminal_growth = stock_parameters['terminal growth rate'] / 100
    discount_rate = stock_parameters['discount rate'] / 100
    terminal_value_tFCF = tFCF[:, -1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal_value = terminal_value_tFCF / ((1 + discount_rate) ** rDCF_years)
    return pv_terminal_value

# Function to calculate the total present value of cash flows and the intrinsic value per share of the company
#@timer
def intrinsic_value_share(stock_parameters: Dict[str, Any], discounted_FCF: np.ndarray, discounted_TV: np.ndarray) -> Tuple[float, float]:

    total_present_value = discounted_FCF + discounted_TV
    intrinsic_value_per_share = total_present_value / stock_parameters['outstanding shares']
    return total_present_value, intrinsic_value_per_share

# Function to calculate the intrinsic value per share (adjusted to use random growth rate)
#@timer
def calculate_intrinsic_value(stock_parameters: Dict[str, Any], growth_rate: np.ndarray) -> np.ndarray:

    # Calculate projected FCF for each year
    projections = projected_FCF(stock_parameters, growth_rate)

    # Calculate discounted FCF
    discounted_FCF = present_value_FCF(stock_parameters, projections)

    # Calculate discounted terminal value (TV)
    discounted_TV = discounted_terminal_value(stock_parameters, projections)

    # Calculate intrinsic value per share
    total_value, intrinsic_value_per_share = intrinsic_value_share(stock_parameters, discounted_FCF, discounted_TV)

    return intrinsic_value_per_share

#@timer
# Monte Carlo simulation to estimate the range of intrinsic values
def monte_carlo_simulation(stock_parameters: Dict[str, Any]) -> Tuple[float, float]:

    # Simulate a random growth rate within the specified range
    simulated_growth_rate = np.random.uniform(0, 100, num_simulations)

    # Calculate intrinsic value for the simulated growth rate
    intrinsic_value = calculate_intrinsic_value(stock_parameters, simulated_growth_rate)

    # Error from current market price
    error = np.abs(intrinsic_value - stock_parameters['stock price'])

    # Track the best growth rate with the minimum error
    min_error_index = np.argmin(error)
    min_error = error[min_error_index]
    best_growth_rate = simulated_growth_rate[min_error_index]

    return best_growth_rate, min_error

def main() -> None:

    parameters = input_stock_parameters()

    # Run the Monte Carlo simulation
    implied_growth_rate, _ = monte_carlo_simulation(parameters)
    print(f'The implied growth rate of the company is: {Fore.GREEN}{implied_growth_rate:.2f}%')

if __name__ == '__main__':
    main()