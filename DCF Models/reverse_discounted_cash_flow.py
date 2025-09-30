"""
reverse_discounted_cash_flow.py
-----------------------------------------------------------------------
Estimate the implied growth rate of a publicly traded company
from its stock price using a reverse discounted cash flow (DCF) model,
for positive free cash flow (FCF).

Inputs
-----------------------------------------------------------------------
- Company name : str
- Trading currency : str
- Stock price : float
- Outstanding shares : int
- Last twelve months free cash flow (FCF) : float

Process
-----------------------------------------------------------------------
- Compute implied growth rates over a 10-year horizon
  for combinations of discount rates and terminal growth rates.
- Solve reverse DCF numerically (Monte Carlo approach).
- Generate heatmaps of implied growth rates.

Dependencies
-----------------------------------------------------------------------
- pandas
- numpy
- scipy

Usage
-----------------------------------------------------------------------
$ python reverse_discounted_cash_flow.py

Author
-----------------------------------------------------------------------
Riccardo Nicolò Iorio

Date
-----------------------------------------------------------------------
2025-02-20
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# ---------------
# v1.0  2025-03-21  Initial version
# v1.1  2025-04-13  Second Version
# v1.2  2025-06-17  Third Version
# v1.3  2025-09-23  Fourth Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard Libraries
from typing import Any, Dict, Tuple     # Type annotation

# Third-party Libraries
import numpy as np                      # Numerical computation
import pandas as pd                     # Data manipulation
from scipy import optimize as opt       # Numerical optimisers
from tabulate import tabulate           # Visualisation (heatmap)
from colorama import Fore, init         # Bash coloring

# Local modules
import utils                            # Custom helper function

# -------------------------------------------------------- #
#                  BASH DISPLAY OPTIONS
# -------------------------------------------------------- #

# Initialize colorama
init(autoreset=True)

# Set display options for DataFrame
pd.set_option('display.width', 500)             # Increase the display width
pd.set_option('display.max_columns', None)      # Ensure all columns of the DataFrame are displayed

# -------------------------------------------------------- #
#                       MODEL CONSTANTS
# -------------------------------------------------------- #

# Projection Years
rDCF_years: int = 10

# Number of simulation for running the Monte-Carlo method
num_simulations: int = int(1e6)

# Terminal growth rates assumptions
terminal_growth_rates: Dict[int, str] = {
        1: 'Minimal Growth',                       # 1% - low growth
        2: 'Moderate Growth',                      # 2% - moderate growth
        3: 'Accelerated Growth',                   # 3% - high growth
        4: 'Exponential Growth'                    # 4% - very high growth
    }

# Discount rates assumptions
discount_rates: Dict[int, str] = {
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

# -------------------------------------------------------- #
#                  MODEL ASSUMPTIONS & NOTES
# -------------------------------------------------------- #


# - Projection horizon of the reversed discounted cash flow is fixed at 10 years
# - Terminal value based is calculated according to Gordon Growth Model
# - Discount rates aligned with average market values (5%–15%)
# - Terminal growth rates aligned with average market values (1%–4%)


# -------------------------------------------------------- #
#                    USER DATA PROMPTS
# -------------------------------------------------------- #

# Prompts for user parameter selection
prompts: Dict[str, str] = {
        'company_stock': 'Enter the stock ticker symbol of the company you want to analyze using the rDCF model: ',               # Company Ticker
        'currency': 'Choose the currency of the company’s stock price from the options provided below: ',                         # Currency of the traded stock
        'stock_price': 'Enter the current stock price of the company: ',                                                          # Current stock price
        'outstanding_shares': 'Enter the number of outstanding shares of the company (in billions, to two decimal places): ',     # Total outstanding shares
        'ltm_fcf': 'Enter the Free Cash Flow of the last twelve months (in billions, to two decimal places): ',                   # Free Cash Flow of the last 12 months
    }

# -------------------------------------------------------- #
#                   USER DATA SELECTION
# -------------------------------------------------------- #

#@utils.timer
def input_stock_parameters() -> Dict[str, Any]:
    """
        This function creates a dictionary with all the data selected from the user through the asked prompts, using custom helper
        functions to validate input and print on bash prompt selection

        Arguments:
        No arguments

        Returns:
        parameters -- Dictionary with key-value pairs based on the selection prompts from the user
    """

    parameters: Dict[str, Any] = {}

    # Validation of user selections based on the predefined prompts dictionary
    for key, prompt in prompts.items():
        while key not in parameters:
            try:
                if key == 'company_stock':
                    value = input(prompt).strip()
                    parameters[key] = value
                elif key == 'currency':
                    print(prompts[key])
                    utils.input_choice(key, currency_types, parameters)      # User's currency selection validation
                else:
                    value = float(input(prompt).strip())
                    utils.validation_numeric_input(key, value)               # User's numeric input validation
                    parameters[key] = value
            except ValueError as e:
                print(f"Error: {e}")

    return parameters


# -------------------------------------------------------- #
#              REVERSE DISCOUNTED CASH FLOW MODEL
# -------------------------------------------------------- #

# Define numpy arrays for the discount and terminal growth rates (simple case)
model_ds_rate = np.array(list(discount_rates.keys()))
model_tg_rate = np.array(list(terminal_growth_rates.keys()))

# # Define reshaped numpy arrays of the discount and terminal growth rates for broadcasting (simple case)
# new_ds_rate = (ds_rate / 100).reshape(1, 1, len(ds_rate), 1)
# new_tgr_rate = (tg_rate / 100).reshape(1, 1, 1, len(tg_rate))®

# Define numpy arrays for the discount and terminal growth rates in the range
ds_rate = np.linspace(model_ds_rate[0], model_ds_rate[-1], 10, endpoint = True)
tg_rate = np.linspace(model_tg_rate[0], model_tg_rate[-1], 7, endpoint = True)

# Define reshaped numpy arrays of the discount and terminal growth rates for broadcasting
ndim_ds_rate = (ds_rate / 100).reshape(1, 1, len(ds_rate), 1)
ndim_tg_rate = (tg_rate / 100).reshape(1, 1, 1, len(tg_rate))

# Define a dictionary with entries containing discount and terminal growth rate for both simple and extended case
simulation_rates = {
    'implied_growth_rates': {'ds_rates': ndim_ds_rate, 'tgr_rates': ndim_tg_rate}
}


# Function to calculate the projected free cash flows of the company for the time period selected (e.g. 10 years)
#@utils.timer
def projected_free_cash_flow(stock_parameters: Dict[str, Any], growth_rate: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:

    time_period = np.arange(1, rDCF_years + 1).reshape(1, rDCF_years, 1, 1)
    decimal_growth_rate = growth_rate.reshape(-1, 1, 1, 1) / 100
    fcf_t = stock_parameters['ltm_fcf'] * ((1 + decimal_growth_rate) ** time_period)
    return time_period, fcf_t

# Function to calculate the present value of the free cash flows for the time period selected (e.g. 10 years)
#@utils.timer
def present_value_free_cash_flow(discount_rate: np.ndarray, projected_fcf: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    time_period, fcf_t = projected_fcf
    pv_tfcf = fcf_t / ((1 + discount_rate) ** time_period)
    return np.sum(pv_tfcf, axis = 1)

# Function to calculate the discounted terminal value of the company
#@utils.timer
def discounted_terminal_value(discount_rate:np.ndarray, terminal_growth:np.ndarray, projected_fcf: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    _, fcf_t = projected_fcf
    terminal_value_fcf_t = fcf_t[:, -1, :, :] * (1 + terminal_growth[:, -1, :, :]) / (discount_rate[:, -1, :, :] - terminal_growth[:, -1, :, :])
    pv_terminal_value = terminal_value_fcf_t / ((1 + discount_rate[:, -1, :, :]) ** rDCF_years)
    return pv_terminal_value

# Function to calculate the total present value of cash flows and the intrinsic value per share of the company
#@utils.timer
def intrinsic_value_share(stock_parameters: Dict[str, Any], discounted_fcf: np.ndarray, discounted_tv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    total_present_value = discounted_fcf + discounted_tv
    intrinsic_value_per_share = total_present_value / stock_parameters['outstanding_shares']
    return total_present_value, intrinsic_value_per_share

# Function to calculate the intrinsic value per share (adjusted to use random growth rate)
#@utils.timer
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

#@utils.timer
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
#@utils.timer
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
        ext_ds_rates_labels = [f"{rates:.0f}%" for rates in ds_rate]
        ext_tgr_rates_labels = [f"{rates:.1f}%" for rates in tg_rate]

        # Create DataFrame with index and columns matching the extended case
        growth_rate_dataframe = pd.DataFrame(index = ext_ds_rates_labels, columns = ext_tgr_rates_labels)

        # Assign implied growth rates for all discount and terminal growth rate pairs to a new DataFrame for the extended case
        growth_rate_dataframe[:] = pd.DataFrame(implied_growth_rates)

    growth_rate_dataframe.index.name = f"{Fore.GREEN}{stock_parameters['company_stock'].upper()}{Fore.RESET}"

    return growth_rate_dataframe.astype(float)

# Display of DataFrame summary table with formatted output
#@utils.timer
def formatted_output(stock_parameters: Dict[str, Any], df_matrix: pd.DataFrame) -> None:

    # List of columns to format
    formatted_matrix_df = df_matrix.map(lambda x: f"{x:.2f}%")

    # Calculate the width based on the column names only
    width = sum(len(str(col)) for col in terminal_growth_rates.items()) + len(df_matrix.columns) + len(stock_parameters['company_stock']) + 20
    print(f"\n{f" Reverse Discounted Cash Flow - {stock_parameters['company_stock']} ":-^{width}}")

    # Print table of the dataframe
    print(f'\n{tabulate(formatted_matrix_df, headers ='keys', tablefmt ='grid', stralign = 'center', showindex = True)}')

def main() -> None:

    parameters = input_stock_parameters()
    for label, rates in simulation_rates.items():
        dataframe = implied_growth_rates_pd(parameters, rates['ds_rates'], rates['tgr_rates'], label)
        formatted_output(parameters, dataframe)
        utils.plt_heatmap(parameters, dataframe)


if __name__ == '__main__':
    main()
