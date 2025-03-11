import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, init
from tabulate import tabulate

# Set display options
pd.set_option('display.width', 500)  # Increase the display width
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed

#Constants
rDCF_years: int = 10
num_simulations: int = int(5e4)

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
            except ValueError as err:
                print(f'Error: {err}')

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
        'outstanding shares': 'Enter the number of outstanding shares of the company (in billions, to two decimal places): ',     # Total outstanding shares
        'LTM FCF': 'Enter the Free Cash Flow of the last twelve months (in billions, to two decimal places): ',                   # Free Cash Flow of the last 12 months
    }

    currency_types: Dict[str, str] = {
        'USD': 'United States Dollar',
        'EUR': 'Euro',
        'GBP': 'British Pound Sterling',
        'CHF': 'Swiss Franc',
        'CAD': 'Canadian Dollar'
    }

    parameters: Dict[str, Any] = {}

    # Inputs validation for remaining prompts
    for key, prompt in prompts.items():
        while key not in parameters:
            try:
                if key == 'company stock':
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
                print(f'Error: {e}')
    return parameters

# Function to calculate the projected cash flows of the company for the time period selected
#@timer
def projected_free_cash_flow(stock_parameters: Dict[str, Any], growth_rate: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:

    time_periods = np.arange(1, rDCF_years + 1).reshape(1, -1)
    decimal_growth_rate = growth_rate.reshape(-1, 1) / 100
    tfcf = stock_parameters['LTM FCF'] * ((1 + decimal_growth_rate) ** time_periods)
    return time_periods, tfcf

# Function to calculate the present value of cash flows for the time period selected
#@timer
def present_value_free_cash_flow(ds_rate: float, projected_fcf: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    time_periods, tfcf = projected_fcf
    discount_rate = np.full((num_simulations, 1), ds_rate / 100)
    pv_tfcf = tfcf / ((1 + discount_rate) ** time_periods)
    return np.sum(pv_tfcf, axis=1)

# Function to calculate the discounted terminal value of the company
#@timer
def discounted_terminal_value(ds_rate:float, tgr_rate:float, total_projected_free_cash_flow: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:

    _, tfcf = total_projected_free_cash_flow
    terminal_growth = tgr_rate / 100
    discount_rate = ds_rate / 100
    terminal_value_tfcf = tfcf[:, -1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal_value = terminal_value_tfcf / ((1 + discount_rate) ** rDCF_years)
    return pv_terminal_value

# Function to calculate the total present value of cash flows and the intrinsic value per share of the company
#@timer
def intrinsic_value_share(stock_parameters: Dict[str, Any], discounted_fcf: np.ndarray, discounted_tv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    total_present_value = discounted_fcf + discounted_tv
    intrinsic_value_per_share = total_present_value / stock_parameters['outstanding shares']
    return total_present_value, intrinsic_value_per_share

# Function to calculate the intrinsic value per share (adjusted to use random growth rate)
#@timer
def calculate_intrinsic_value(stock_parameters: Dict[str, Any], growth_rate: np.ndarray, ds_rate:float, tgr_rate:float) -> np.ndarray:

    # Calculate projected FCF for each year
    projections = projected_free_cash_flow(stock_parameters, growth_rate)

    # Calculate discounted FCF
    discounted_fcf = present_value_free_cash_flow(ds_rate, projections)

    # Calculate discounted terminal value (TV)
    discounted_tv = discounted_terminal_value(ds_rate, tgr_rate, projections)

    # Calculate intrinsic value per share
    total_value, intrinsic_value_per_share = intrinsic_value_share(stock_parameters, discounted_fcf, discounted_tv)

    return intrinsic_value_per_share

#@timer
# Monte Carlo simulation to estimate the range of intrinsic values
def monte_carlo_simulation(stock_parameters: Dict[str, Any], ds_rate: float, tgr_rate: float) -> Tuple[float, float]:

    # Simulate a random growth rate within the specified range
    simulated_growth_rate = np.random.uniform(-100, 100, num_simulations)

    # Calculate intrinsic value for the simulated growth rate
    intrinsic_value = calculate_intrinsic_value(stock_parameters, simulated_growth_rate, ds_rate, tgr_rate)

    # Error from current market price
    error = np.abs(intrinsic_value - stock_parameters['stock price'])

    # Track the best growth rate with the minimum error
    min_error_index = np.argmin(error)
    min_error = error[min_error_index]
    best_growth_rate = simulated_growth_rate[min_error_index]

    return float(best_growth_rate), float(min_error)

# Creating a DataFrame summary table of the implied growth rates for each value of terminal growth and discount rate
#@timer
def implied_growth_rates_matrix(stock_parameters: Dict[str, Any]) -> pd.DataFrame:

    # Numpy array for the discount and terminal growth rates
    discount_rates_labels = [f'{value:<20} {key}%'.rjust(15) for key, value in discount_rates.items()]
    terminal_growth_rates_labels = [f'{value} {key}%' for key,value in terminal_growth_rates.items()]

    growth_rate_dataframe = pd.DataFrame(index=discount_rates_labels, columns=terminal_growth_rates_labels)

    # Convert discount rates and terminal growth rates to arrays
    discount_rate_keys = np.array(list(discount_rates.keys()))
    terminal_growth_rate_keys = np.array(list(terminal_growth_rates.keys()))

    # Create a meshgrid of all combinations (broadcasting)
    dr_grid, tgr_grid = np.meshgrid(discount_rate_keys, terminal_growth_rate_keys, indexing='ij')

    # Pre-allocate memory for performance
    implied_growth_rates = np.empty_like(dr_grid, dtype=float)

    # Vectorized computation: Apply monte_carlo_simulation across all rate pairs
    for i in range(dr_grid.shape[0]):
        for j in range(tgr_grid.shape[1]):
            implied_growth_rates[i, j] = monte_carlo_simulation(stock_parameters, float(dr_grid[i, j]), float(tgr_grid[i, j]))[0]

    growth_rate_dataframe[:] = implied_growth_rates

    growth_rate_dataframe.index.name = f'{Fore.GREEN}{stock_parameters['company stock'].upper()}{Fore.RESET}'

    return growth_rate_dataframe.astype(float)

# Display of DataFrame summary table with formatted output
#@timer
def formatted_output(stock_parameters: Dict[str, Any], df_matrix: pd.DataFrame) -> None:

    # List of columns to format
    formatted_matrix_df = df_matrix.map(lambda x: f'{x:,.2f}%')

    # Calculate the width based on the column names only
    width = sum(len(str(col)) for col in terminal_growth_rates.items()) + len(df_matrix.columns) + len(stock_parameters['company stock']) + 20
    print(f'\n{f' Reverse Discounted Cash Flow - {stock_parameters['company stock']} ':-^{width}}')

    # Print table of the dataframe
    print(f'\n{tabulate(formatted_matrix_df, headers='keys', tablefmt='grid', colalign=['center'] * len(df_matrix.columns), showindex=True)}')

# Implied growth rates as a heatmap
#@timer
def plot_implied_growth_rates(stock_parameters: Dict[str, Any], df_matrix: pd.DataFrame) ->None:

    df_matrix.index.name = None
    min_val = math.floor(df_matrix.min().min() / 5) * 5
    max_val = math.ceil(df_matrix.max().max() / 5) * 5

    plt.figure(figsize=(16, 6))

    sns.heatmap(df_matrix, vmin=min_val, vmax=max_val, cmap='Greens', annot=True, fmt='.2f', linewidths=0.5, square=False)

    # Customize tick label fonts
    for tick_label in plt.gca().get_xticklabels():
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')

    for tick_label in plt.gca().get_yticklabels():
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')

    # Add the % sign to the annotations after formatting
    for text in plt.gca().texts:
        text.set_text(f'{text.get_text()}%')
        text.set_fontweight('bold')

    # Apply plot label and title
    plt.title(f'Implied Growth Rates - {stock_parameters['company stock']}', fontsize=14, fontweight='bold', color='black', pad=15)
    #plt.xlabel('Terminal Growth Rate', fontsize= 12, fontweight='bold', color='black', labelpad=10)
    #plt.ylabel('Risk Profiles', fontsize= 12, fontweight='bold', color='black', labelpad=10)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig('Implied_growth_rates_HM.jpg', dpi=500)
    plt.close()



def main() -> None:

    parameters = input_stock_parameters()
    matrix = implied_growth_rates_matrix(parameters)
    formatted_output(parameters, matrix)
    plot_implied_growth_rates(parameters, matrix)

if __name__ == '__main__':
    main()
'''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import cm

    # Sample DataFrame
    data = {
        'Year': [5, 10, 15, 20, 25, 30, 35],
        'Total Cost (USD)': [29209.69, 47987.35, 66765.01, 85542.66, 104320.32, 123097.98, 141875.64],
        'Total Return (USD)': [39700.67, 85148.77, 155720.21, 265302.94, 435462.09, 699683.81, 1109965.09],
        'Profit (USD)': [10490.98, 37161.43, 88955.20, 179760.27, 331141.77, 576585.83, 968089.45],
        'Return On Investment': [35.92, 77.44, 133.24, 210.14, 317.43, 468.40, 682.35],
        'Profit Margin': [26.43, 43.64, 57.13, 67.76, 76.04, 82.41, 87.22]
    }

    # Convert into DataFrame
    df = pd.DataFrame(data)

    # Set 'Year' as the index
    df.set_index('Year', inplace=True)

    # Normalize monetary values to millions and percentage values to 0-1 scale
    df_normalized = df.copy()

    # Normalize monetary values by dividing by 1 million (to bring them into millions)
    df_normalized[['Total Cost (USD)', 'Total Return (USD)', 'Profit (USD)']] /= 1_000_000  # Normalize to millions

    # Normalize percentage columns by dividing by 100 (to bring them into a 0-1 scale)
    df_normalized[['Return On Investment', 'Profit Margin']] /= 100  # Normalize to 0-1

    # Create a heatmap
    plt.figure(figsize=(10, 6))

    # Define the color maps for monetary and percentage values
    cmap_monetary = cm.Blues  # For monetary values
    cmap_percentage = cm.RdYlGn  # For percentage values

    # Create two different normalizations
    norm_monetary = mcolors.Normalize(
        vmin=df_normalized[['Total Cost (USD)', 'Total Return (USD)', 'Profit (USD)']].min().min(),
        vmax=df_normalized[['Total Cost (USD)', 'Total Return (USD)', 'Profit (USD)']].max().max())

    norm_percentage = mcolors.Normalize(vmin=df_normalized[['Return On Investment', 'Profit Margin']].min().min(),
                                        vmax=df_normalized[['Return On Investment', 'Profit Margin']].max().max())

    # Plot the heatmap
    sns.heatmap(df_normalized, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar=False)

    # Add colorbars for each normalization
    sm_monetary = plt.cm.ScalarMappable(cmap=cmap_monetary, norm=norm_monetary)
    sm_monetary.set_array([])
    plt.colorbar(sm_monetary, label="Monetary Values (Million USD)", pad=0.02)

    sm_percentage = plt.cm.ScalarMappable(cmap=cmap_percentage, norm=norm_percentage)
    sm_percentage.set_array([])
    plt.colorbar(sm_percentage, label="Percentage Values", pad=0.02, orientation='horizontal')

    # Title and show the plot
    plt.title('Normalized Financial Metrics Heatmap with Two Colorbars')
    plt.show()
    '''
