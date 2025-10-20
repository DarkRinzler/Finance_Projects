"""
utils.py
-----------------------------------------------------------------------
Collections of custom functions for the script reverse_discounted_cash_flow.py

Dependencies
-----------------------------------------------------------------------
- pandas
- seaborn

Usage
-----------------------------------------------------------------------
$ python utils.py

Author
-----------------------------------------------------------------------
Riccardo Nicolò Iorio

Date
-----------------------------------------------------------------------
2025-01-28
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# -------------------------------------------------------- #
# v1.0  02-20-2025  Initial version
# v1.3  09-26-2025  Second Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
import time                                  # Time
from functools import wraps                  # Wrapper
import math                                  # Math operations

# Third-party libraries
import pandas as pd                          # Data manipulation
import matplotlib.pyplot as plt              # Visualisation
import seaborn as sns                        # Visualisation (heatmap)
from colorama import Fore, init              # Bash coloring

# Initialize colorama
init(autoreset = True)

# -------------------------------------------------------- #
#                       WRAPPERS
# -------------------------------------------------------- #

def timer(func: callable) -> callable:
    """
        Decorator that measures the execution time of a function.

        Parameters:
            func:  **(callable)**
                Function to be timed

        Returns:
            **(callable)**:
                A wrapped version of the input function that measures and reports its execution time when called.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> any:
        start_time: float = time.perf_counter()
        result: any = func(*args, **kwargs)
        end_time: float = time.perf_counter()
        print(f"Time for {func.__name__} function: {end_time - start_time:.4f} seconds")

        return result
    return wrapper

# -------------------------------------------------------- #
#                       USER SELECTION
# -------------------------------------------------------- #

def input_choice(key_prompt: str, stock_parameters: dict[any, any], parameters: dict[str, any]) -> None:
    """
        The function prints the available currency options in color and validates the user's selection

        Parameters:
            key_prompt: **(str)**
                Dictionary key corresponding to the currency numerical value in the prompts dictionary

            stock_parameters: **(dict[any, any])**
                Dictionary containing the available currency options

            parameters: **(dict[any, any])**
                Dictionary used to store the user's selected options based on the prompts

        Returns:
            **None**
    """

    # Output to bash the available currency options
    curr_keys = list(stock_parameters.keys())
    for i, (sel_key, sel_type) in enumerate(stock_parameters.items(), start = 1):
        print(f'{i}. {Fore.GREEN}{sel_key} {Fore.RESET}({sel_type})')

    while key_prompt not in parameters:
        try:
            choice = int(input(f"Please enter the number corresponding to your selected {key_prompt}: ").strip())
            if 1 <= choice <= len(curr_keys):
                parameters[key_prompt] = curr_keys[choice - 1]
            else:
                raise ValueError("Invalid selection. Please choose a valid option.")
        except ValueError as err:
            print(f"Error: {err}")

def validation_numeric_input(key_prompt: str, value_prompt: float) -> None:
    """
        The function checks if the numerical inputs of the user are consistent with the assumptions of the reverse
        discounted cash flow model of the script reverse_discounted_cash_flow.py

        Parameters:
            key_prompt: **(str)**
                Dictionary key corresponding to a numerical value in the prompts dictionary

            value_prompt: **(float)**
                Numerical value associated with key_prompt key in the prompts dictionary

        Returns:
            **None**
    """

    # Raise an error if numerical inputs (e.g., stock price, outstanding shares, or LTM free cash flow) are invalid or missing
    if key_prompt == 'stock_price' and value_prompt <= 0:
        raise ValueError("The stock price of the company can not be less or equal to 0")
    if key_prompt == 'outstanding_shares' and value_prompt <= 0:
        raise ValueError("The number of outstanding shares can not be less or equal to 0")
    if key_prompt == 'ltm_fcf' and value_prompt <= 0:
        raise ValueError("The Free Cash Flow of the last twelve months can not be less or equal to 0")

# -------------------------------------------------------- #
#                       PLOTTING
# -------------------------------------------------------- #

def plt_heatmap(stock_parameters: dict[str, any], growth_rates: pd.DataFrame) -> None:
    """
        The function creates a heatmap of the implied growth rates for all pairs of discount and terminal growth rates

        Parameters:
            stock_parameters: **(dict[str, any])**
                Dictionary containing the user's company selection

            growth_rates: **(pd.DataFrame)**
                Dataframe containing the implied growth rates for all discount/terminal growth rate pairs

        Returns:
            **None**
    """

    # Remove the DataFrame index before plotting Heat-Map
    growth_rates.index.name = None

    # Normalize the color scale to rounded min/max values for pretty plotting
    min_val = math.floor(growth_rates.min().min() / 5) * 5
    max_val = math.ceil(growth_rates.max().max() / 5) * 5

    # Define the size of the plot figure
    plt.figure(figsize = (16, 6))

    # Create a colormap
    lightblue = sns.color_palette("vlag_r", as_cmap = True)

    # Define of the heatmap with corresponding parameters
    sns.heatmap(growth_rates, vmin = min_val, vmax = max_val, cmap = lightblue, annot = True, fmt = '.2f', linewidths = 0.5, square = False)

    # Customize tick label fonts
    for tick_label in plt.gca().get_xticklabels():
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')

    for tick_label in plt.gca().get_yticklabels():
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')
        tick_label.set_rotation(0)

    # Add the % sign to the annotations after formatting
    for text in plt.gca().texts:
        text.set_text(f'{text.get_text()}%')
        text.set_fontweight('bold')

    # Apply plot label and title
    plt.title(f'Implied Growth Rates - {stock_parameters['company_stock']}', fontsize = 14, fontweight = 'bold', color = 'black', pad = 15)
    plt.xlabel('Terminal Growth Rate', fontsize = 12, fontweight = 'bold', color = 'black', labelpad = 10)
    plt.ylabel('Discount Rate', fontsize = 12, fontweight = 'bold', color = 'black', labelpad = 10)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format) with company name
    plt.savefig(f'{stock_parameters['company_stock']}_growth_rates.jpg', dpi = 500)
    plt.close()

