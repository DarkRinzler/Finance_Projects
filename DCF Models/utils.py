# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #


# Standard libraries
from typing import Any, Dict, Callable       # Type annotation
import time                                  # Time
from functools import wraps                  # Wrapper
import math                                  # Math operations

# Third-party libraries
import pandas as pd                          #
import matplotlib.pyplot as plt              #
import seaborn as sns                        #
from colorama import Fore, init              # Bash coloring


# Initialize colorama
init(autoreset=True)


# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #


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


# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

def input_choice(key_prompt: str, curr_dict : Dict[Any, Any], param: Dict[str, Any]) -> None:
    curr_keys = list(curr_dict.keys())
    for i, (sel_key, sel_type) in enumerate(curr_dict.items(), start = 1):
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


# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Function to plot a Heat-Map of the implied growth rates for all pairs of discount and terminal growth rates (discrete case)
def plt_igr(stock_parameters: Dict[str, Any], df: pd.DataFrame) -> None:

    # Remove DataFrame index before plotting Heat-Map
    df.index.name = None

    # Set the
    min_val = math.floor(df.min().min() / 5) * 5
    max_val = math.ceil(df.max().max() / 5) * 5

    plt.figure(figsize=(16, 6))

    # Create Heat-Map
    sns.heatmap(df, vmin = min_val, vmax = max_val, cmap = 'Blues', annot = True, fmt = '.2f', linewidths = 0.5, square = False)

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
    plt.title(f'Implied Growth Rates - {stock_parameters['company_stock']}', fontsize = 14, fontweight = 'bold', color = 'black', pad = 15)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format) with corresponding company name
    plt.savefig(f'IGR_HM_({stock_parameters['company_stock']}.jpg', dpi = 500)
    plt.close()

# Function to plot a Heat-Map of the implied growth rates for all pairs of discount and terminal growth rates (extended case)
def ext_plt_igr(stock_parameters: Dict[str, Any], df: pd.DataFrame) -> None:

    # Remove DataFrame index before plotting Heat-Map
    df.index.name = None
    min_val = math.floor(df.min().min() / 5) * 5
    max_val = math.ceil(df.max().max() / 5) * 5

    plt.figure(figsize = (16, 6))

    sns.heatmap(df, vmin = min_val, vmax = max_val, cmap = 'Blues', annot = True, fmt = '.2f', linewidths = 0.5, square = False)

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

    # Save plot as an image (JPG format)
    plt.savefig(f'Ext_IGR_HM_({stock_parameters['company_stock']}.jpg', dpi = 500)
    plt.close()
