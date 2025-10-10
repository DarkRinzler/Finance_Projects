"""
utils.py
-----------------------------------------------------------------------
Collections of custom functions for the script compound_investment_calculator.py

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
# v1.0  2025-10-06  First Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
import time                                         # Time
from functools import wraps                         # Wrapper

# Third-party libraries
import numpy as np                           # Numerical computation
import matplotlib.pyplot as plt              # Visualisation
from matplotlib.ticker import FuncFormatter  # Visualisation formatter
from colorama import Fore, init              # Bash coloring

# Initialize colorama
init(autoreset=True)

# -------------------------------------------------------- #
#                       WRAPPERS
# -------------------------------------------------------- #

def timer(func: callable) -> callable:
    """
        Decorator that measures the execution time of a function.

        Arguments:

            func (Callable) -- Function to be timed

        Returns:

            Callable -- A wrapped version of the input function that measures and reports its execution time when called.
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

def input_choice(key_prompt: str, currency_types: dict[any, any], parameters: dict[str, any]) -> None:
    """
        The function prints the available currency options in color and validates the user's selection

        Arguments:

            key_prompt (str): Dictionary key corresponding to the currency numerical value in the prompts dictionary

            currency_types (Dict[Any, Any]): Dictionary containing the available currency options

            parameters (Dict[Any, Any]): Dictionary used to store the user's selected options based on the prompts

        Returns:

            None
    """

    currency_keys = list(currency_types.keys())

    for i, (curr_key, curr_type) in enumerate(currency_types.items(), start=1):
        print(f'{i}. {Fore.GREEN}{curr_key} {Fore.RESET}({curr_type})')

    while key_prompt not in parameters:
        try:
            choice = int(input(f"Please enter the number corresponding to your selected {key_prompt}: ").strip())
            if 1 <= choice <= len(currency_keys):
                parameters['currency'] = currency_keys[choice - 1]
            else:
                raise ValueError('Invalid selection. Please choose a valid option.')
        except ValueError as err:
            print(f'Error: {err}')

def validation_numeric_input(key_prompt: str, value_prompt: float) -> None:
    """
        The function checks if the numerical inputs of the user are consistent with the assumptions of compound investment model
        of the script compound_investment_calculator.py

        Arguments:

            key_prompt (str): Dictionary key corresponding to a numerical value in the prompts dictionary

            value_prompt (float): Numerical value associated with key_prompt key in the prompts dictionary

        Returns:

            None
    """

    if key_prompt == 'initial investment' and value_prompt <= 0:
        raise ValueError("The initial investment can not be less or equal to 0")
    if key_prompt == 'monthly contribution' and value_prompt < 0:
        raise ValueError("The monthly contribution can not be less than 0")
    if key_prompt == 'annual return_rate' and not (0 <= value_prompt <= 100):
        raise ValueError("The average annual return rate must be between 0 and 100.")
    if key_prompt == 'investment period_years' and value_prompt < 1:
        raise ValueError("The minimum investment period must be of 1 year")

# -------------------------------------------------------- #
#                       PLOTTING
# -------------------------------------------------------- #

def plt_investment(investment_parameters: dict[str, any], investment_growth: tuple[np.ndarray, np.ndarray], months_in_year: int) -> None:
    """
        The function plots the yearly forecast of the user's selected financial instrument, displaying both the projected returns and costs,
        highlighting every 5 years the corresponding return and cost values

        Arguments:

            investment_parameters (Dict[str, Any]): Dictionary containing investment-specific parameters based on user input

            investment_growth (Tuple[np.ndarray, np.ndarray]): Tuple of two arrays:

                -- investment_returns (np.ndarray): Array of shape (total_months + 1, ) representing the investment returns for each month within
                                                    the selected investment period

                -- investment_costs (np.ndarray): Array of shape (total_months + 1, ) representing the investment costs for each month within
                                                  the selected investment period

            months_in_year (int): Number of months in a year

        Returns:

            None
    """

    # Retrieve the user's investment period and the yearly returns and costs
    investment_period_years = int(investment_parameters['investment period years'])
    investment_returns, investment_costs = investment_growth

    # Slice the data once to plot yearly data (not full data)
    years_range = range(0, investment_period_years + 1)
    yearly_returns = investment_returns[::months_in_year]
    yearly_costs = investment_costs[::months_in_year]

    # Define the size of the plot figure
    plt.figure(figsize = (12, 6))

    # Plot every 12th data point (yearly data)
    investment_line = plt.plot(years_range, yearly_returns, label = 'Return', zorder = 1)[0]
    cost_line = plt.plot(years_range, yearly_costs, label = 'Cost', zorder = 1)[0]

    # Fill the area between the curves with light green
    plt.fill_between(years_range, yearly_returns, yearly_costs, color = 'green', alpha = 0.3, zorder = 0)

    # Get the colors of the plot lines
    investment_color = investment_line.get_color()
    cost_color = cost_line.get_color()

    # Highlight key years with markers (every 5 years)
    key_years = list(range(5, investment_period_years + 1, 5)) + [investment_period_years]
    key_returns = investment_returns[np.array(key_years) * months_in_year]
    key_costs = investment_costs[np.array(key_years) * months_in_year]

    # Scatter plot for the key years
    plt.scatter(key_years, key_returns, color = 'red', marker = '.', s = 60,  label ='_nolegend_', zorder = 2)  # Square marker for returns
    plt.scatter(key_years, key_costs, color = 'red', marker = '.', s = 60, label = '_nolegend_',zorder = 2)      # Square marker for returns

    # Add text inside same marker for each key year
    for year, ret, cost in zip(key_years, key_returns, key_costs):

        # Adjust the vertical position for the "Value" and "Cost" text
        return_offset = 0.6 * ret           # Offset for value text
        cost_offset = 0.32 * ret            # Offset for cost text

        # Adjust the position of the text to the left slightly
        left_offset = -0.32

        # Plot dashed lines to connect markers to key years
        plt.plot([year, year], [0, ret], linestyle = '--', color = 'grey', linewidth = 1)

        # Add text labels showing the return value for each key year on the plot
        plt.text(
            year + left_offset,
            ret + return_offset,
            f'Return: {ret:,.2f}',
            ha = 'center',
            va = 'center',
            fontsize = 8,
            color = investment_color,
            bbox = dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'),
            zorder = 3
        )

        # Add text labels showing the cost value for each key year on the plot
        plt.text(
            year + left_offset,
            ret + cost_offset,
            f'Cost: {cost:,.2f}',
            ha = 'center',
            va = 'center',
            fontsize = 8,
            color = cost_color,
            bbox = dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.2'),
            zorder = 3
        )

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Set the x-axis limit to start from 0
    plt.xlim(left = 0, right = key_years[-1] + 1)

    # Set the y-axis to start from the initial investment value
    plt.ylim(bottom = investment_returns[0], top = investment_returns[-1] * 2.5)

    # Apply plot label and title
    plt.title(f'Investment Growth - Total Return: {investment_returns[-1]:,.2f} {investment_parameters['currency']}',
              fontsize = 14, fontweight = 'bold', color='black', pad = 10)
    plt.xlabel('Years', fontsize = 12, fontweight = 'bold', color = 'black', labelpad = 8)
    plt.ylabel(f'Return', fontsize = 12, fontweight = 'bold', color = 'black', labelpad = 8)
    plt.legend(loc='upper left')

    # Disable grid
    plt.grid(False)

    # Apply currency formatting to the y-axis
    formatter = FuncFormatter(lambda x, pos: f'{x:,.2f}')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig('investment_forecast.jpg', dpi = 500)
    plt.close()
