"""
reverse_discounted_cash_flow.py
-----------------------------------------------------------------------
Estimate the implied growth rate of a publicly traded company
from its stock price using a reverse discounted cash flow (DCF) model,
for positive free cash flow (FCF)

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
  for combinations of discount rates and terminal growth rates
- Solve reverse DCF numerically (Monte Carlo approach)
- Generate heatmaps of implied growth rates

Dependencies
-----------------------------------------------------------------------
- pandas
- numpy

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
# -------------------------------------------------------- #
# v1.0  03-21-2025  Initial version
# v1.1  04-13-2025  Second Version
# v1.2  06-17-2025  Third Version
# v1.3  09-23-2025  Fourth Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Third-party Libraries
import numpy as np                      # Numerical computation
import pandas as pd                     # Data manipulation
from tabulate import tabulate           # Visualisation (bash)
from colorama import Fore, init         # Bash coloring

# Local modules
import utils                            # Custom helper functions

# -------------------------------------------------------- #
#                  BASH DISPLAY OPTIONS
# -------------------------------------------------------- #

# Initialize colorama
init(autoreset = True)

# Set display options for DataFrame
pd.set_option('display.width', 500)             # Increase the display width
pd.set_option('display.max_columns', None)      # Ensure all columns of the DataFrame are displayed

# -------------------------------------------------------- #
#                       MODEL CONSTANTS
# -------------------------------------------------------- #

# Projection Years
rDCF_years: int = 10

# Length of growth rates interval
growth_rates_interval: int = int(1e3)

# Terminal growth rates assumptions
terminal_growth_rates: dict[int, str] = {
        1: 'Minimal Growth',                       # 1% - low growth
        2: 'Moderate Growth',                      # 2% - moderate growth
        3: 'Accelerated Growth',                   # 3% - high growth
        4: 'Exponential Growth'                    # 4% - very high growth
    }

# Discount rates assumptions
discount_rates: dict[int, str] = {
        6: 'Capital Protection',                  # 6% - very low risk
        8: 'Low Volatility',                      # 8% - low risk
        10: 'Moderate Risk',                      # 10% - normal risk
        12: 'Growth Oriented',                    # 12% - high risk
        15: 'Aggressive Growth'                   # 15% - very high risk
    }

currency_types: dict[str, str] = {                # Available currency from API
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
prompts: dict[str, str] = {
        'company_stock': 'Enter the stock ticker symbol of the company you want to analyze using the rDCF model: ',               # Company Ticker
        'currency': 'Choose the currency of the company’s stock price from the options provided below: ',                         # Currency of the traded stock
        'stock_price': 'Enter the current stock price of the company: ',                                                          # Current stock price
        'outstanding_shares': 'Enter the number of outstanding shares of the company (in billions, to two decimal places): ',     # Total outstanding shares
        'ltm_fcf': 'Enter the Free Cash Flow of the last twelve months (in billions, to two decimal places): ',                   # Free Cash Flow of the last 12 months
    }

# -------------------------------------------------------- #
#                   USER DATA SELECTION
# -------------------------------------------------------- #

@utils.timer
def input_stock_parameters() -> dict[str, any]:
    """
        The function creates a dictionary with all the data selected from the user through the asked prompts, using custom helper
        functions to validate input and print on bash prompt selection

        Parameters:
            No parameters

        Returns:
            parameters (dict[str, any]): Dictionary containing stock-specific parameters based on user input
    """

    parameters: dict[str, any] = {}

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

# Numpy arrays for the discount and terminal growth rates in the range
ds_rate = np.linspace(min(discount_rates), max(discount_rates), 10, endpoint = True)
tg_rate = np.linspace(min(terminal_growth_rates), max(terminal_growth_rates), 7, endpoint = True)

# Reshaped numpy arrays of the discount and terminal growth rates for broadcasting
ndim_ds_rate = (ds_rate / 100).reshape(1, 1, len(ds_rate), 1)
ndim_tg_rate = (tg_rate / 100).reshape(1, 1, 1, len(tg_rate))

# Dictionary with entries containing discount and terminal growth rates
simulation_rates = {
    'implied_growth_rates': {'ds_rates': ndim_ds_rate, 'tgr_rates': ndim_tg_rate}
}

@utils.timer
def projected_free_cash_flow(stock_parameters: dict[str, any], growth_rates: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """
        The function calculates the projected free cash flow (FCF) matrix for a given time period and set of growth rates

        Parameters:
            stock_parameters: **(dict[str, any])**
                Dictionary containing stock-specific parameters based on user input

            growth_rates: **(np.ndarray)**
                Array of shape (growth_rates_interval, 1, 1, 1) containing the growth rate

        Returns:
            **(tuple[np.ndarray, np.ndarray])**:
                Tuple containing the time period of calculation and projected free cash flow (FCF) matrix for a given time period and set of growth rates
                - **time_period** (**np.ndarray**): Array of shape (1, n_years, 1, 1) representing the time steps
                - **free_cash_flow** (**np.ndarray**): Array of shape (growth_rates_interval, n_years, 1, 1) containing the projected FCF for each simulated growth rate over the time period
    """

    # Numpy array of the time period, reshaped for broadcasting
    time_period = np.arange(1, rDCF_years + 1).reshape(1, rDCF_years, 1, 1)

    # Reshape the growth rate array and convert percentages to decimal values
    decimal_growth_rate = growth_rates.reshape(-1, 1, 1, 1) / 100

    # Calculate the free cash flow matrix
    free_cash_flow = stock_parameters['ltm_fcf'] * ((1 + decimal_growth_rate) ** time_period)

    return time_period, free_cash_flow

@utils.timer
def present_value_free_cash_flow(discount_rate: np.ndarray, projected_fcf: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
        The function calculates present value (PV) of the free cash flow (FCF) matrix for a given time period and set of growth rates

        Parameters:
            discount_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_ds_rate, 1) containing the assumed discount rates

            projected_fcf: **(tuple[np.ndarray, np.ndarray])**
                Tuple containing the time period of calculation and projected free cash flow (FCF) matrix for a given time period and set of growth rates
                - **time_period** **(np.ndarray)**: Array of shape (1, n_years, 1, 1) representing the time steps
                - **free_cash_flow** **(np.ndarray)**: Array of shape (growth_rates_interval, n_years, 1, 1) containing the projected FCF for each simulated growth rate over the time period

        Returns:
            **(np.ndarray)**:
                Array of shape (growth_rates_interval, 1, ndim_ds_rate, 1) representing the present value of the FCF for each simulation over the time period
    """

    # Unpack the tuple to retrieve the time_period array and free cash flow (FCF) matrix
    time_period, free_cash_flow = projected_fcf

    # Calculate the discounted free cash flow matrix
    present_value_fcf = free_cash_flow / ((1 + discount_rate) ** time_period)

    return np.sum(present_value_fcf, axis = 1, keepdims = True)

@utils.timer
def discounted_terminal_value(discount_rate: np.ndarray, terminal_growth_rate: np.ndarray, projected_fcf: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
        The function calculates the discounted terminal value (TV) of the free cash flow (FCF) matrix for a given time period and set of growth rates

        Parameters:
            discount_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_ds_rate, 1) containing the assumed discount rates

            terminal_growth_rate: **(np.ndarray)**
                Array of shape (1, 1, 1, ndim_tg_rate) containing the assumed terminal growth rates

            projected_fcf: **(tuple[np.ndarray, np.ndarray])**
                Tuple containing the time period of calculation and projected free cash flow (FCF) matrix for a given time period and set of growth rates
                - **time_period** **(np.ndarray)**: Array of shape (1, n_years, 1, 1) representing the time steps
                - **free_cash_flow** **(np.ndarray)**: Array of shape (growth_rates_interval, n_years, 1, 1) containing the projected FCF for each simulated growth rate over the time period

        Returns:
            present_terminal_value: **(np.ndarray)**
                Array of shape (growth_rates_interval, n_years, ndim_ds_rate, ndim_tg_rate) representing the discounted present value of the terminal value for each simulation over the time period
        """

    # Unpack the tuple to retrieve the free cash flow (FCF) matrix
    _, free_cash_flow = projected_fcf

    # Calculates the terminal value (TV)
    terminal_value_fcf = free_cash_flow[:, -1:, :, :] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)

    # Calculates the discounted terminal value (TV)
    present_terminal_value = terminal_value_fcf / ((1 + discount_rate) ** rDCF_years)

    return present_terminal_value

@utils.timer
def intrinsic_value_share(stock_parameters: dict[str, any], discounted_fcf: np.ndarray, discounted_tv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        The function calculates the total present value (PV) of the terminal value (TV) of the free cash flow (FCF) matrix
        and the intrinsic value per share for a given time period and set of growth rates

        Parameters:
            stock_parameters:  **(dict[str, any])**
                Dictionary containing stock-specific parameters based on user input

            discounted_fcf: **(np.ndarray)**
                Array of shape (growth_rates_interval, 1, ndim_ds_rate, 1) representing the present value of the FCF for each simulation over the time period

            discounted_tv: **(np.ndarray)**
                Array of shape (growth_rates_interval, n_years, ndim_ds_rate, ndim_tg_rate) representing the present value of the terminal value for each simulation over the time period

        Returns:
            **(tuple[np.ndarray, np.ndarray])**:
                Tuple containing the total present value (PV) of the terminal value (TV) of the free cash flow (FCF) matrix and the intrinsic value per share for a given time period and set of growth rates
                - **total_present_value **(np.ndarray)**: Array of shape (growth_rates_interval, n_years, ndim_ds_rate, ndim_tg_rate) representing the present value of the total present value for each simulation over the time period, across all combinations of discount rates and terminal growth rates
                - **intrinsic_value_per_share** **(np.ndarray)**: Array of shape (growth_rates_interval, n_years, ndim_ds_rate, ndim_tg_rate) representing the intrinsic value per share computed from the total PV for each simulation over the time period, across all combinations of discount rates and terminal growth rates
    """

    # Calculates the total present value (PV) of the free cash flow (FCF)
    total_present_value = discounted_fcf + discounted_tv

    # Calculates the intrinsic value per share given the total present value (PV)
    intrinsic_value_per_share = total_present_value / stock_parameters['outstanding_shares']

    return total_present_value, intrinsic_value_per_share

@utils.timer
def calculate_intrinsic_value(stock_parameters: dict[str, any], discount_rate: np.ndarray, terminal_growth_rate: np.ndarray, growth_rate: np.ndarray) -> np.ndarray:
    """
        The function calculates the intrinsic value per share by executing previously defined functions in succession

        Parameters:
            discount_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_ds_rate, 1) containing the assumed discount rates

            terminal_growth_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_tg_rate) containing the assumed terminal growth rates

            stock_parameters: **(dict[str, any])**
                Dictionary containing stock-specific parameters based on user input

            growth_rate: **(np.ndarray)**
                Array of shape (1, ) containing the growth rate

        Returns:
            intrinsic_value_per_share: **(np.ndarray)**
                Array of shape (growth_rates_interval, 1, ndim_ds_rate, ndim_tg_rate) representing the intrinsic value per share for each simulation, across all combinations of discount rates and terminal growth rates
    """

    # The function calculates the projected free cash flow (FCF) matrix
    projections = projected_free_cash_flow(stock_parameters, growth_rate)

    # The function calculates present value (PV) of the free cash flow (FCF)
    discounted_fcf = present_value_free_cash_flow(discount_rate, projections)

    # The function calculates the terminal value (TV) of the free cash flow (FCF)
    discounted_tv = discounted_terminal_value(discount_rate, terminal_growth_rate, projections)

    # The function calculates the total present value (PV) and the intrinsic value per share
    total_present_value, intrinsic_value_per_share = intrinsic_value_share(stock_parameters, discounted_fcf, discounted_tv)

    return intrinsic_value_per_share

@utils.timer
def growth_rate_optimisation(stock_parameters: dict[str, any], discount_rate: np.ndarray, terminal_growth_rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        The function calculates the intrinsic value per share by executing a Monte Carlo simulation to find the closest growth rate that minimises
        the error with the current market stock price and the respective error

        Parameters:
            stock_parameters: **(dict[str, any])**
                Dictionary containing stock-specific parameters based on user input

            discount_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_ds_rate, 1) containing the assumed discount rates

            terminal_growth_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_tg_rate) containing the assumed terminal growth rates

        Returns:
            **(tuple[np.ndarray, np.ndarray])**:
                Tuple containing the best growth rate and relative error
                - **best_growth_rate** **(np.ndarray)**: Array of shape (1, 1, ndim_ds_rate, ndim_tg_rate) representing the growth rate that minimises the difference between the intrinsic value per share and the stock price across all combinations of discount rates and terminal growth rates
                - **min_error** **(np.ndarray)**: Array of shape (1, 1, ndim_ds_rate, ndim_tg_rate) representing the minimum deviation over the time period, across all combinations of discount rates and terminal growth rates
    """

    # # Generate array of candidate growth rates within [-20, 40] based on expected industry trends
    candidate_growth_rates = np.linspace(-20, 40, num = growth_rates_interval, endpoint = True)

    # Calculate intrinsic value per share calculation for every combination of discount rates, terminal growth rates, and growth rates
    intrinsic_share_value = calculate_intrinsic_value(stock_parameters, discount_rate, terminal_growth_rate, candidate_growth_rates)

    # Array of deviations from the current market price for all discount rate and terminal growth rate combinations
    error = np.abs(intrinsic_share_value - stock_parameters['stock_price'])

    # # Identify the index minimizing deviation from current market price across all discount rate and terminal growth rate combinations
    min_error_index = np.argmin(error, axis = 0, keepdims = True)

    # Array of minimized deviations for all discount rate and terminal growth rate combinations
    min_error = np.take_along_axis(error, min_error_index, axis = 0).astype(float)

    # Growth rates minimizing deviations for all discount rate and terminal growth rate combinations
    best_growth_rate = np.take_along_axis(candidate_growth_rates[:, None, None, None], min_error_index, axis = 0).astype(float)

    return best_growth_rate, min_error

# -------------------------------------------------------- #
#                     DATAFRAME CREATION
# -------------------------------------------------------- #


@utils.timer
def implied_growth_rates_pd(stock_parameters: dict[str, any], discount_rate:np.ndarray, terminal_growth_rate: np.ndarray) -> pd.DataFrame:
    """
        The function creates a DataFrame of the implied growth rates of the stock across all combinations of discount rates
        and terminal growth rates

        Parameters:
            stock_parameters: **(dict[str, any])**
                Dictionary containing stock-specific parameters based on user input

            discount_rate: **(np.ndarray)**
                Array of shape (1, 1, ndim_ds_rate, 1) containing the assumed discount rates

            terminal_growth_rate: **(np.ndarray)**
                Array of shape (1, 1, 1, ndim_tg_rate) containing the assumed terminal growth rates

        Returns:
            growth_rate_dataframe: **(pd.DataFrame)**
                DataFrame of shape (ndim_ds_rate, ndim_tg_rate) with the implied growth rates for all discount–terminal growth rate combinations
    """

    # Vectorized computation of the implied growth rates through Monte Carlo simulations across all discount and terminal growth rate pairs
    implied_growth_rates = growth_rate_optimisation(stock_parameters, discount_rate, terminal_growth_rate)[0]

    # Lists of labels for the discount and terminal growth rates
    ndim_ds_rates_labels = [f"{rates:.0f}%" for rates in ds_rate]
    ndim_tgr_rates_labels = [f"{rates:.1f}%" for rates in tg_rate]

    # Create DataFrame indexed by discount rates with columns for terminal growth rates
    growth_rate_dataframe = pd.DataFrame(index = ndim_ds_rates_labels, columns = ndim_tgr_rates_labels)

    # Eliminate dimensions of size 1
    implied_growth_rates_2D = implied_growth_rates.squeeze(axis = (0, 1))

    # Assign implied growth rates across all discount–terminal growth rate combinations to the DataFrame
    growth_rate_dataframe[:] = pd.DataFrame(implied_growth_rates_2D)

    # The DataFrame index is set to the company stock name with color formatting
    growth_rate_dataframe.index.name = f"{Fore.GREEN}{stock_parameters['company_stock'].upper()}{Fore.RESET}"

    return growth_rate_dataframe.astype(float)

@utils.timer
def formatted_output(stock_parameters: dict[str, any], implied_gr_rate_df: pd.DataFrame) -> None:
    """
        The function formats and displays the DataFrame with improved output readability

        Parameters:
            stock_parameters: **(dict[str, any])**
                Dictionary containing stock-specific parameters based on user input

            implied_gr_rate_df: **(pd.DataFrame)**
                DataFrame of shape (ndim_ds_rate, ndim_tg_rate) with the implied growth rates for all discount–terminal growth rate combinations

        Returns:
            **None**
    """

    # Format the implied growth rates to percentages
    formatted_matrix_df = implied_gr_rate_df.map(lambda x: f"{x:.2f}%")

    # Calculate the width based on the column names and formatting for bash title printing
    width = sum(len(str(col)) for col in terminal_growth_rates.items()) + len(implied_gr_rate_df.columns) + len(stock_parameters['company_stock']) + 20
    header = " Reverse Discounted Cash Flow - {stock_parameters['company_stock']} "
    print(f"\n{header:-^{width}}")

    # Print the DataFrame in tabular form
    print(f"\n{tabulate(formatted_matrix_df, headers ='keys', tablefmt = 'grid', stralign = 'center', showindex = True)}")

def main() -> None:
    """
        Entry point for the script. Executes the main workflow.
    """

    # User's stock parameters selection
    parameters = input_stock_parameters()

    for label, rates in simulation_rates.items():
        # DataFrame creation of implied growth rates
        dataframe = implied_growth_rates_pd(parameters, rates['ds_rates'], rates['tgr_rates'])

        # Format DataFrame for bash terminal display
        formatted_output(parameters, dataframe)

        # Plotting of heatmap depicting implied growth rates for each discount–terminal growth rate combinations
        utils.plt_heatmap(parameters, dataframe)


if __name__ == '__main__':
    main()
