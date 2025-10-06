"""
compound_investment_calculator.py
-----------------------------------------------------------------------
Estimate the implied growth rate of a publicly traded company
from its stock price using a reverse discounted cash flow (DCF) model,
for positive free cash flow (FCF).

Inputs
-----------------------------------------------------------------------
- Currency : str
- Initial investment : str
- Monthly contribution : float
- Annual return rate : int
- Investment period years : float

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
2024-12-05
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# -------------------------------------------------------- #
# v1.0  2024-12-03  Initial version
# v1.1  2025-03-20  Second Version
# v1.2  2025-10-06  Third Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
from typing import Any, Dict, Tuple

# Third-party libraries
import numpy as np
import pandas as pd
from tabulate import tabulate

# Local modules
import utils                                  # Custom helper function
import exchange_rate_api as exchange_api      # Exchange rate API custom helper function


# -------------------------------------------------------- #
#                  BASH DISPLAY OPTIONS
# -------------------------------------------------------- #

# Set display options for DataFrame
pd.set_option('display.width', 500)             # Increase the display width
pd.set_option('display.max_columns', None)      # Ensure all columns of the DataFrame are displayed

# -------------------------------------------------------- #
#                       MODEL CONSTANTS
# -------------------------------------------------------- #

# Constants
months_in_year = 12

# Currency symbols
currency_symbols: Dict[str, str] = {
        'USD': '$',                             # United States Dollar
        'EUR': '€',                             # Euro
        'GBP': '£',                             # British Pound Sterling
        'CHF': 'Fr.',                           # Swiss Franc
        'CAD': '$'                              # Canadian Dollar
    }

currency_types: Dict[str, str] = {
        'USD' : 'United States Dollar',
        'EUR' : 'Euro',
        'GBP' : 'British Pound Sterling',
        'CHF' : 'Swiss Franc',
        'CAD' : 'Canadian Dollar'
    }

# -------------------------------------------------------- #
#                  MODEL ASSUMPTIONS & NOTES
# -------------------------------------------------------- #

# - Compound investment is calculated using the Gordon Growth Model.
# - The average return rate is assumed to remain constant across all forecasted years.

# -------------------------------------------------------- #
#                    USER DATA PROMPTS
# -------------------------------------------------------- #

# Prompts for user parameter selection
prompts: Dict[str, str] = {
        'currency' : 'Select the desired currency of the investment from the options below: ',                          # Currency of the desired investment
        'initial investment' : 'Enter the amount of the initial investment in EUR (€): ',                               # Initial investment
        'monthly contribution' : 'Enter the expected monthly contribution to the investment in EUR (€): ',              # Monthly contribution
        'annual return rate': 'Enter the anticipated average annual return rate of the financial instrument (%): ',     # Annual return rate
        'investment period years': 'Enter the investment period in years: '                                             # Total investment period in years
    }

# -------------------------------------------------------- #
#                   USER DATA SELECTION
# -------------------------------------------------------- #

@utils.timer
def investment_input_parameters() -> Dict[str, Any]:
    """
        The function creates a dictionary with all the data selected from the user through the asked prompts, using custom helper
        functions to validate input and print on bash prompt selection

        Arguments:

            No arguments

        Returns:

            parameters (Dict[str, Any]) -- Dictionary containing investment-specific parameters based on user input
    """

    parameters: Dict[str, Any] = {}

    # Handling of the currency selection
    print(prompts['currency'])
    utils.input_choice('currency', currency_types, parameters)

    # Retrieve the exchange rate of the user currency selection
    rate = exchange_api.exchange_rate_request(parameters['currency'])

    # Inputs validation for remaining prompts
    for key, prompt in prompts.items():
        while key not in parameters:
            try:
                value = float(input(prompt).strip())
                utils.validation_numeric_input(key, value)
                if key in ['initial investment', 'monthly contribution']:
                    parameters[key] = value * rate
                else:
                    parameters[key] = value
            except ValueError as e:
                print(f'Error: {e}')

    return parameters

# -------------------------------------------------------- #
#                 COMPOUND INVESTMENT MODEL
# -------------------------------------------------------- #

@utils.timer
def investment_values(investment_parameters: Dict[str, Any]) -> Tuple[float, int]:
    """
        Calculates the average monthly return rate of the user's selected financial instrument, based on its average annual return rate,
        and the total number of months in the chosen investment period for forecasting

        Arguments:

            investment_parameters (Dict[str, Any]): Dictionary containing investment-specific parameters based on user input

        Returns:

            Tuple[float, int]:

                return_rate (float): Average monthly return rate of the selected financial instrument

                total_months (int): Total number of months in the user-selected investment period
    """

    # Average annual return rate of the user's financial instrument in decimals
    average_annual_rate = investment_parameters['annual return rate'] / 100

    # Average monthly return rate of the user's financial instrument in decimals
    return_rate = (1 + average_annual_rate) ** (1 / months_in_year) - 1

    # Total length in months of the investment period selected by the user
    total_months = int(investment_parameters['investment period years'] * months_in_year)

    return return_rate, total_months

@utils.timer
def investment_data_forecasting(investment_parameters: Dict[str, Any], investment_data_values: Tuple[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
        The function calculates the forecasted investment returns and costs within the selected investment period

        Arguments:

            investment_parameters (Dict[str, Any]): Dictionary containing investment-specific parameters based on user input

            investment_data_values (Tuple[float, int]): Tuple of two arrays:

                -- return_rate (float): Average monthly return rate of the selected financial instrument

                -- total_months (int): Total number of months in the user-selected investment period

        Returns:

            Tuple[np.ndarray, np.ndarray]:

                investment_returns (np.ndarray): Array of shape (total_months + 1, ) representing the investment returns for each month within
                                                    the selected investment period

                investment_costs (np.ndarray): Array of shape (total_months + 1, ) representing the investment costs for each month within
                                                  the selected investment period
    """

    # Forecasted investment returns and costs
    initial_investment = investment_parameters['initial investment']

    # Average monthly return rate and total investment duration in months
    return_rate, total_months = investment_data_values

    # User's monthly contribution for compounding
    monthly_contribution = investment_parameters['monthly contribution']

    # Time periods of the investment
    time_periods = np.arange(0, total_months + 1)

    # Future value of the initial investment
    fv_initial = initial_investment * (1 + return_rate) ** time_periods

    # Future value of monthly contributions
    if np.isclose(return_rate, 0):
        fv_contributions = monthly_contribution * time_periods  # No growth case
    else:
        fv_contributions = monthly_contribution * ((1 + return_rate) ** time_periods - 1) / return_rate

    # Total investment returns (initial + contributions)
    investment_returns = fv_initial + fv_contributions

    # Total investment costs (initial investment + sum of contributions)
    investment_costs = initial_investment + time_periods * monthly_contribution

    return investment_returns, investment_costs

@utils.timer
def investment_forecasting(investment_parameters: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
        The function calculates the forecasted investment returns and costs within the selected investment period by executing previously
        defined functions in succession

        Arguments:

            investment_parameters (Dict[str, Any]): Dictionary containing investment-specific parameters based on user input

        Returns:

            np.ndarray: Array of shape (growth_rates_interval, 1, ndim_ds_rate, ndim_tg_rate) representing the intrinsic value per share
                        for each simulation, across all combinations of discount rates and terminal growth rates
    """

    # The function calculates the investment's monthly return rate and total number of months for forecasting
    investment_data_values = investment_values(investment_parameters)

    # The function calculates the forecasted investment returns and costs
    investment_forecasted_data = investment_data_forecasting(investment_parameters, investment_data_values)

    return investment_forecasted_data

@utils.timer
def summary_table(investment_parameters: Dict[str, Any], investment_data_forecast: Tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    """
        The function generates a DataFrame of investment metrics — returns, costs, profit, profit margin, and return on investment — at 5-year intervals

        Arguments:

            investment_parameters (Dict[str, Any]): Dictionary containing investment-specific parameters based on user input

            investment_data_forecast (Tuple[np.ndarray, np.ndarray]) : Tuple of two arrays

                -- investment_returns (np.ndarray): Array of shape (total_months + 1, ) representing the investment returns for each month within
                                                    the selected investment period

                -- investment_costs (np.ndarray): Array of shape (total_months + 1, ) representing the investment costs for each month within
                                                  the selected investment period

        Returns:

            pd.DataFrame: DataFrame of shape (number_of_years, number_of_metrics) containing forecasted investment metrics at 5-year intervals
    """

    # Forecasted investment returns and costs
    investment_returns, investment_costs = investment_data_forecast

    # Generate a list of years at 5-year intervals for forecasting
    investment_period_years = int(investment_parameters['investment period years'])
    years = list(range(5, investment_period_years + 1, 5))

    # Append the final year if it is not already included
    if investment_period_years % 5 != 0:
        years.append(investment_period_years)

    # Calculate the total investment's cost at 5-year intervals
    total_cost = np.array([investment_costs[year * months_in_year] for year in years])

    # Calculate the total investment's return at 5-year intervals
    total_return = np.array([investment_returns[year * months_in_year] for year in years])

    # Calculate the total investment's profit at 5-year intervals
    profit = total_return - total_cost

    # Calculate the return on investment at 5-year intervals
    return_on_investment = (profit / total_cost) * 100

    # Calculate the investment's profit margin at 5-year intervals
    profit_margin = (profit / total_return) * 100

    # Construct a dictionary that stores all data associated with each investment metric
    interval_summary = {
        'Year': years,
        'Total Cost': total_cost,
        'Total Return': total_return,
        'Profit': profit,
        'Return On Investment': return_on_investment,
        'Profit Margin': profit_margin
    }
    return pd.DataFrame(interval_summary)

@utils.timer
def formatted_output(investment_parameters: Dict[str, Any], investment_metrics_df: pd.DataFrame) -> None:
    """
        The function formats and displays the DataFrame with improved output readability

        Arguments:

            investment_parameters (Dict[str, Any]): Dictionary containing investment-specific parameters based on user input

            investment_metrics_df (pd.DataFrame): DataFrame of shape (number_of_years, number_of_metrics) containing forecasted investment metrics
                                                  at 5-year intervals

        Returns:

            None
    """

    # # Fetch user's currency selection
    selected_currency = investment_parameters['currency']

    # Create copy of the DataFrame
    formatted_metrics_df = investment_metrics_df.copy()

    # Format the investment metrics to percentages
    for columns in investment_metrics_df.columns:
        if columns in ['Total Cost', 'Total Return', 'Profit']:
            formatted_metrics_df[columns] = formatted_metrics_df[columns].map(lambda x: f'{x:,.2f}{currency_symbols[selected_currency]}')
        elif columns in ['Return On Investment', 'Profit Margin']:
            formatted_metrics_df[columns] = formatted_metrics_df[columns].map(lambda x: f'{x:,.2f}%')

    # Calculate the width based on the column names and formatting for bash title printing
    width = sum(len(str(col) + selected_currency) + 2 for col in investment_metrics_df.columns) + 22
    print(f'\n{f' Compound Investment in {currency_types[selected_currency]}s ':-^{width}}')

    # Print the DataFrame in tabular form
    print(f'\n{tabulate(formatted_metrics_df, headers = 'keys', tablefmt = 'grid', stralign = 'center', showindex = False)}')


def main() -> None:
    """
        Entry point for the script. Executes the main workflow.
    """

    # User's investment parameters selection
    parameters = investment_input_parameters()

    # Forecasted returns and costs
    forecasted_data = investment_forecasting(parameters)

    # DataFrame creation of investment metrics
    dataframe = summary_table(parameters, forecasted_data)

    # Format DataFrame for bash terminal display
    formatted_output(parameters, dataframe)

    # Plotting of heatmap depicting implied growth rates for each discount–terminal growth rate combinations
    utils.plt_investment(parameters, forecasted_data, months_in_year)

if __name__ == '__main__':
    main()