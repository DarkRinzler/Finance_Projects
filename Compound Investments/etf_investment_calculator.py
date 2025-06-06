import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import exchange_rate_api as exrAPI
from colorama import Fore, Style, init
from matplotlib.ticker import FuncFormatter
from tabulate import tabulate

#Constants
months_in_year = 12

# Initialize colorama
init(autoreset=True)

# Timer decorator
def timer(func: Callable) -> Callable:

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time: float = time.perf_counter()
        result: Any = func(*args, **kwargs)
        end_time: float = time.perf_counter()
        print(f'Time for {func.__name__}: {end_time - start_time:.4f} seconds')

        return result
    return wrapper

# Parameters needed for calculations
#@timer
def investment_input_parameters() -> Dict[str, float]:

    def validation_numeric_input(key_prompt: str, value_prompt: float) -> None:
        if key_prompt == 'initial investment' and value_prompt <= 0:
            raise ValueError("The initial investment can not be less than 0")
        if key_prompt == 'monthly contribution' and value_prompt <= 0:
            raise ValueError("The monthly contribution can not be less or equal to 0")
        if key_prompt == 'annual return_rate' and not (0 <= value_prompt <= 100):
            raise ValueError("The average annual return rate must be between 0 and 100.")
        if key_prompt == 'investment period_years' and value_prompt < 1:
            raise ValueError("The minimum investment period must be of 1 year")

    prompts: Dict[str, str] = {
        'currency': 'Select the currency of the ETF investment from the options provided below: ',                 # Currency for the ETF investment
        'initial investment': 'Enter the initial investment amount in EUR (€): ',                                  # Amount of initial investment
        'monthly contribution': 'Enter the expected monthly contribution to the ETF investment in EUR (€): ',      # Expected monthly contribution to the ETF
        'annual return_rate': 'Enter the projected average annual return rate of the ETF investment (%): ',        # Expected annual return rate (e.g., 10%)
        'dividend yield': 'Enter the dividend yield of the ETF investment: ',                                      # Dividend yield (e.g., 0.01)
        'expense ratio': 'Enter the expense ratio of the ETF investment (%): ',                                    # Annual expense ratio of the ETF (e.g., 0.07%)
        'investment period years': 'Enter the investment period in years: ',                                       # Total investment period in years
    }

    currency_types: Dict[str, str] = {
        'USD' : 'United States Dollar',
        'EUR' : 'Euro',
        'GBP' : 'British Pound Sterling',
        'CHF' : 'Swiss Franc',
        'CAD' : 'Canadian Dollar'
    }

    currency_keys = list(currency_types.keys())
    parameters = {}

    # Handling of the currency selection
    print(prompts['currency'])
    for i, (curr_key, curr_type) in enumerate(currency_types.items(), start=1):
        print(f"{i}. {Fore.GREEN + Style.BRIGHT}{curr_key} {Fore.RESET}({curr_type})")

    while "currency" not in parameters:
        try:
            choice = int(input("Please enter the number corresponding to your selected currency: ").strip())
            if 1 <= choice <= len(currency_keys):
                parameters['currency'] = currency_keys[choice - 1]
            else:
                raise ValueError("Invalid selection. Please choose a valid option.")
        except ValueError as e:
            print(f"Error: {e}")

    # Exchange rate for currency selection
    rate = exrAPI.exchange_rate_request(parameters['currency'])

    # Inputs validation for remaining prompts
    for key, prompt in prompts.items():
        while key not in parameters:
            try:
                value = float(input(prompt).strip())
                validation_numeric_input(key, value)
                if key in ['initial investment', 'monthly contribution']:
                    parameters[key] = value * rate
                else:
                    parameters[key] = value
            except ValueError as e:
                print(f'Error: {e}')
    return parameters

# Investment derived values needed for computation of the return on investment
#@timer
def derived_values(investment_parameters: Dict[str, float]) -> Tuple[float, int]:
    average_annual_rate = (investment_parameters['annual return_rate'] - investment_parameters['expense ratio']) / 100
    return_rate = (1 + average_annual_rate) ** (1 / months_in_year) - 1
    total_months = int(investment_parameters['investment period years'] * months_in_year)
    return return_rate, total_months

# Variables tracking the investment growth through the selected investment period
#@timer
def investment_tracking(investment_parameters: Dict[str, float], investment_derived_values: Tuple[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    initial_investment = investment_parameters['initial investment']
    monthly_contribution = investment_parameters['monthly contribution']
    return_rate, total_months = investment_derived_values

    # Compute investment costs and returns (Future Value of an Annuity (FVA)) using vectorized operations

    # Time periods (months)
    time_periods = np.arange(0, total_months + 1)

    # Future value of the initial investment
    fv_initial = initial_investment * (1 + return_rate) ** time_periods

    # Future value of monthly contributions
    if np.isclose(return_rate, 0):
        fv_contributions = monthly_contribution * time_periods  # No growth case
    else:
        fv_contributions = monthly_contribution * ((1 + return_rate) ** time_periods - 1) / return_rate

    # Total investment returns
    investment_returns = fv_initial + fv_contributions

    # Investment costs (total contributions made)
    investment_costs = initial_investment +  time_periods * monthly_contribution
    return investment_returns, investment_costs

# Creating a DataFrame summary table of the investment progress at 5-year intervals
#@timer
def summary_table(investment_parameters: Dict[str, float], investment_growth: Tuple[np.ndarray, np.ndarray]) -> pd.DataFrame:
    investment_returns, investment_costs = investment_growth
    investment_period_years = int(investment_parameters['investment period years'])
    years = list(range(5, investment_period_years + 1, 5))

    # Include the final year if not already listed
    if investment_period_years % 5 != 0:
        years.append(investment_period_years)

    total_cost = np.array([investment_costs[year * months_in_year] for year in years])
    total_return = np.array([investment_returns[year * months_in_year] for year in years])
    profit = total_return - total_cost
    return_on_investment = profit / total_cost
    profit_margin = profit / total_return

    interval_summary = {
        'Year': years,
        'Total Cost': total_cost,
        'Total Return': total_return,
        'Profit': profit,
        'Return On Investment': return_on_investment,
        'Profit Margin': profit_margin
    }
    return pd.DataFrame(interval_summary)

# Plot the results (yearly data)
#@timer
def results(investment_parameters: Dict[str, float], investment_growth: Tuple[np.ndarray, np.ndarray]) -> None:
    investment_period_years = int(investment_parameters['investment period years'])
    investment_returns, investment_costs = investment_growth

    # Slice the data to plot yearly data
    years_range = range(0, investment_period_years + 1)
    yearly_returns = investment_returns[::months_in_year]
    yearly_costs = investment_costs[::months_in_year]

    plt.figure(figsize=(12, 6))
    # Plot every 12th data point (yearly data)
    investment_line = plt.plot(years_range, yearly_returns, label = 'Return', zorder=1)[0]                       # Yearly balances data
    cost_line = plt.plot(years_range, yearly_costs, label = 'Cost', zorder=1)[0]                                # Yearly expenditures data

    # Get the colors of the plot lines
    investment_color = investment_line.get_color()
    cost_color = cost_line.get_color()

    # Highlight key years with markers (every 5 years)
    key_years = list(range(5, investment_period_years + 1, 5)) + [investment_period_years]
    key_returns = investment_returns[np.array(key_years) * months_in_year]
    key_costs = investment_costs[np.array(key_years) * months_in_year]

    #Scatter plot for the key years
    plt.scatter(key_years, key_returns, color = 'red', marker = '.', s = 60,  label = '_nolegend_', zorder = 2)                     # Square marker for returns

    # Add text inside same marker for each key year
    for year, ret, cost in zip(key_years, key_returns, key_costs):

        # Adjust the vertical position for the "Value" and "Cost" text
        value_offset = 0.52 * ret  # Offset for value text
        cost_offset = 0.32 * ret  # Offset for cost text

        # Adjust the position of the text to the left slightly
        left_offset = -0.32

        # Plot dashed lines to connect markers to the year
        plt.plot([year, year], [0, ret], linestyle = '--', color = 'grey', linewidth=1)

        plt.text(
            year + left_offset,
            ret + value_offset,
            f"Return: {ret:,.2f}€",
            ha = 'center',
            va = 'center',
            fontsize = 8,
            color = investment_color,
            bbox = dict(facecolor = 'white', edgecolor = 'none', boxstyle = 'round,pad=0.2'),
            zorder = 3
        )
        plt.text(
            year + left_offset,
            ret + cost_offset,
            f"Cost: {cost:,.2f}€",
            ha = 'center',
            va = 'center',
            fontsize = 8,
            color = cost_color,
            bbox = dict(facecolor = 'none', edgecolor = 'none', boxstyle = 'round,pad=0.2'),
            zorder = 3
        )

    # Set the y-axis to logarithmic scale
    plt.yscale('log')

    # Set the x-axis limit to start from 0
    plt.xlim(left = 0, right = key_years[-1] + 1)

    # Set the y-axis to start from the initial investment value
    plt.ylim(bottom = investment_returns[0], top = investment_returns[-1] * 2.5)

    # Apply plot label and title
    plt.title(f"Investment Growth Over Time (Final Return {investment_returns[-1]:,.2f} {investment_parameters['currency']})")
    plt.xlabel("Years")
    plt.ylabel("Value (€)")
    plt.legend(loc = 'upper left')

    # Disable grid
    plt.grid(False)

    # Apply currency formatting to the y-axis
    formatter = FuncFormatter(lambda x, pos: f"€{x:,.2f}")
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save the plot as an image (e.g., JPG format)
    plt.savefig('ETF_investment_growth.jpg', dpi = 500)
    plt.close()

# Display of DataFrame summary table with formatted output
#@timer
def formatted_output(investment_parameters: Dict[str, float], summary_df: pd.DataFrame) -> None:
    # List of columns to format
    for columns in summary_df.columns:
        if columns in ['Total Cost', 'Total Return', 'Profit']:
            summary_df[columns] = summary_df[columns].map(lambda x: f'{x:,.2f}')
        elif columns in ['Return On Investment', 'Profit Margin']:
            summary_df[columns] = summary_df[columns].map(lambda x: f'{x:,.2f}%')

    # Add the currency symbol to the header of relevant columns
    summary_df[columns] = [f"{col} ({investment_parameters['currency']}" if col in ['Total Cost', 'Total Return', 'Profit'] else col for col in summary_df[columns]]

    # Print table of the dataframe
    print(f"\n{tabulate(summary_df, headers = 'keys', tablefmt = 'pipe', colalign = ['right'] * len(summary_df[columns]), showindex = False)}")


def main() -> None:

    parameters = investment_input_parameters()
    derived = derived_values(parameters)
    growth = investment_tracking(parameters, derived)
    data_frame = summary_table(parameters, growth)
    results(parameters, growth)
    formatted_output(parameters, data_frame)


if __name__ == '__main__':
    main()