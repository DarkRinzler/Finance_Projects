"""
performance_share_plan.py
-----------------------------------------------------------------------
Estimate the

Inputs
-----------------------------------------------------------------------
- aurora_metals_data (Excel file): Raw company dataset containing historical stock prices, exchange rates, and interest rates

Process
-----------------------------------------------------------------------
- Compute implied growth rates over a 10-year horizon
  for combinations of discount rates and terminal growth rates
- Solve reverse DCF numerically (Monte Carlo approach)
- Generate heatmaps of implied growth rates

# CHANGE SECTION

Dependencies
-----------------------------------------------------------------------
- pandas
- numpy

Usage
-----------------------------------------------------------------------
$ python performance_share_plan.py

Author
-----------------------------------------------------------------------
Riccardo Nicolò Iorio

Date
-----------------------------------------------------------------------
2025-10-07
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# -------------------------------------------------------- #
# v1.0  2025-10-07  Initial version
# v1.0  2025-10-10  Second version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Third-party Libraries
import numpy as np                      # Numerical computation
import pandas as pd                     # Data manipulation
from tabulate import tabulate           # Visualisation (bash)

# Local modules
import utils                            # Custom helper functions

# -------------------------------------------------------- #
#                  BASH DISPLAY OPTIONS
# -------------------------------------------------------- #

# Set display options for DataFrame
pd.set_option('display.width', 500)             # Increase the display width
pd.set_option('display.max_columns', None)      # Ensure all columns of the DataFrame are displayed

# -------------------------------------------------------- #
#                  MODEL CONSTANTS
# -------------------------------------------------------- #

# Number of simulations
n_simulations: int = int(1e5)

# Number of total trading days in a year
trading_days: int = 252

# Time step for the stochastic evolution
dt: float = float(1 / trading_days)

# Days of correlation
days: int = 7

# Performance period in years
T: int = 1

# Seed generator
np.random.seed(seed = 12345)

# Dictionary containing total shareholder return and annualised volatility tranche weight and targets
targets: dict[str, dict[str, any]] = {
    'total_shareholder_return': {
        'weight': 0.5,
        'return_targets': (0, 0.05, 0.1)
    },
    'annualised_volatility': {
        'weight': 0.5,
        'volatility_targets': (0.5, 0.4, 0.3)
    }
}

# -------------------------------------------------------- #
#            DATAFRAME EXPLORATION & STATISTICS
# -------------------------------------------------------- #

def log_data(data: pd.DataFrame) -> pd.DataFrame:
    """
        The function adds log-returns for the company's stock price and exchange rate to the input DataFrame, and renames columns for improved clarity

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data

        Returns:

            **enriched_data** (pd.DataFrame): DataFrame of shape (n_rows, n_cols + 2) containing the original company data with additional columns for log-returns
                                                  of the stock price and the exchange rate
    """

    # Rename columns of the Dataframe and sort it in ascending order by date for later use
    enriched_data = (
        data
        .sort_values(by = 'Date', ascending = True, ignore_index = True)
        .rename(columns = {
            'Date': 'date',
            'EUR/SEK': 'eur/sek',
            'AME1V Close': 'stock_eur_price',
            'EURO 1 y': 'eur_1y_rate',
            'SEK 1 y': 'sek_1y_rate',
        })
    )

    # Add daily price in Swedish Krone (SEK)
    enriched_data['stock_sek_price'] = enriched_data['stock_eur_price'] * enriched_data['eur/sek']

    # Add daily log-returns for the company stock euro price
    enriched_data['log_sek_returns'] = np.log(enriched_data['stock_sek_price'] / enriched_data['stock_sek_price'].shift(periods = 1, axis = 0))

    # Add daily log-returns for the company stock euro price
    enriched_data['log_eur_returns'] = np.log(enriched_data['stock_eur_price'] / enriched_data['stock_eur_price'].shift(periods = 1, axis = 0))

    # Add daily log-returns for the exchange rate
    enriched_data['log_eur/sek'] = np.log(enriched_data['eur/sek'] / enriched_data['eur/sek'].shift(periods = 1, axis = 0))

    return enriched_data.round(decimals = 4)

def nw_annualised_moments(data: np.ndarray, lag: int, axis: int, keepdims: bool) -> np.ndarray:
    """
        The function computes the Newey–West adjusted standard deviation for a time series or array of observations

        The Newey–West estimator corrects the standard variance formula for autocorrelation up to a given lag. This adjustment provides a more accurate estimate
        of the variance when returns or residuals are serially correlated, as is common in financial time series

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            lag (int): Number of autocorrelation lags to include in the Newey–West correction

            axis (int): Axis along which to compute the statistic

            keepdims (bool): Whether to keep the reduced dimensions in the output

        Returns:

            **nw_var** (np.ndarray): Newey–West adjusted standard deviation of the input data, computed along the specified axis
    """

    # Number of samples along the specified axis
    n_samples  = data.shape[axis]

    # Center the data by subtracting the mean along the given axis
    mean = data.mean(axis = axis, keepdims = True)
    deviations = data - mean

    # Assign standard (uncorrected) sample variance
    nw_var = np.sum(deviations ** 2, axis = axis, keepdims = keepdims) / (n_samples - 1)

    # Add weighted autocovariance terms for each lag up to 'lag'
    for k in range(1, lag + 1):
        if axis == 0:
            gamma_k = np.sum(deviations[k:] * deviations[:-k], axis = axis, keepdims = keepdims) / (n_samples - k)
        else:
            gamma_k = np.sum(deviations[:, k:] * deviations[:, :-k], axis = axis, keepdims = keepdims) / (n_samples - k)
        weight_k = 1 - k / (lag + 1)
        nw_var += 2 * weight_k * gamma_k

    return np.sqrt(nw_var)


def annualised_moments(data: pd.DataFrame, log_stock: str = 'log_sek_returns', log_exchange: str = 'log_eur/sek') -> dict[str, tuple[float, float, float]]:
    """
        The function computes the annualised first two moments (mean and standard deviation) of log-returns for the company stock price and exchange rate

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            log_stock (str): Column name for the stock log-returns, default is 'log_returns'

            log_exchange (str): Column name for the exchange rate log-returns, default is 'log_eur/sek'

        Returns:

            **log_returns_stats** (dict[str, tuple[float, float]]): Dictionary mapping each series to a tuple of its first two moments (annualised mean and
                                                                annualised standard deviation), for the log-returns of the stock price and the exchange rate
    """

    # Columns to compute statistics on
    stats_columns = [log_stock, log_exchange]

    # Calculate annualised first two moments (mean and standard deviation)
    annualised_stats = pd.DataFrame({
            'mean': data[stats_columns].mean(skipna = True) * trading_days,
            'std': data[stats_columns].std(skipna = True, ddof = 1) * np.sqrt(trading_days),
            'nw_std' : nw_annualised_moments(data = data[stats_columns].dropna().to_numpy(), lag = days, axis = 0, keepdims = False) * np.sqrt(trading_days)
        }, index = stats_columns)

    # Dictionary containing tuples containing the first and second moment for the stock price and exchange rate log-returns
    log_returns_stats = {
        f'{column}_moments': (
            float(annualised_stats.loc[column, 'mean']),
            float(annualised_stats.loc[column, 'std']),
            float(annualised_stats.loc[column, 'nw_std'])
        )
        for column in stats_columns
    }

    return log_returns_stats

def cholesky_factorisation(data: pd.DataFrame, stock: str = 'stock_sek_price', exchange: str = 'eur/sek') -> tuple[bool, np.ndarray | None]:
    """
        The function checks if the correlation matrix between the stock price and exchange rate allows for Cholesky factorisation

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            stock (str): Column name for the stock price, default is 'stock_price'

            exchange (str): Column name for the exchange rate, default is 'eur/sek'

        Returns:
            tuple[bool, np.ndarray | None]:

                - **factorisation** (bool): True if the correlation matrix is positive definite and Cholesky factorisation is possible, False otherwise

                - **triangular_factor** (np.ndarray | None): Lower-triangular matrix from the Cholesky factorisation if possible, None otherwise
    """

    # Initialize flag for Cholesky factorisation
    factorisation = False

    # Compute correlation matrix between stock prices and exchange rates
    corr_matrix = data.loc[:, [stock, exchange]].corr(method = 'pearson', numeric_only = True)

    # Compute eigenvalues of the correlation matrix
    eigenvalues = np.linalg.eigvals(corr_matrix)

    # If the correlation matrix is positive definite, compute the Cholesky lower triangular matrix
    if np.all(eigenvalues > 0):
        factorisation = True
        triangular_factor = np.linalg.cholesky(corr_matrix)

        return factorisation, triangular_factor
    else:
        return factorisation, None

# -------------------------------------------------------- #
#                   STOCHASTIC SIMULATION
# -------------------------------------------------------- #

def stochastic_evolution(data: pd.DataFrame, log_stats_moments: dict[str, tuple[float, float, float]],
                         cholesky_flag: bool, nw_flag: bool, stock: str = 'stock_sek_price', log_stock: str = 'log_sek_returns',
                         exchange: str = 'eur/sek', log_exchange: str = 'log_eur/sek') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        The function computes the stochastic evolution of the stock price, stock log-returns, and exchange rate

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            log_stats_moments (dict[str, tuple[float, float]]): Dictionary of tuples of the first two moments (annualised mean and annualised standard deviation)
                                                                for the log-returns of the sek stock price and exchange rate

            cholesky_flag (bool): Boolean flag that allows for cholesky factorisation

            nw_flag (bool): Boolean flag that allows for Newey_West standard deviation

            stock (str): Column name for the stock price, default is 'stock_price'

            log_stock (str): Column name for the stock log-returns, default is 'log_returns'

            exchange (str): Column name for the exchange rate, default is 'eur/sek'

            log_exchange (str): Column name for the exchange rate log-returns, default is 'log_eur/sek'

        Returns:

            tuple[np.ndarray, np.ndarray, np.ndarray]:

                - **stock_paths** (np.ndarray): Array of shape (n_simulations, trading_days) with simulated stock prices

                - **log_return_paths** (np.ndarray): Array of shape (n_simulations, trading_days) with simulated stock log-returns

                - **exchange_rate_paths** (np.ndarray): Array of shape (n_simulations, trading_days) with simulated exchange rates
        """

    # Fetch the log-returns moments for the stock price and exchange rate
    mu_stock, var_stock, nw_var_stock = log_stats_moments[f'{log_stock}_moments']
    mu_exchange, var_exchange, nw_var_exchange = log_stats_moments[f'{log_exchange}_moments']

    if nw_flag:
        sigma_stock = nw_var_stock
        sigma_exchange = nw_var_exchange
    else:
        sigma_stock = var_stock
        sigma_exchange = var_exchange


    # Retrieve the grant date values for the sek stoch price and exchange rate for starting the stochastic evolution
    grant_date_stock_price = data[stock].iloc[-1]
    grant_date_exchange_rate= data[exchange].iloc[-1]

    # Generate two gaussian distributed numbers to evolve both the stock price and the exchange rate
    normal_samples = np.random.normal(loc = 0, scale = 1, size = (n_simulations, trading_days, 2))

    # Apply Cholesky correlation if available, else keep independent
    cholesky_possible, cholesky_matrix = cholesky_factorisation(data = data)
    if cholesky_flag and cholesky_possible:
        samples = normal_samples @ cholesky_matrix.T
    else:
        samples = normal_samples

    # Daily multiplicative factors for both stock price and exchange rate
    stock_factors = np.exp((mu_stock - (sigma_stock ** 2) / 2) * dt + sigma_stock * np.sqrt(dt) * samples[:, :, 0])
    exchange_rate_factors = np.exp((mu_exchange - (sigma_exchange ** 2) / 2) * dt + sigma_exchange * np.sqrt(dt) * samples[:, :, 1])

    # Cumulative product to get stock price and exchange rate stochastic evolution
    stock_paths = grant_date_stock_price * np.cumprod(stock_factors, axis = 1)
    exchange_rate_paths = grant_date_exchange_rate * np.cumprod(exchange_rate_factors, axis = 1)

    # Daily log-returns for stochastic evolved stock price
    log_returns_paths = (mu_stock - ((sigma_stock ** 2) / 2)) * dt + sigma_stock * np.sqrt(dt) * normal_samples[:, :, 0]

    return stock_paths, log_returns_paths, exchange_rate_paths

# -------------------------------------------------------- #
#              PERFORMANCE SHARE PLAN STRUCTURE
# -------------------------------------------------------- #

def tranche_contribution(data: pd.DataFrame, stock_sek_prices_paths: np.ndarray, stock_sek_log_returns_paths: np.ndarray,
                         nw_flag: bool, stock: str = 'stock_sek_price') -> tuple[np.ndarray, np.ndarray]:
    """
        The function computes the tranche contribution arrays representing the percentage achievement of performance targets for each simulation

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            stock_sek_prices_paths (np.ndarray): Array of shape (n_simulations, trading_days) with simulated stock prices

            stock_sek_log_returns_paths (np.ndarray): Array of shape (n_simulations, trading_days) with simulated stock log-returns

            nw_flag (bool): Boolean flag that allows for Newey_West standard deviation

            stock (str): Column name for the stock price, default is 'stock_price'

        Returns:

            tuple[np.ndarray, np.ndarray]:

            - **tranche1** (np.ndarray): Array of shape (n_simulations, 1) with percentage values of target achievement for the first tranche (based on total shareholder return)

            - **tranche2** (np.ndarray): Array of shape (n_simulations, 1) with percentage values of target achievement for the second tranche (based on annualised volatility)
    """

    # Fetch the target values for the yearly returns and the volatility
    return_targets = targets['total_shareholder_return']['return_targets']
    volatility_targets = targets['annualised_volatility']['volatility_targets']

    # Fetch the latest stock price from historical data
    initial_stock_price = np.array([[data[stock].iloc[-1]]], dtype = float)

    # Calculate the annualised volatility for each of the simulations
    if nw_flag:
        annualised_std = nw_annualised_moments(data = stock_sek_log_returns_paths, lag = days, axis = 1, keepdims = True) * np.sqrt(trading_days)
    else:
        annualised_std = np.std(stock_sek_log_returns_paths, axis = 1, ddof = 1, keepdims = True) * np.sqrt(trading_days)

    # Calculate the yearly log-return from last observed to simulated final value
    yearly_return = stock_sek_prices_paths[:, -1].reshape(-1, 1) / initial_stock_price - 1

    # Calculate the percentage target achievement for first tranche (return-based (0, 0.05, 0.1)) for each simulation
    tranche1 = np.where(
        yearly_return <= return_targets[0], 0, np.where(
            yearly_return >= return_targets[2], 1, (yearly_return - return_targets[0]) / (return_targets[2] - return_targets[0]))
    )

    # Calculate the percentage target achievement for second tranche (volatility-based (0.5, 0.4, 0.3)) for each simulation
    tranche2 = np.where(
        annualised_std >= volatility_targets[0], 0, np.where(
            annualised_std <= volatility_targets[2], 1, (volatility_targets[0] - annualised_std) / (volatility_targets[0] - volatility_targets[2]))
    )

    return tranche1, tranche2


def trigger(stock_sek_prices_paths: np.ndarray, exchange_rate_paths: np.ndarray) -> np.ndarray:
    """
        The function computes the boolean trigger value for each simulation

        Arguments:

            stock_sek_prices_paths (np.ndarray): Array of shape (n_simulations, trading_days) with simulated sek stock prices

            exchange_rate_paths (np.ndarray): Array of shape (n_simulations, trading_days) with simulated exchange rates

        Returns:

             triggers (np.ndarray): Array of shape (n_simulations, 1) indicating whether the trigger condition (stock price >= 10) was met in each simulation
    """

    # Calculate the corresponding stock prices in EUR from the simulated one in SEK
    stock_eur_prices_paths = stock_sek_prices_paths / exchange_rate_paths

    # Calculate the trigger condition for each simulation
    triggers = np.any(stock_eur_prices_paths >= 10, axis = 1, keepdims = True)

    return triggers


def fair_value(data: pd.DataFrame, stock_sek_prices_paths: np.ndarray, tranches: tuple[np.ndarray, np.ndarray],
               triggers: np.ndarray, stock: str = 'stock_sek_price', exchange: str = 'eur/sek', yearly_rate: str = 'sek_1y_rate') -> dict[str, any]:
    """
        The function computes the expected fair value of the performance share plan (PSP) by combining simulation-based payout factors, trigger conditions,
        and discounting with the latest one year euro interest rate

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            stock_sek_prices_paths (np.ndarray): Array of shape (n_simulations, trading_days) with simulated stock prices

            tranches (tuple[np.ndarray, np.ndarray]):

                - Array of shape (n_simulations, 1) with percentage values of target achievement for the first tranche (based on total shareholder return)

                - Array of shape (n_simulations, 1) with percentage values of target achievement for the second tranche (based on annualised volatility)

            triggers (np.ndarray): Array of shape (n_simulations, 1) indicating whether the trigger condition (stock price >= 10) was met in each simulation

            stock (str): Column name for the sek stock price, default is 'stock_sek_price'

            exchange (str): Column name for the exchange rate, default is 'eur/sek'

            yearly_rate (str): Column name for the yearly sek interest rate, default is 'sek_1y_rate'

        Returns:

            dict[str, Any]: Dictionary containing the discounted expected fair values and payout factors:

                - **fair_value** (dict[str, float]): Dictionary containing the expected fair value per unit share under the non-path-dependent and path-dependent assumption

                - **payout_factors** (np.ndarray): Array of shape (n_simulations, 1) containing the payout factors computed across all simulations
    """

    # Unpack tranche contributions
    tranche1, tranche2 = tranches

    # Retrieve latest one year euro interest rate
    grant_sek_interest_rate = float(data[yearly_rate].iloc[-1] / 100)

    # Retrieve the grant date stock price from historical data
    grant_sek_stock_price = np.array([[data[stock].iloc[-1]]], dtype = float)

    # Retrieve the grant date exchange rae from historical data
    grant_exchange = np.array([[data[exchange].iloc[-1]]], dtype = float)

    # Calculate payout factors for each simulation
    payout_factors = (tranche1 * targets['total_shareholder_return']['weight'] + tranche2 * targets['annualised_volatility']['weight']) * triggers

    # Calculate the average payout factor across all simulation for no path dependent case
    non_path_mean_payout = grant_sek_stock_price * np.mean(payout_factors, axis = 0, keepdims = True)

    # Calculate the average payout factor across all simulations for path dependent case
    path_mean_payout = np.mean(payout_factors * stock_sek_prices_paths[:, -1].reshape(-1, 1), axis = 0, keepdims = True)

    # Discount factor
    discount_factor = np.exp(- (grant_sek_interest_rate * T))

    # Convert the unit share fair value to euro at grant date
    non_path_grant_fair_value = float(np.squeeze((non_path_mean_payout * discount_factor) / grant_exchange))
    path_grant_fair_value = float(np.squeeze((path_mean_payout * discount_factor) / grant_exchange))

    # Dictionary containing the discounted expected unit share fair value in euro and payout factors
    unit_fair_value = {
        'fair_value': {
            'non_path_dependent': non_path_grant_fair_value,
            'path_dependent': path_grant_fair_value
        },
        'payout_factors': payout_factors
    }

    return unit_fair_value

# -------------------------------------------------------- #
#                     DATAFRAME CREATION
# -------------------------------------------------------- #

@utils.timer
def fair_price_pd(data: pd.DataFrame, cholesky_flag: bool, nw_flag: bool) -> dict[str, any]:
    """
        The function computes the boolean trigger value for each simulation

        Arguments:

            data (pd.DataFrame): DataFrame containing the raw data as well as log-returns

            cholesky_flag (bool): Boolean flag that allows for cholesky factorisation

            nw_flag (bool): Boolean flag that allows for Newey_West standard deviation

        Returns:

             result (dict[str, any]): Array of shape (n_simulations, 1) indicating whether the trigger condition (stock price >= 10) was met in each simulation
    """

    # Compute the annualised moments
    statistic_values = annualised_moments(data = data)

    # Run stochastic evolution
    ft_stock_sek_prices, ft_sek_log_returns, ft_exchange_rate = stochastic_evolution(data = data, log_stats_moments = statistic_values, cholesky_flag = cholesky_flag, nw_flag = nw_flag)

    # Compute trigger
    boolean_trigger = trigger(stock_sek_prices_paths = ft_stock_sek_prices, exchange_rate_paths = ft_exchange_rate)

    # Compute both tranches contributions
    tranches = tranche_contribution(data = data, stock_sek_prices_paths = ft_stock_sek_prices, stock_sek_log_returns_paths = ft_sek_log_returns, nw_flag = nw_flag)

    print(np.sum((tranches[0] > 0) & (tranches[0] < 1)))

    # Compute fair value dictionary
    share_value_dict = fair_value(data = data, stock_sek_prices_paths = ft_stock_sek_prices, tranches = tranches, triggers = boolean_trigger)

    # Extract results
    unit_fair_value = share_value_dict['fair_value']
    payout_factors = share_value_dict['payout_factors']

    # Plot percentile of simulated final stock prices when both flag are True
    if cholesky_flag and nw_flag:

        # Dictionary containing simulated quantities
        stochastic_dict = {
            'stock price (SEK)': ft_stock_sek_prices,
            'stock price (EUR)': ft_stock_sek_prices / ft_exchange_rate,
            'tranche 1': tranches[0],
            'tranche 2': tranches[1],
            'trigger': boolean_trigger,
            'payouts': payout_factors
        }

        for key, quantity in stochastic_dict.items():
            if quantity.shape[1] == trading_days:
                percentiles = np.percentile(quantity[:, -1], [5, 50, 95])
                utils.percentile_plot(data = quantity[:, -1], percentiles = percentiles, plot_name = key)
            elif key in ['tranche 2', 'payouts']:
                utils.histogram_plot(data = quantity.reshape(-1), plot_name = key)
            #elif key == 'tranche 1':
                #utils.bar_plot(data = quantity.reshape(-1), plot_name = key)
            else:
                utils.pie_plot(data = np.ravel(quantity), plot_name = key)

    # Append results to list for DataFrame
    result = {
        'fv_eur_non_path': f'{unit_fair_value['non_path_dependent']:.2f}',
        'fv_eur_path': f'{unit_fair_value['path_dependent']:.2f}',
        'mean_tot_payout': f'{np.mean(payout_factors) * 100:.2f}%',
        'prob_tr1_max': f'{np.mean(tranches[0] == 1) * 100:.2f}%',
        'prob_tr2_max': f'{np.mean(tranches[1] == 1) * 100:.2f}%',
        'prob_trig': f'{np.mean(boolean_trigger) * 100:.2f}%',
        'mean_payout_tr1': f'{np.mean(tranches[0]) * 100:.2f}%',
        'mean_payout_tr2': f'{np.mean(tranches[1]) * 100:.2f}%',
        'cholesky': cholesky_flag,
        'newey_west': nw_flag
    }

    return result

def main() -> None:
    """
        Entry point for the script. Executes the main workflow.
    """

    # Read Excel file and cast it into a DataFrame
    aurora_metals: pd.DataFrame = pd.read_excel('aurora_metals_data.xlsx')

    # Sorted DataFrame with columns for log-returns and log-exchange rate values
    extended_data = log_data(data = aurora_metals)

    print(extended_data)

    # List containing the results for each combination of the flags
    combinations = []

    # Run all combinations
    for cholesky_flag in [False, True]:
        for nw_flag in [False, True]:
            combinations.append(fair_price_pd(data = extended_data, cholesky_flag = cholesky_flag, nw_flag = nw_flag))

    dataframe = pd.DataFrame(combinations)

    # Print the DataFrame in tabular form
    print(f'\n{tabulate(dataframe, headers = 'keys', tablefmt = 'grid', stralign = 'center', showindex = False)}')

if __name__ == '__main__':
    main()