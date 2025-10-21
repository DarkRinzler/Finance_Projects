"""
performance_share_plan.py
-----------------------------------------------------------------------
Estimate the

Inputs
-----------------------------------------------------------------------
- aurora_metals_data (Excel file): Raw company dataset containing historical stock prices, exchange rates, and interest rates

Process
-----------------------------------------------------------------------
- Evolve stochastically quantity of interest through the use of Cholesky decomposition and Newey-West method over a 1-year horizon
- Calculate the fair value of the share plan (Monte Carlo approach)
- Generate plots for different lag days

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
10-07-2025
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# -------------------------------------------------------- #
# v1.0  07-10-2025  Initial Version
# v1.1  10-10-2025  Second Version
# v1.2  17-10-2025  Third Version
# v1.3  20-10-2025  Fourth Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
import sys

# Third-party libraries
import numpy as np                      # Numerical computation
import pandas as pd                     # Data manipulation
from itertools import product           # Cartesian product
from tabulate import tabulate           # Visualisation (bash)
from tqdm import tqdm                   # Loop progress


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

# Minimum and maximum days lag
min_lag: int = 1
max_lag: int = 7

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

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data

        Returns:
            enriched_data: **(pd.DataFrame)**
                DataFrame of shape (n_rows, n_cols + 2) containing the original company data with additional columns for log-returns of the stock price and the exchange rate
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

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            lag: **(int)**
                Number of autocorrelation lags to include in the Newey–West correction

            axis: **(int)**
                Axis along which to compute the statistic

            keepdims: **(bool)**
                Whether to keep the reduced dimensions in the output

        Returns:
            nw_var: **(np.ndarray)**
                Newey–West adjusted standard deviation of the input data, computed along the specified axis
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


def annualised_moments(data: pd.DataFrame, days: int, arbitrage: bool, risk_free: str = 'sek_1y_rate', log_stock: str = 'log_sek_returns', log_exchange: str = 'log_eur/sek') -> dict[str, tuple[float, float, float]]:
    """
        The function computes the annualised first two moments (mean and standard deviation) of log-returns for the company stock price and exchange rate

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            days: **(int)**
                Number of days to correlate in the Newey-West estimator

            arbitrage: **(bool)**
                Flag that selects if arbitrage free conditions are assumed

            risk_free: **(str)** = 'sek_1y_rate'
                Column name for the sek 1 year risk freer rate, default is 'sek_1y_rate'

            log_stock: **(str)**
                Column name for the stock log-returns, default is 'log_returns'

            log_exchange: **(str)**
                Column name for the exchange rate log-returns, default is 'log_eur/sek'

        Returns:
            log_returns_stats: **(dict[str, tuple[float, float]])**:
                Dictionary mapping each series to a tuple of its first two moments (annualised mean and annualised standard deviation), for the log-returns of the stock price and the exchange rate
    """

    # Columns to compute statistics on
    stats_columns = [log_stock, log_exchange]

    # Calculate annualised first two moments (mean and standard deviation)
    if arbitrage:
        mean_annual = data[stats_columns].mean(skipna = True) * trading_days
    else:
        mean_annual =  np.log(1 + data[risk_free].iloc[-1] / 100)

    std_annual = data[stats_columns].std(skipna = True, ddof = 1) * np.sqrt(trading_days)

    nw_std_annual = nw_annualised_moments(data = data[stats_columns].dropna().to_numpy(), lag = days, axis = 0, keepdims = False) * np.sqrt(trading_days)

    # DataFrame containing annualised statistics
    annualised_stats = pd.DataFrame({
            'mean': mean_annual,
            'std': std_annual,
            'nw_std' : nw_std_annual
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

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            stock: **(str)**
                Column name for the stock price, default is 'stock_price'

            exchange: **(str)**
                Column name for the exchange rate, default is 'eur/sek'

        Returns:
            **(tuple[bool, np.ndarray | None])**:
                Tuple containing the factorisation result and the corresponding Cholesky decomposition
                - **factorisation** **(bool)**: True if the correlation matrix is positive definite and Cholesky factorisation is possible, False otherwise
                - **triangular_factor** **(np.ndarray | None)**: Lower-triangular matrix from the Cholesky factorisation if possible, None otherwise
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
                         ch_flag: bool, nw_flag: bool, stock: str = 'stock_sek_price', log_stock: str = 'log_sek_returns',
                         exchange: str = 'eur/sek', log_exchange: str = 'log_eur/sek') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        The function computes the stochastic evolution of the stock price, stock log-returns, and exchange rate

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            log_stats_moments: **(dict[str, tuple[float, float]])**
                Dictionary of tuples of the first two moments (annualised mean and annualised standard deviation) for the log-returns of the sek stock price and exchange rate

            ch_flag: **(bool)**
                Boolean flag that allows for cholesky factorisation

            nw_flag: **(bool)**
                Boolean flag that allows for Newey_West standard deviation

            stock: **(str)**
                Column name for the stock price, default is 'stock_price'

            log_stock: **(str)**
                Column name for the stock log-returns, default is 'log_returns'

            exchange: **(str)**
                Column name for the exchange rate, default is 'eur/sek'

            log_exchange: **(str)**
                Column name for the exchange rate log-returns, default is 'log_eur/sek'

        Returns:
            **(tuple[np.ndarray, np.ndarray, np.ndarray])**:
                Tuple containing the stochastic evolution of the stock price, stock log-returns, and exchange rate
                - **stock_paths** **(np.ndarray)**: Array of shape (n_simulations, trading_days) with simulated stock prices
                - **log_return_paths** **(np.ndarray)**: Array of shape (n_simulations, trading_days) with simulated stock log-returns
                - **exchange_rate_paths** **(np.ndarray)**: Array of shape (n_simulations, trading_days) with simulated exchange rates
        """

    # Fetch the annualised log-returns moments for the stock price and exchange rate
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
    if ch_flag and cholesky_possible:
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
                         days: int, nw_flag: bool, stock: str = 'stock_sek_price') -> tuple[np.ndarray, np.ndarray]:
    """
        The function computes the tranche contribution arrays representing the percentage achievement of performance targets for each simulation

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            stock_sek_prices_paths: **(np.ndarray)**
                Array of shape (n_simulations, trading_days) with simulated stock prices

            stock_sek_log_returns_paths: **(np.ndarray)**
                Array of shape (n_simulations, trading_days) with simulated stock log-returns

            days: **(int)**
                Number of days to extend the autocorrelation using the Newey–West method

            nw_flag: **(bool)**
                Boolean flag that allows for Newey_West standard deviation

            stock: **(str)**
                Column name for the stock price, default is 'stock_price'

        Returns:
            **(tuple[np.ndarray, np.ndarray])**:
                Tuple containing the tranche contribution arrays representing the percentage achievement of performance targets for each simulation
                - **tranche1** **(np.ndarray)**: Array of shape (n_simulations, 1) with percentage values of target achievement for the first tranche (based on total shareholder return)
                - **tranche2** **(np.ndarray)**: Array of shape (n_simulations, 1) with percentage values of target achievement for the second tranche (based on annualised volatility)
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

        Parameters:
            stock_sek_prices_paths: **(np.ndarray)**
                Array of shape (n_simulations, trading_days) with simulated sek stock prices

            exchange_rate_paths: **(np.ndarray)**
                Array of shape (n_simulations, trading_days) with simulated exchange rates

        Returns:
             triggers: **(np.ndarray)**
                Array of shape (n_simulations, 1) indicating whether the trigger condition (stock price >= 10) was met in each simulation
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

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            stock_sek_prices_paths: **(np.ndarray)**
                Array of shape (n_simulations, trading_days) with simulated stock prices

            tranches: **(tuple[np.ndarray, np.ndarray])**
                Tuple containing the tranche contribution arrays representing the percentage achievement of performance targets for each simulation
                - Array of shape (n_simulations, 1) with percentage values of target achievement for the first tranche (based on total shareholder return)
                - Array of shape (n_simulations, 1) with percentage values of target achievement for the second tranche (based on annualised volatility)

            triggers: **(np.ndarray)**
                Array of shape (n_simulations, 1) indicating whether the trigger condition (stock price >= 10) was met in each simulation

            stock: **(str)**
                Column name for the sek stock price, default is 'stock_sek_price'

            exchange: **(str)**
                Column name for the exchange rate, default is 'eur/sek'

            yearly_rate: **(str)**
                Column name for the yearly sek interest rate, default is 'sek_1y_rate'

        Returns:
            **(dict[str, Any])**:
                Dictionary containing the discounted expected fair values and payout factors
                - **fair_value** **(dict[str, float])**: Dictionary containing the expected fair value per unit share under the non-path-dependent and path-dependent assumption
                - **payout_factors** **(np.ndarray)**: Array of shape (n_simulations, 1) containing the payout factors computed across all simulations
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

    # Calculate the average and standard deviation of the payout factors across all simulation for no path dependent case
    non_path_mean_payout = grant_sek_stock_price * np.mean(payout_factors, axis = 0, keepdims = True)
    non_path_std_payout = grant_sek_stock_price * np.std(payout_factors, axis = 0, ddof = 1, keepdims = True)

    # Calculate the average and standard deviation of the payout factors across all simulations for path dependent case
    path_mean_payout = np.mean(payout_factors * stock_sek_prices_paths[:, -1].reshape(-1, 1), axis = 0, keepdims = True)
    path_std_payout = np.std(payout_factors * stock_sek_prices_paths[:, -1].reshape(-1, 1), axis = 0, ddof = 1, keepdims = True)

    # Discount factor
    discount_factor = np.exp(- (grant_sek_interest_rate * T))

    # Convert the unit share fair value to euro at grant date
    non_path_fv = float(np.squeeze((non_path_mean_payout * discount_factor) / grant_exchange))
    path_fv = float(np.squeeze((path_mean_payout * discount_factor) / grant_exchange))

    # Convert the standard deviation of unit share fair value to euro at grant date
    non_path_std_fv = float(np.squeeze((non_path_std_payout * discount_factor) / grant_exchange))
    path_std_fv = float(np.squeeze((path_std_payout * discount_factor) / grant_exchange))

    # Dictionary containing the discounted expected unit share fair value in euro and payout factors
    unit_fair_value = {
        'fair_value': {
            'non_path_dependent': {
                'mean': non_path_fv,
                'standard_error': non_path_std_fv / np.sqrt(n_simulations)
            },
            'path_dependent': {
                'mean': path_fv,
                'standard_error': path_std_fv / np.sqrt(n_simulations)
            }
        },
        'payout_factors': payout_factors
    }

    return unit_fair_value

# -------------------------------------------------------- #
#                     DATAFRAME CREATION
# -------------------------------------------------------- #

#@utils.timer
def fair_price_pd(data: pd.DataFrame, days: int, ch_flag: bool, nw_flag: bool, arbitrage: bool) -> tuple[dict[str, any], dict[str, any], dict[str, any]]:
    """
        The function computes the boolean trigger value for each simulation

        Parameters:
            data: **(pd.DataFrame)**
                DataFrame containing the raw data as well as log-returns

            days: **(int)**
                Number of days to extend the autocorrelation using the Newey–West method

            ch_flag: **(bool)**
                Boolean flag that allows for cholesky factorisation

            nw_flag: **(bool)*
                Boolean flag that allows for Newey_West standard deviation

            arbitrage: **(bool)**
                Flag that selects if arbitrage free conditions are assumed

        Returns:
            **(tuple[dict[str, any], dict[str, any], dict[str, any]])**:
                Tuple containing three dictionaries with the simulation results used for reporting and plotting
                - **bash_data** **(dict[str, any])**: Contains fair value estimates (path-dependent and non-path-dependent), mean total payout, and probabilities of maximum tranche achievement and trigger activation
                - **bash_flags** **(dict[str, any])**: Contains simulation metrics when Cholesky decomposition and the Newey–West method are applied for console output
                - **plot_flags** **(dict[str, any])**: Contains simulation results used for generating plots of stock prices, tranches, payouts, and trigger activations for both arbitrage and non-arbitrage case
    """

    # Define dictionaries for storing quantities of interest
    plot_flags, bash_flags = {}, {}

    # Compute the annualised moments
    statistic_values = annualised_moments(data = data, days = days, arbitrage = arbitrage)

    # Run stochastic evolution
    ft_stock_sek_prices, ft_sek_log_returns, ft_exchange_rate = stochastic_evolution(data = data, log_stats_moments = statistic_values, ch_flag = ch_flag, nw_flag = nw_flag)

    # Compute trigger
    boolean_trigger = trigger(stock_sek_prices_paths = ft_stock_sek_prices, exchange_rate_paths = ft_exchange_rate)

    # Compute both tranches contributions
    tranches = tranche_contribution(data = data, stock_sek_prices_paths = ft_stock_sek_prices, stock_sek_log_returns_paths = ft_sek_log_returns, days = days, nw_flag = nw_flag)

    # Compute fair value dictionary
    share_value_dict = fair_value(data = data, stock_sek_prices_paths = ft_stock_sek_prices, tranches = tranches, triggers = boolean_trigger)

    # Extract results
    unit_fair_value = share_value_dict['fair_value']
    payout_factors = share_value_dict['payout_factors']

    # Store stochastic quantities when both flags are True
    if ch_flag and nw_flag:

        # Dictionary containing simulated quantities for plotting
        plot_flags = {
            'stock_price_(sek)': ft_stock_sek_prices[:, -1],
            'stock_price_(eur)': ft_stock_sek_prices[:, -1] / ft_exchange_rate[:, -1],
            'tranche_1': tranches[0],
            'tranche_2': tranches[1],
            'payouts': payout_factors,
            'trigger': boolean_trigger
        }

        # Dictionary containing simulated quantities for bash output
        bash_flags = {
            'fv_eur_no_path': utils.mean_ci(unit_fair_value['non_path_dependent']['mean'], unit_fair_value['non_path_dependent']['standard_error']),
            'fv_eur_path': utils.mean_ci(unit_fair_value['path_dependent']['mean'], unit_fair_value['path_dependent']['standard_error']),
            'mean_tot_payout': utils.percent(payout_factors),
            'prob_tr1_max': utils.percent(tranches[0] == 1),
            'prob_tr2_max': utils.percent(tranches[1] == 1),
            'prob_trig': utils.percent(boolean_trigger),
            'mean_payout_tr1': utils.percent(tranches[0]),
            'mean_payout_tr2': utils.percent(tranches[1]),
            'lag_days': days,
            'arbitrage': arbitrage
        }

    # Results for all flag combinations (for DataFrame)
    bash_data = {
        'fv_eur_no_path': utils.mean_ci(unit_fair_value['non_path_dependent']['mean'], unit_fair_value['non_path_dependent']['standard_error']),
        'fv_eur_path': utils.mean_ci(unit_fair_value['path_dependent']['mean'], unit_fair_value['path_dependent']['standard_error']),
        'mean_tot_payout': utils.percent(payout_factors),
        'prob_tr1_max': utils.percent(tranches[0] == 1),
        'prob_tr2_max': utils.percent(tranches[1] == 1),
        'prob_trig': utils.percent(boolean_trigger),
        'mean_payout_tr1': utils.percent(tranches[0]),
        'mean_payout_tr2': utils.percent(tranches[1]),
        'cholesky': ch_flag,
        'newey_west': nw_flag,
        'lag_days': days,
        'arbitrage': arbitrage
    }

    return bash_data, bash_flags, plot_flags

def main() -> None:
    """
        Entry point for the script. Executes the main workflow.
    """

    # Stores simulation results when both flags are True for bash output
    best_bash_results, all_combinations = [], []

    # Stores simulation results when both flags are True for plotting through cases for each quantity of interest for arbitrage and not
    quantities = {name : {True: {}, False: {}} for name in ['stock_sek', 'stock_eur', 'tranche_1', 'tranche_2', 'payouts', 'triggers']}

    # Read Excel file and cast it into a DataFrame
    aurora_metals: pd.DataFrame = pd.read_excel('aurora_metals_data.xlsx')

    # Sorted DataFrame with columns for log-returns and log-exchange rate values
    extended_data = log_data(data = aurora_metals)

    # Arbitrage combinations
    arbitrage_combinations = product([False, True], [lag for lag in range(min_lag, max_lag + 1, 2)])

    # Run all combinations of flags and days lag
    for arbitrage, days in arbitrage_combinations:

        header = f" Simulation results for {days} days correlation"
        print(f"\n{header:-^{150}}")

        # Stores simulation results for all flag combinations for bash output
        all_combinations = []

        # 4 flag combinations
        flag_combinations = product([False, True], [False, True])

        # Print progress bar when looping over combinations of flags
        for ch_flag, nw_flag in tqdm(flag_combinations, total = 4, desc = 'Progress', dynamic_ncols = True, colour = 'green', file = sys.stdout):

            daily_results, best_results, plot_data = fair_price_pd(data = extended_data, days = days, ch_flag = ch_flag, nw_flag = nw_flag, arbitrage = arbitrage)

            # Append daily results
            all_combinations.append(daily_results)

            if best_results:
                # Append best results
                best_bash_results.append(best_results)

                # Append plot data for each quantity for arbitrage conditions
                quantities['stock_sek'][arbitrage][f'{days}_day'] = plot_data['stock_price_(sek)']
                quantities['stock_eur'][arbitrage][f'{days}_day'] = plot_data['stock_price_(eur)']
                quantities['tranche_1'][arbitrage][f'{days}_day'] = plot_data['tranche_1']
                quantities['tranche_2'][arbitrage][f'{days}_day'] = plot_data['tranche_2']
                quantities['payouts'][arbitrage][f'{days}_day'] = plot_data['payouts']
                quantities['triggers'][arbitrage][f'{days}_day'] = plot_data['trigger']

        # Create DataFrame of all simulations results for all flag values
        all_data_df = pd.DataFrame(all_combinations)

        # Print the DataFrame in tabular form
        print(f"\n{tabulate(all_data_df, headers = 'keys', tablefmt = 'grid', stralign = 'center', showindex = False)}")

    header = " Simulation results for Cholesky decomposition and Newey-West method employed "
    print(f"\n{header:-^{150}}")

    # Create DataFrame of all simulations results when Cholesky decomposition and the Newey–West method are applied
    best_data_df = pd.DataFrame(best_bash_results)

    # Print the DataFrame in tabular form
    print(f"\n{tabulate(best_data_df, headers = 'keys', tablefmt = 'grid', stralign = 'center', showindex = False)}")

    # Dictionary where each quantity stores its full set of lag-dependent simulation data
    plot_collections = {
        "stock_(sek)": quantities['stock_sek'],
        "stock_(eur)": quantities['stock_eur'],
        "tranche_1": quantities['tranche_1'],
        "tranche_2": quantities['tranche_2'],
        "payouts": quantities['payouts'],
        "triggers": quantities['triggers']
    }

    # Save the kde plots showing the probability density distribution for multiple groups of data
    for plot_name, plot_dict_values in plot_collections.items():
        stock_plot = plot_name in ['stock_(sek)', 'stock_(eur)']

        for arbitrage, plot_values in plot_dict_values.items():

            title_suffix = "arbitrage" if arbitrage else "no_arbitrage"
            full_plot_name = f"{plot_name}_{title_suffix}"

            utils.kde_stock_plot(data = plot_values, plot_name = full_plot_name, x_axis = stock_plot)

if __name__ == '__main__':
    main()