"""
utils.py
-----------------------------------------------------------------------
Collections of custom functions for the script performance_share_plan.py

Dependencies
-----------------------------------------------------------------------
-numpy
- pandas
- seaborn
- matplotlib

Usage
-----------------------------------------------------------------------
$ python utils.py

Author
-----------------------------------------------------------------------
Riccardo Nicolò Iorio

Date
-----------------------------------------------------------------------
2025-10-12
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# -------------------------------------------------------- #
# v1.0  10-12-2025  First Version
# v1.1  10-20-2025  Second Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
import time                                         # Time
from functools import wraps                         # Wrapper

# Third-party libraries
import numpy as np                                  # Numerical computation
import pandas as pd                                 # Data manipulation
import matplotlib.pyplot as plt                     # Visualisation
import seaborn as sns                               # Visualisation

# -------------------------------------------------------- #
#                       WRAPPERS
# -------------------------------------------------------- #

def timer(func: callable) -> callable:
    """
        Decorator that measures the execution time of a function

        Parameters:
            func: **(callable)**
                Function to be timed

        Returns:
            **(callable)**:
                A wrapped version of the input function that measures and reports its execution time when called
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
#                      BASH FORMATTING
# -------------------------------------------------------- #

def percent(data: np.ndarray | bool) -> str:
    """
        The function returns the mean of the data as a percentage string with 2 decimals

        Parameters:
            data: **(np.ndarray | bool)**
                Array of data values to format

        Returns:
            **(str)**:
                Formatted data as percentage
    """

    return f"{np.mean(data) * 100:.3f}%"

def mean_ci(mean: float, ste: float, z: float = 1.96) -> str:
    """
        The function returns the mean of the data with a confidence interval of 95%

        Parameters:
            mean: **(np.ndarray)**
                Mean value of the data

            ste: **(np.ndarray)**
                Standard error of the data

            z: **(float)**
                Critical value for the confidence interval, default 1.96 for 95% CI

        Returns:
            **(str)**:
                Formatted string as "mean ± confidence interval"
    """

    return f"{mean:.3f} ± {z * ste:.3f}"

# -------------------------------------------------------- #
#                       PLOTTING
# -------------------------------------------------------- #

def kde_stock_plot(data: dict[str, np.ndarray], plot_name: str, x_axis: bool) -> None:
    """
        The function generates a plot showing the probability density distribution for multiple groups of data

        Parameters:
            data: **(np.ndarray)**
                Array of stock data values to visualize

            plot_name: **(str)**
                Name of the plot file to save

            x_axis: **(bool)**
                Flag to select whether to format the x-axis

        Returns:
            **None**
    """

    # Create a DataFrame from the input dictionary, flattening each array
    data_dataframe = pd.DataFrame({day: arr.ravel() for day, arr in data.items()})

    # Reshape the DataFrame to a long format for easier plotting with Seaborn
    lag_dataframe = data_dataframe.melt(value_vars = data_dataframe.columns, var_name = 'lag', value_name = 'values')

    # Randomly sample 10,000 data points for each lag category to reduce dataset size
    sampled_data = (
        lag_dataframe
        .groupby(by = 'lag')
        .sample(n = int(1e4), random_state = 42)
    )

    # Define the size of the plot figure
    plt.figure(figsize = (12, 6))

    # Define palette of color based on the number of lag days in the DataFrame
    palette = sns.color_palette("viridis", n_colors = sampled_data['lag'].nunique())

    # KDE probability density distribution of the data values
    sns.kdeplot(data = sampled_data, x = 'values', hue = 'lag', fill = True, linewidth = 0, palette = palette, alpha = 0.3, common_norm = False)

    # Apply plot label and title
    plt.title(f"{plot_name.title()} KDE Distribution", fontsize = 11, fontweight = 'bold', color = 'black')
    plt.xlabel(f"{plot_name.title()}", fontsize = 10, fontweight = 'bold', color = 'black')
    plt.ylabel('Density', fontsize = 10, fontweight = 'bold', color = 'black')

    # Disable grid
    plt.grid(False)

    if x_axis:
        # Get the current axes instance to customize the plot
        ax = plt.gca()

        # Move x-axis to y = 0
        ax.spines['bottom'].set_position(('data', 0))

        # Move y-axis to x = 0
        ax.spines['left'].set_position(('data', 0))

        # Avoid negative padding
        plt.xlim(left = 0)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig(f"kde_distribution_{plot_name}.jpg", dpi = 500)
    plt.close()
