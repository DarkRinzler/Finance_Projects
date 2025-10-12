"""
utils.py
-----------------------------------------------------------------------
Collections of custom functions for the script performance_share_plan.py

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
2025-10-12
"""

# -------------------------------------------------------- #
#                       SCRIPT VERSION
# -------------------------------------------------------- #

# Version History
# -------------------------------------------------------- #
# v1.0  2025-10-12  First Version

# -------------------------------------------------------- #
#                       LIBRARIES
# -------------------------------------------------------- #

# Standard libraries
import time                                         # Time
from functools import wraps                         # Wrapper

# Third-party libraries
import numpy as np                                  # Numerical computation
import matplotlib.pyplot as plt                     # Visualisation
from matplotlib.ticker import MultipleLocator       # Ticker locator


# -------------------------------------------------------- #
#                       WRAPPERS
# -------------------------------------------------------- #

def timer(func: callable) -> callable:
    """
        Decorator that measures the execution time of a function.

        Arguments:

            func (callable): Function to be timed

        Returns:

            callable: A wrapped version of the input function that measures and reports its execution time when called
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
#                       PLOTTING
# -------------------------------------------------------- #

def percentile_plot(data: np.ndarray, percentiles: np.ndarray) -> None:
    """
            Decorator that measures the execution time of a function.

            Arguments:

                data (np.ndarray): Function to be timed

                percentiles (np.ndarray):

            Returns:

                None
    """

    # Unpack to get percentiles
    per5, per50, per95 = percentiles

    # Define the size of the plot figure
    plt.figure(figsize = (12, 6))

    # Histogram of the data value
    plt.hist(data, bins = 400, histtype = 'bar', color = 'lightsteelblue')

    # Add vertical line for each percentile spanning the whole figure
    plt.axvline(per5, color = 'orangered', linestyle = '--', linewidth = 1, label = '5th percentile')
    plt.axvline(per50, color = 'khaki', linestyle = '--', linewidth = 1, label = '50th percentile')
    plt.axvline(per95, color = 'forestgreen', linestyle = '--', linewidth = 1, label = '95th percentile')

    # Apply plot label and title
    plt.title('Final Stock Price Distribution', fontsize = 11, fontweight = 'bold', color = 'black')
    plt.xlabel('Stock Price', fontsize = 10, fontweight = 'bold', color = 'black')
    plt.ylabel('Frequency', fontsize = 10, fontweight = 'bold', color = 'black')
    plt.legend(loc = 'upper right')

    # Disable grid
    plt.grid(False)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))

    # Move x-axis to y = 0
    ax.spines['bottom'].set_position(('data', 0))

    # Move y-axis to x = 0
    ax.spines['left'].set_position(('data', 0))

    # Avoid negative padding
    plt.xlim(left = 0)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig('investment_forecast.jpg', dpi = 500)
    plt.close()
