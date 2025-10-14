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

def percentile_plot(data: np.ndarray, percentiles: np.ndarray, plot_name: str) -> None:
    """
        Decorator that measures the execution time of a function.

        Arguments:

            data (np.ndarray): Function to be timed

            percentiles (np.ndarray):

            plot_name (str):

        Returns:

            None
    """

    # Unpack to get percentiles
    per5, per50, per95 = percentiles

    # Define the size of the plot figure
    plt.figure(figsize = (12, 6))

    # Histogram of the data values
    plt.hist(data, bins = 400, histtype = 'bar', color = 'lightsteelblue')

    # Add vertical line for each percentile spanning the whole figure
    plt.axvline(per5, color = 'orangered', linestyle = '--', linewidth = 1, label = '5th percentile')
    plt.axvline(per50, color = 'khaki', linestyle = '--', linewidth = 1, label = '50th percentile')
    plt.axvline(per95, color = 'forestgreen', linestyle = '--', linewidth = 1, label = '95th percentile')

    # Apply plot label and title
    plt.title(f'Final {plot_name.title()} Distribution', fontsize = 11, fontweight = 'bold', color = 'black')
    plt.xlabel(f'{plot_name.title()}', fontsize = 10, fontweight = 'bold', color = 'black')
    plt.ylabel('Frequency', fontsize = 10, fontweight = 'bold', color = 'black')
    plt.legend(loc = 'upper right')

    # Disable grid
    plt.grid(False)

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
    plt.savefig(f'Distribution {plot_name.title()}.jpg', dpi = 500)
    plt.close()

def histogram_plot(data: np.ndarray, plot_name: str) -> None:
    """
        Decorator that measures the execution time of a function.

        Arguments:

            data (np.ndarray): Function to be timed

            plot_name (str):

        Returns:

            None
    """

    # Define the size of the plot figure
    plt.figure(figsize = (12, 6))

    # Histogram of the data values
    plt.hist(data, bins = 400, histtype = 'bar', color = 'lightsteelblue')

    # Apply plot label and title
    plt.title(f'{plot_name.title()} Distribution', fontsize = 11, fontweight = 'bold', color = 'black')
    plt.xlabel(f'{plot_name.title()}', fontsize = 10, fontweight = 'bold', color = 'black')
    plt.ylabel('Frequency', fontsize = 10, fontweight = 'bold', color = 'black')

    # Disable grid
    plt.grid(False)

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig(f'Histogram {plot_name.title()}.jpg', dpi = 500)
    plt.close()

def bar_plot(data: np.ndarray, plot_name: str) -> None:
    """
        Decorator that measures the execution time of a function.

        Arguments:

            data (np.ndarray): Function to be timed

            plot_name (str):

        Returns:

            None
    """

    # Define the size of the plot figure
    plt.figure(figsize=(12, 6))

    # Find unique elements and their counts
    values, counts = np.unique(data, return_counts = True)

    # Calculate the percentage of each respective unique element
    proportions = counts / np.sum(counts)

    # Bar plot the data values
    plt.bar(values, proportions, width = 0.2, color = 'lightsteelblue', edgecolor = 'black')

    # Apply plot label and title
    plt.title(f'{plot_name.title()} Distribution', fontsize = 11, fontweight = 'bold', color = 'black')
    plt.xlabel(f'{plot_name.title()}', fontsize = 10, fontweight='bold', color = 'black')
    plt.ylabel('Probability', fontsize = 10, fontweight = 'bold', color='black')

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig(f'Percentages {plot_name.title()}.jpg', dpi = 500)
    plt.close()

def pie_plot(data: np.ndarray, plot_name: str) -> None:

    labels = ['No payout', 'Partial payout', 'Full payout']
    counts = [
        np.sum(data == 0),
        np.sum((data > 0) & (data < 1)),
        np.sum(data == 1)
    ]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'])

    # Use tight_layout to adjust the spacing and center the plot
    plt.tight_layout()

    # Save plot as an image (JPG format)
    plt.savefig(f'Pie {plot_name.title()}.jpg', dpi=500)
    plt.close()