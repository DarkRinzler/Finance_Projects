from typing import Any, Dict

import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
