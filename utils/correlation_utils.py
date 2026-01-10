"""
Correlation Analysis Utilities
Basic mathematical and statistical methods for correlation analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_growth_months_string(months_str):
    """
    Parse growth months string to list of integers

    Args:
        months_str: String like "1,2,3" or "1-3" or single month "1"

    Returns:
        list: List of integer months
    """
    if pd.isna(months_str) or months_str in ['-', '', 'nan']:
        return []

    months_str = str(months_str).strip()

    # Handle comma-separated
    if ',' in months_str:
        return [int(m.strip()) for m in months_str.split(',') if m.strip().isdigit()]

    # Handle range format (e.g., "1-3")
    elif '-' in months_str:
        try:
            start, end = map(int, months_str.split('-'))
            return list(range(start, end + 1))
        except:
            return []

    # Handle single month
    else:
        try:
            return [int(months_str)]
        except:
            return []


def filter_by_growing_months(data_df, state, growth_months, state_col='State', month_col='Month'):
    """
    Filter data for specific state and growth months

    Args:
        data_df: DataFrame containing data
        state: State name
        growth_months: List of growth months
        state_col: Column name for state
        month_col: Column name for month

    Returns:
        DataFrame: Filtered data
    """
    if not growth_months:
        return pd.DataFrame()

    return data_df[(data_df[state_col] == state) &
                   (data_df[month_col].isin(growth_months))].copy()


def aggregate_annual_spi(spi_data, state_col='State', year_col='Year',
                         spi_col='SPI_3', precip_col='monthly_precip'):
    """
    Aggregate monthly SPI data to annual statistics

    Args:
        spi_data: Monthly SPI DataFrame
        state_col: Column name for state
        year_col: Column name for year
        spi_col: Column name for SPI values
        precip_col: Column name for precipitation

    Returns:
        DataFrame: Annual SPI statistics
    """
    # Group by state and year
    grouped = spi_data.groupby([state_col, year_col])

    # Calculate statistics
    agg_dict = {
        spi_col: ['mean', 'std', 'min', 'max', 'count'],
    }

    if precip_col in spi_data.columns:
        agg_dict[precip_col] = ['mean', 'sum']

    aggregated = grouped.agg(agg_dict).reset_index()

    # Flatten column names
    if precip_col in spi_data.columns:
        aggregated.columns = [
            state_col, year_col,
            'SPI_mean', 'SPI_std', 'SPI_min', 'SPI_max', 'SPI_months',
            'Precip_mean', 'Precip_sum'
        ]
    else:
        aggregated.columns = [
            state_col, year_col,
            'SPI_mean', 'SPI_std', 'SPI_min', 'SPI_max', 'SPI_months'
        ]

    return aggregated


def calculate_pearson_correlation(x, y):
    """
    Calculate Pearson correlation coefficient and p-value

    Args:
        x: Array-like data
        y: Array-like data

    Returns:
        tuple: (correlation_coefficient, p_value)
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    try:
        return pearsonr(x, y)
    except Exception:
        return np.nan, np.nan


def calculate_spearman_correlation(x, y):
    """
    Calculate Spearman correlation coefficient and p-value

    Args:
        x: Array-like data
        y: Array-like data

    Returns:
        tuple: (correlation_coefficient, p_value)
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    try:
        return spearmanr(x, y)
    except Exception:
        return np.nan, np.nan


def determine_significance_level(p_value):
    """
    Determine significance level marker based on p-value

    Args:
        p_value: p-value from correlation test

    Returns:
        str: Significance marker ('***', '**', '*', 'ns')
    """
    if pd.isna(p_value):
        return 'ns'

    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def validate_correlation_data(x, y, min_samples=3):
    """
    Validate data for correlation analysis

    Args:
        x: First variable data
        y: Second variable data
        min_samples: Minimum number of samples required

    Returns:
        bool: True if data is valid for correlation analysis
    """
    # Check sample size
    if len(x) < min_samples or len(y) < min_samples:
        return False

    # Check for all NaN or constant values
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return False

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return False

    return True


def calculate_summary_statistics(correlation_results):
    """
    Calculate summary statistics from correlation results

    Args:
        correlation_results: DataFrame with correlation results

    Returns:
        dict: Summary statistics
    """
    if len(correlation_results) == 0:
        return {}

    summary = {
        'total_states': len(correlation_results),
        'pearson_significant': len(correlation_results[correlation_results['Pearson_sig'] != 'ns']),
        'spearman_significant': len(correlation_results[correlation_results['Spearman_sig'] != 'ns']),
        'pearson_mean': correlation_results['Pearson_r'].mean(),
        'pearson_std': correlation_results['Pearson_r'].std(),
        'pearson_min': correlation_results['Pearson_r'].min(),
        'pearson_max': correlation_results['Pearson_r'].max(),
        'positive_correlation': len(correlation_results[correlation_results['Pearson_r'] > 0]),
        'negative_correlation': len(correlation_results[correlation_results['Pearson_r'] < 0]),
    }

    # Calculate percentages
    summary['pearson_significant_pct'] = (summary['pearson_significant'] / summary['total_states'] * 100)
    summary['spearman_significant_pct'] = (summary['spearman_significant'] / summary['total_states'] * 100)
    summary['positive_correlation_pct'] = (summary['positive_correlation'] / summary['total_states'] * 100)
    summary['negative_correlation_pct'] = (summary['negative_correlation'] / summary['total_states'] * 100)

    return summary