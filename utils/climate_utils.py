"""
Climate Analysis Utilities
Basic tools for CMIP6 climate model data processing and analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk
import warnings

warnings.filterwarnings('ignore')


def read_cmip_excel_sheets(excel_path):
    """Read all sheets from CMIP6 Excel file"""
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")

    all_data = {}
    for sheet in sheet_names:
        df = xl.parse(sheet)
        print(f"  Sheet '{sheet}' shape: {df.shape}, columns: {df.columns.tolist()[:3]}...")
        all_data[sheet] = df

    return all_data, sheet_names


def standardize_model_data(df, model_name):
    """Clean and standardize single model data format"""
    # Check data format type 1: code, name, date columns
    if 'code' in df.columns and 'name' in df.columns:
        df_melted = df.melt(
            id_vars=['code', 'name'],
            var_name='date',
            value_name='precip'
        )

    # Check data format type 2: date columns, state rows
    elif 'date' in df.columns or any(col.startswith('20') for col in df.columns):
        df_melted = df.melt(
            id_vars=df.columns[0],
            var_name='date',
            value_name='precip'
        )
        df_melted = df_melted.rename(columns={df.columns[0]: 'name'})
        df_melted['code'] = df_melted['name']

    else:
        raise ValueError(f"Cannot recognize data format for {model_name}")

    # Convert date format
    df_melted['date'] = pd.to_datetime(df_melted['date'], errors='coerce')
    df_melted = df_melted.dropna(subset=['date'])

    # Add model label
    df_melted['model'] = model_name

    # Sort by state and date
    df_melted = df_melted.sort_values(['name', 'date']).reset_index(drop=True)

    return df_melted


def calculate_spi_for_model(precip_series, scale=3, historical_years=(2015, 2024)):
    """Calculate SPI index for single model data"""
    if len(precip_series) == 0:
        return pd.Series(np.nan, index=precip_series.index)

    # Ensure time series index
    if not isinstance(precip_series.index, pd.DatetimeIndex):
        try:
            precip_series.index = pd.to_datetime(precip_series.index)
        except:
            return pd.Series(np.nan, index=precip_series.index)

    # Calculate 3-month cumulative precipitation
    precip_accum = precip_series.rolling(window=scale, min_periods=1).sum()

    # Separate historical period
    hist_mask = (precip_accum.index.year >= historical_years[0]) & (precip_accum.index.year <= historical_years[1])
    historical = precip_accum[hist_mask].dropna()

    # Check historical data
    if len(historical) < 12:
        # Use simple Z-score method
        spi_simple = (precip_accum - precip_accum.mean()) / precip_accum.std()
        spi_simple = spi_simple.clip(-3, 3)
        return spi_simple

    # Use empirical distribution method
    sorted_hist = np.sort(historical.values)
    spi_values = []

    for value in precip_accum.values:
        if np.isnan(value):
            spi_values.append(np.nan)
        else:
            # Calculate empirical CDF
            rank = np.searchsorted(sorted_hist, value, side='right')
            cdf = rank / len(sorted_hist)

            # Avoid 0 or 1
            cdf = np.clip(cdf, 1 / (2 * len(sorted_hist)), 1 - 1 / (2 * len(sorted_hist)))

            # Convert to SPI
            spi = stats.norm.ppf(cdf)
            spi_values.append(spi)

    return pd.Series(spi_values, index=precip_accum.index)


def extract_growing_season(data_series, season_type='main'):
    """Extract growing season data"""
    if data_series.empty:
        return pd.Series([], dtype=float)

    months = data_series.index.month

    if season_type == 'main':
        mask = months.isin([9, 10, 11, 12, 1, 2])
    else:
        mask = months.isin([3, 4, 5, 6, 7, 8])

    growing_data = data_series[mask].copy()

    if growing_data.empty:
        return pd.Series([], dtype=float)

    # Calculate annual averages
    annual_means = []
    annual_years = []

    min_year = data_series.index.year.min()
    max_year = data_series.index.year.max()

    for year in range(min_year, max_year + 1):
        if season_type == 'main':
            year_mask = ((data_series.index.year == year - 1) & data_series.index.month.isin([9, 10, 11, 12])) | \
                        ((data_series.index.year == year) & data_series.index.month.isin([1, 2]))
        else:
            year_mask = (data_series.index.year == year) & data_series.index.month.isin([3, 4, 5, 6, 7, 8])

        year_data = data_series[year_mask]
        if len(year_data) >= 3:
            annual_means.append(year_data.mean())
            annual_years.append(year)

    return pd.Series(annual_means, index=annual_years, name='annual_spi')


def perform_mann_kendall_test(annual_series, alpha=0.05):
    """Perform Mann-Kendall trend test"""
    if isinstance(annual_series, pd.Series):
        values = annual_series.values
    elif isinstance(annual_series, np.ndarray):
        values = annual_series
    elif isinstance(annual_series, list):
        values = np.array(annual_series)
    else:
        return {'trend': 'invalid_input', 'p': np.nan, 'slope': np.nan}

    # Remove NaN values
    values = values[~np.isnan(values)]

    if len(values) < 8:
        return {'trend': 'insufficient_data', 'p': np.nan, 'slope': np.nan}

    # Check data variation
    if np.std(values) < 1e-10:
        return {'trend': 'no_variation', 'p': 1.0, 'slope': 0.0}

    try:
        result = mk.original_test(values, alpha=alpha)
        return {'trend': result.trend, 'p': result.p, 'slope': result.slope}
    except Exception as e:
        # Fallback to linear regression
        try:
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            trend = 'increasing' if slope > 0 else ('decreasing' if slope < 0 else 'no_trend')
            return {'trend': trend, 'p': p_value, 'slope': slope}
        except:
            return {'trend': 'test_failed', 'p': np.nan, 'slope': np.nan}


def calculate_ensemble_stats(spi_data, group_cols=['state', 'date']):
    """Calculate ensemble statistics for multiple models"""
    ensemble_stats = []

    for group_key, group in spi_data.groupby(group_cols):
        if len(group) >= 2:
            spi_values = group['spi_3month'].dropna()

            if len(spi_values) > 0:
                if len(group_cols) == 2:
                    state, date = group_key
                    ensemble_stats.append({
                        'state': state,
                        'date': date,
                        'spi_median': np.median(spi_values),
                        'spi_p25': np.percentile(spi_values, 25),
                        'spi_p75': np.percentile(spi_values, 75),
                        'model_count': len(spi_values)
                    })
                else:
                    # Handle different grouping if needed
                    pass

    if ensemble_stats:
        return pd.DataFrame(ensemble_stats)
    else:
        return pd.DataFrame()