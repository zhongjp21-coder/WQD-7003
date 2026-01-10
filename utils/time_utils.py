"""
Time series utilities for precipitation data
"""

import numpy as np
import pandas as pd
import xarray as xr
import warnings
from typing import Tuple


def process_time_series(ds: xr.Dataset) -> pd.DatetimeIndex:
    """
    Extract and standardize time series

    Why needed:
    1. Unify time formats (different sources may have different formats)
    2. Handle timezone issues
    3. Validate time continuity
    4. Provide standard index for monthly aggregation
    """
    dates = pd.to_datetime(ds.time.values)

    # Check time continuity
    time_diff = np.diff(dates)
    expected_diff = pd.Timedelta(days=1)

    gaps = np.where(time_diff != expected_diff)[0]
    if len(gaps) > 0:
        warnings.warn(f"Found {len(gaps)} time interval anomalies")
        print(f"\nInterval anomaly details:")
        for i, gap_idx in enumerate(gaps):
            date_before = dates[gap_idx]
            date_after = dates[gap_idx + 1]
            actual_diff = time_diff[gap_idx]

            print(f"\n  Anomaly {i + 1}:")
            print(f"    Before: {date_before}")
            print(f"    After: {date_after}")
            print(f"    Actual interval: {actual_diff}")
            print(f"    Missing days: {actual_diff.days - 1}")

            # List missing dates
            if actual_diff.days <= 10:  # Only show <=10 days missing
                missing_dates = pd.date_range(
                    date_before + pd.Timedelta(days=1),
                    date_after - pd.Timedelta(days=1),
                    freq='D'
                )
                print(f"    Missing dates: {list(missing_dates)}")
    else:
        print("  ✓ Time series is completely continuous")

    print(f"✓ Time series processing completed")
    print(f"  Start date: {dates[0]}")
    print(f"  End date: {dates[-1]}")
    print(f"  Total days: {len(dates)}")

    return dates


def calculate_monthly_aggregation(df_daily, precip_col='daily_precip',
                                  lat_col='lat', lon_col='lon',
                                  completeness_threshold=0.8):
    """
    Calculate monthly precipitation aggregation

    Parameters:
    -----------
    df_daily : DataFrame
        Daily precipitation data
    precip_col : str
        Precipitation column name
    completeness_threshold : float
        Minimum data completeness for monthly data

    Returns:
    --------
    DataFrame with monthly precipitation
    """
    monthly_list = []

    # Group by grid point
    for grid_id, group in df_daily.groupby('grid_id'):
        precip_series = group[precip_col]

        # Monthly aggregation (sum and count)
        monthly_agg = precip_series.resample('MS').agg(['sum', 'count'])

        # Calculate expected days per month
        expected_days = monthly_agg.index.to_series().dt.days_in_month.values

        # Filter incomplete months
        completeness = monthly_agg['count'].values / expected_days
        complete_mask = completeness > completeness_threshold

        # Keep only complete months
        monthly_precip = monthly_agg.loc[complete_mask, 'sum']

        if len(monthly_precip) > 0:
            grid_df = pd.DataFrame({
                'time': monthly_precip.index,
                'monthly_precip': monthly_precip.values,
                'grid_id': grid_id,
                lat_col: group[lat_col].iloc[0],
                lon_col: group[lon_col].iloc[0]
            })
            monthly_list.append(grid_df)

    if monthly_list:
        return pd.concat(monthly_list, ignore_index=True)
    else:
        return pd.DataFrame()