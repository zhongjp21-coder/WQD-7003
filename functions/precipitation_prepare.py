"""
Quality control functions for precipitation data
"""

import xarray as xr
from typing import Tuple, Dict
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import gamma, pearson3
from utils.spatial_utils import create_grid_polygons


def robust_missing_value_handling(
        ds: xr.Dataset,
        missing_val: float,
        valid_range: Tuple[float, float],
        max_missing_ratio: float = 0.3,
        handle_negatives: str = 'zero',
        handle_high_values: str = 'nan',
        apply_interpolation: bool = True
) -> Tuple[xr.Dataset, Dict]:
    """
    Robust missing value handling strategy

    Processing steps:
    1. Mark official missing values
    2. Handle negative values
    3. Handle extreme high values
    4. Grid-level quality control
    5. Time series interpolation (optional)
    6. Climatology filling (optional)
    """
    precip_var = 'precip' if 'precip' in ds else 'precipitation'
    data = ds[precip_var].copy()

    # Get dimension names
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'

    # Load data to memory if Dask array
    if hasattr(data, 'compute'):
        print("Loading data to memory...")
        data = data.compute()

    print("=" * 60)
    print("Starting missing value processing")
    print("=" * 60)

    # Step 1: Identify invalid values
    total_points = data.size

    # Official missing values
    official_mask = (data == missing_val)
    n_official_missing = int(official_mask.sum())

    # Negative values
    negative_mask = (data < 0) & (data != missing_val)
    n_negative = int(negative_mask.sum())

    # Extreme high values
    high_mask = (data > valid_range[1])
    n_high = int(high_mask.sum())

    # Existing NaN values
    existing_nan_mask = np.isnan(data)
    n_existing_nan = int(existing_nan_mask.sum())

    print(f"\n1. Official missing values ({missing_val}):")
    print(f"   Count: {n_official_missing:,}")
    print(f"   Percentage: {n_official_missing / total_points * 100:.2f}%")

    print(f"\n2. Negative values:")
    print(f"   Count: {n_negative:,}")
    print(f"   Percentage: {n_negative / total_points * 100:.2f}%")

    print(f"\n3. Extreme high values (> {valid_range[1]}):")
    print(f"   Count: {n_high:,}")
    print(f"   Percentage: {n_high / total_points * 100:.2f}%")

    print(f"\n4. Existing NaN values:")
    print(f"   Count: {n_existing_nan:,}")
    print(f"   Percentage: {n_existing_nan / total_points * 100:.2f}%")

    # Step 2: Apply processing strategies
    print(f"\n{'=' * 60}")
    print("Applying data cleaning strategies")
    print("=" * 60)

    # Handle official missing values
    data = data.where(~official_mask, np.nan)
    print(f"\n✓ Official missing values → NaN")

    # Handle negative values
    if handle_negatives == 'zero':
        data = data.where(~negative_mask, 0.0)
        print(f"✓ Negative values → 0 (assuming minor measurement error)")
    elif handle_negatives == 'nan':
        data = data.where(~negative_mask, np.nan)
        print(f"✓ Negative values → NaN (considered invalid)")

    # Handle extreme high values
    if handle_high_values == 'nan':
        data = data.where(~high_mask, np.nan)
        print(f"✓ Extreme high values → NaN")
    elif handle_high_values == 'cap':
        data = data.where(~high_mask, valid_range[1])
        print(f"✓ Extreme high values → capped at {valid_range[1]}")

    # Step 3: Grid-level quality control
    print(f"\n{'=' * 60}")
    print("Grid-level quality control")
    print("=" * 60)

    missing_ratio = data.isnull().sum(dim='time') / len(ds.time)
    valid_grids = missing_ratio < max_missing_ratio

    n_total_grids = valid_grids.size
    n_valid_grids = int(valid_grids.sum())
    n_removed_grids = n_total_grids - n_valid_grids

    print(f"\nMissing ratio threshold: {max_missing_ratio * 100:.0f}%")
    print(f"Retained grids: {n_valid_grids} / {n_total_grids}")
    print(f"Removed grids: {n_removed_grids} ({n_removed_grids / n_total_grids * 100:.1f}%)")

    # Apply grid mask
    data = data.where(valid_grids)

    # Step 4: Time series interpolation (optional)
    if apply_interpolation:
        print(f"\n{'=' * 60}")
        print("Time series interpolation")
        print("=" * 60)

        n_missing_before = int(data.isnull().sum())

        # Linear interpolation
        print("\nApplying linear interpolation (max gap=7 days)...")
        data_interpolated = data.interpolate_na(
            dim='time',
            method='linear',
            max_gap=np.timedelta64(7, 'D'),
            use_coordinate=True
        )

        # Climatology filling
        print("\nApplying climatology filling...")
        month_climatology = data_interpolated.groupby('time.month').mean(dim='time')

        data_filled = data_interpolated.copy()
        for month in range(1, 13):
            month_mask = data_filled.time.dt.month == month
            month_missing = data_filled.sel(time=month_mask).isnull()

            if month_missing.any():
                clim_value = month_climatology.sel(month=month)
                data_filled = data_filled.where(
                    ~(month_mask & data_filled.isnull()),
                    clim_value
                )

        n_missing_after = int(data_filled.isnull().sum())
        total_filled = n_missing_before - n_missing_after

        print(f"\nTotal filled: {total_filled:,} values")
        print(f"Remaining missing: {n_missing_after:,} values")

        data = data_filled

    # Step 5: Final statistics
    print(f"\n{'=' * 60}")
    print("Final data quality report")
    print("=" * 60)

    data_values = data.values if hasattr(data, 'values') else data
    total_points_final = data_values.size
    nan_points = np.sum(np.isnan(data_values))
    zero_points = np.sum(data_values == 0)
    valid_points = np.sum((data_values > 0) & (data_values <= valid_range[1]) & ~np.isnan(data_values))

    print(f"\nData point statistics:")
    print(f"  Total points:      {total_points_final:,}")
    print(f"  NaN:              {nan_points:,} ({nan_points / total_points_final * 100:.2f}%)")
    print(f"  Zero values:      {zero_points:,} ({zero_points / total_points_final * 100:.2f}%)")
    print(f"  Valid values(>0): {valid_points:,} ({valid_points / total_points_final * 100:.2f}%)")

    # Generate report
    report = {
        'original_issues': {
            'total_points': total_points,
            'official_missing': n_official_missing,
            'negative_values': n_negative,
            'high_values': n_high,
            'existing_nan': n_existing_nan,
        },
        'grid_quality_control': {
            'total_grids': n_total_grids,
            'valid_grids': n_valid_grids,
            'removed_grids': n_removed_grids,
        },
        'final_quality': {
            'total_points': total_points_final,
            'nan_count': nan_points,
            'zero_count': zero_points,
            'valid_count': valid_points,
        },
        'parameters': {
            'missing_val': missing_val,
            'valid_range': valid_range,
            'handle_negatives': handle_negatives,
            'handle_high_values': handle_high_values,
            'apply_interpolation': apply_interpolation,
            'max_missing_ratio': max_missing_ratio
        }
    }

    # Update dataset
    ds_clean = ds.copy()
    ds_clean[precip_var] = data

    ds_clean.attrs['quality_control'] = 'Robust missing value handling applied'
    ds_clean.attrs['qc_timestamp'] = pd.Timestamp.now().isoformat()

    print(f"\n{'=' * 60}")
    print("✓ Missing value processing completed")
    print("=" * 60)

    return ds_clean, report


def calculate_monthly_precipitation(ds: xr.Dataset,
                                    dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate monthly precipitation for SPI analysis

    Parameters:
    -----------
    ds : xr.Dataset
        Cleaned dataset
    dates : pd.DatetimeIndex
        Standardized time series

    Returns:
    --------
    df : DataFrame
        Monthly precipitation data
    grid_metadata : Dict
        Grid metadata
    """
    precip_var = 'precip' if 'precip' in ds else 'precipitation'
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'

    print("=" * 60)
    print("Calculating monthly precipitation")
    print("=" * 60)

    # Ensure time coordinates are correct
    ds = ds.assign_coords(time=dates)

    # Convert to DataFrame
    print("\nStep 1: Converting to DataFrame...")
    df_daily = ds[precip_var].to_dataframe(name='daily_precip').reset_index()

    print(f"  Daily data: {len(df_daily):,} rows")
    print(f"  Time range: {df_daily['time'].min()} to {df_daily['time'].max()}")

    # Add grid ID
    df_daily['grid_id'] = (
            df_daily[lat_name].round(4).astype(str) + '_' +
            df_daily[lon_name].round(4).astype(str)
    )

    n_grids = df_daily['grid_id'].nunique()
    print(f"  Grid points: {n_grids}")

    # Monthly aggregation using pandas
    print("\nStep 2: Monthly aggregation...")
    df_daily = df_daily.set_index('time')

    from utils.time_utils import calculate_monthly_aggregation
    df_monthly = calculate_monthly_aggregation(
        df_daily,
        precip_col='daily_precip',
        lat_col=lat_name,
        lon_col=lon_name
    )

    # Save metadata
    grid_metadata = {
        'lat_coords': ds[lat_name].values,
        'lon_coords': ds[lon_name].values,
        'resolution': float(np.abs(ds[lon_name].diff(lon_name).median())),
        'time_range': (dates[0], dates[-1])
    }

    print(f"\n{'=' * 60}")
    print("✓ Monthly precipitation calculation completed")
    print("=" * 60)
    print(f"  Grid points:  {df_monthly['grid_id'].nunique()}")
    print(f"  Months: {df_monthly['time'].nunique()}")
    print(f"  Data shape: {df_monthly.shape}")

    return df_monthly, grid_metadata


def clean_data(df: pd.DataFrame, min_valid_months: int = 90) -> pd.DataFrame:
    """
    Remove grid points with insufficient data quality

    Parameters:
    -----------
    df : DataFrame
        Monthly precipitation data
    min_valid_months : int
        Minimum number of valid months required

    Returns:
    --------
    df_clean : DataFrame
        Cleaned data
    """
    # Count valid months per grid point
    valid_counts = df.groupby('grid_id')['monthly_precip'].count()
    valid_grids = valid_counts[valid_counts >= min_valid_months].index

    df_clean = df[df['grid_id'].isin(valid_grids)].copy()

    removed = len(df['grid_id'].unique()) - len(valid_grids)
    print(f"✓ Data cleaning completed")
    print(f"  Removed grids: {removed}")
    print(f"  Retained grids: {len(valid_grids)}")

    return df_clean


def robust_missing_value_handling(
        ds: xr.Dataset,
        missing_val: float,
        valid_range: Tuple[float, float],
        max_missing_ratio: float = 0.3,
        handle_negatives: str = 'zero',
        handle_high_values: str = 'nan',
        apply_interpolation: bool = True
) -> Tuple[xr.Dataset, Dict]:
    """
    Robust missing value handling strategy

    Processing steps:
    1. Mark official missing values
    2. Handle negative values
    3. Handle extreme high values
    4. Grid-level quality control
    5. Time series interpolation (optional)
    6. Climatology filling (optional)
    """
    precip_var = 'precip' if 'precip' in ds else 'precipitation'
    data = ds[precip_var].copy()

    # Get dimension names
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'

    # Load data to memory if Dask array
    if hasattr(data, 'compute'):
        print("Loading data to memory...")
        data = data.compute()

    print("=" * 60)
    print("Starting missing value processing")
    print("=" * 60)

    # Step 1: Identify invalid values
    total_points = data.size

    # Official missing values
    official_mask = (data == missing_val)
    n_official_missing = int(official_mask.sum())

    # Negative values
    negative_mask = (data < 0) & (data != missing_val)
    n_negative = int(negative_mask.sum())

    # Extreme high values
    high_mask = (data > valid_range[1])
    n_high = int(high_mask.sum())

    # Existing NaN values
    existing_nan_mask = np.isnan(data)
    n_existing_nan = int(existing_nan_mask.sum())

    print(f"\n1. Official missing values ({missing_val}):")
    print(f"   Count: {n_official_missing:,}")
    print(f"   Percentage: {n_official_missing / total_points * 100:.2f}%")

    print(f"\n2. Negative values:")
    print(f"   Count: {n_negative:,}")
    print(f"   Percentage: {n_negative / total_points * 100:.2f}%")

    print(f"\n3. Extreme high values (> {valid_range[1]}):")
    print(f"   Count: {n_high:,}")
    print(f"   Percentage: {n_high / total_points * 100:.2f}%")

    print(f"\n4. Existing NaN values:")
    print(f"   Count: {n_existing_nan:,}")
    print(f"   Percentage: {n_existing_nan / total_points * 100:.2f}%")

    # Step 2: Apply processing strategies
    print(f"\n{'=' * 60}")
    print("Applying data cleaning strategies")
    print("=" * 60)

    # Handle official missing values
    data = data.where(~official_mask, np.nan)
    print(f"\n✓ Official missing values → NaN")

    # Handle negative values
    if handle_negatives == 'zero':
        data = data.where(~negative_mask, 0.0)
        print(f"✓ Negative values → 0 (assuming minor measurement error)")
    elif handle_negatives == 'nan':
        data = data.where(~negative_mask, np.nan)
        print(f"✓ Negative values → NaN (considered invalid)")

    # Handle extreme high values
    if handle_high_values == 'nan':
        data = data.where(~high_mask, np.nan)
        print(f"✓ Extreme high values → NaN")
    elif handle_high_values == 'cap':
        data = data.where(~high_mask, valid_range[1])
        print(f"✓ Extreme high values → capped at {valid_range[1]}")

    # Step 3: Grid-level quality control
    print(f"\n{'=' * 60}")
    print("Grid-level quality control")
    print("=" * 60)

    missing_ratio = data.isnull().sum(dim='time') / len(ds.time)
    valid_grids = missing_ratio < max_missing_ratio

    n_total_grids = valid_grids.size
    n_valid_grids = int(valid_grids.sum())
    n_removed_grids = n_total_grids - n_valid_grids

    print(f"\nMissing ratio threshold: {max_missing_ratio * 100:.0f}%")
    print(f"Retained grids: {n_valid_grids} / {n_total_grids}")
    print(f"Removed grids: {n_removed_grids} ({n_removed_grids / n_total_grids * 100:.1f}%)")

    # Apply grid mask
    data = data.where(valid_grids)

    # Step 4: Time series interpolation (optional)
    if apply_interpolation:
        print(f"\n{'=' * 60}")
        print("Time series interpolation")
        print("=" * 60)

        n_missing_before = int(data.isnull().sum())

        # Linear interpolation
        print("\nApplying linear interpolation (max gap=7 days)...")
        data_interpolated = data.interpolate_na(
            dim='time',
            method='linear',
            max_gap=np.timedelta64(7, 'D'),
            use_coordinate=True
        )

        # Climatology filling
        print("\nApplying climatology filling...")
        month_climatology = data_interpolated.groupby('time.month').mean(dim='time')

        data_filled = data_interpolated.copy()
        for month in range(1, 13):
            month_mask = data_filled.time.dt.month == month
            month_missing = data_filled.sel(time=month_mask).isnull()

            if month_missing.any():
                clim_value = month_climatology.sel(month=month)
                data_filled = data_filled.where(
                    ~(month_mask & data_filled.isnull()),
                    clim_value
                )

        n_missing_after = int(data_filled.isnull().sum())
        total_filled = n_missing_before - n_missing_after

        print(f"\nTotal filled: {total_filled:,} values")
        print(f"Remaining missing: {n_missing_after:,} values")

        data = data_filled

    # Step 5: Final statistics
    print(f"\n{'=' * 60}")
    print("Final data quality report")
    print("=" * 60)

    data_values = data.values if hasattr(data, 'values') else data
    total_points_final = data_values.size
    nan_points = np.sum(np.isnan(data_values))
    zero_points = np.sum(data_values == 0)
    valid_points = np.sum((data_values > 0) & (data_values <= valid_range[1]) & ~np.isnan(data_values))

    print(f"\nData point statistics:")
    print(f"  Total points:      {total_points_final:,}")
    print(f"  NaN:              {nan_points:,} ({nan_points / total_points_final * 100:.2f}%)")
    print(f"  Zero values:      {zero_points:,} ({zero_points / total_points_final * 100:.2f}%)")
    print(f"  Valid values(>0): {valid_points:,} ({valid_points / total_points_final * 100:.2f}%)")

    # Generate report
    report = {
        'original_issues': {
            'total_points': total_points,
            'official_missing': n_official_missing,
            'negative_values': n_negative,
            'high_values': n_high,
            'existing_nan': n_existing_nan,
        },
        'grid_quality_control': {
            'total_grids': n_total_grids,
            'valid_grids': n_valid_grids,
            'removed_grids': n_removed_grids,
        },
        'final_quality': {
            'total_points': total_points_final,
            'nan_count': nan_points,
            'zero_count': zero_points,
            'valid_count': valid_points,
        },
        'parameters': {
            'missing_val': missing_val,
            'valid_range': valid_range,
            'handle_negatives': handle_negatives,
            'handle_high_values': handle_high_values,
            'apply_interpolation': apply_interpolation,
            'max_missing_ratio': max_missing_ratio
        }
    }

    # Update dataset
    ds_clean = ds.copy()
    ds_clean[precip_var] = data

    ds_clean.attrs['quality_control'] = 'Robust missing value handling applied'
    ds_clean.attrs['qc_timestamp'] = pd.Timestamp.now().isoformat()

    print(f"\n{'=' * 60}")
    print("✓ Missing value processing completed")
    print("=" * 60)

    return ds_clean, report


def add_administrative_regions(df: pd.DataFrame,
                               states_geojson_path: str,
                               grid_size: float = 0.5) -> pd.DataFrame:
    """
    Add administrative region labels to DataFrame with lat/lon columns

    Parameters:
    -----------
    df : DataFrame
        Input data with 'lat' and 'lon' columns
    states_geojson_path : str
        Path to administrative boundaries GeoJSON file
    grid_size : float
        Grid cell size in degrees

    Returns:
    --------
    DataFrame with added 'region' column
    """
    # Create grid polygons
    grids_gdf = create_grid_polygons(df, grid_size=grid_size)

    # Read administrative boundaries
    states = gpd.read_file(states_geojson_path)

    # Spatial join
    result = gpd.sjoin(
        grids_gdf,
        states[['NAME_1', 'geometry']],
        how='left',
        predicate='intersects'
    )

    # Rename and clean columns
    result = result.rename(columns={'NAME_1': 'region'})

    # Handle unmatched grids
    unmatched = result['region'].isna()
    if unmatched.any():
        result.loc[unmatched, 'region'] = 'Sea'  # Mark as sea/ocean

    # Clean up
    result = result.drop(columns=['geometry', 'index_right'], errors='ignore')

    # Statistics
    region_counts = result.groupby('region')['grid_id'].nunique()
    print(f"✓ Region labeling completed")
    for region, count in region_counts.items():
        print(f"  {region}: {count} grid points")

    return result


def add_geographic_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add East/West Malaysia classification

    Classification criteria:
    - West Malaysia (Peninsular): 99°E - 104°E
    - East Malaysia: 109°E - 119°E
    """
    lon_name = 'lon' if 'lon' in df.columns else 'longitude'

    def classify_region(lon):
        if 99 <= lon <= 104:
            return 'Peninsular'
        elif 109 <= lon <= 119:
            return 'East'
        else:
            return 'Other'

    df['geographic_region'] = df[lon_name].apply(classify_region)

    # Statistics
    region_counts = df.groupby('geographic_region')['grid_id'].nunique()
    print(f"✓ Geographic region labeling completed")
    for region, count in region_counts.items():
        print(f"  {region}: {count} grid points")

    return df


def calculate_spi(df: pd.DataFrame,
                  timescales: list = [1, 3, 6, 12],
                  distribution: str = 'gamma') -> pd.DataFrame:
    """
    Calculate Standardized Precipitation Index (SPI)

    Parameters:
    -----------
    df : DataFrame
        Monthly precipitation data
    timescales : list
        Time scales in months
    distribution : str
        Distribution type ('gamma' or 'pearson3')

    Features:
    1. Support multiple time scales
    2. Automatic distribution selection
    3. Handle zero precipitation
    4. Provide fitting diagnostics
    """
    results = []

    for grid_id in df['grid_id'].unique():
        grid_data = df[df['grid_id'] == grid_id].sort_values('time')

        for scale in timescales:
            # Calculate rolling accumulated precipitation
            rolling_precip = grid_data['monthly_precip'].rolling(
                window=scale,
                min_periods=scale
            ).sum()

            # Remove NaN
            valid_data = rolling_precip.dropna()

            if len(valid_data) < 30:  # Minimum 30 samples
                continue

            # Handle zero values (add small perturbation)
            valid_data_adjusted = valid_data.copy()
            zero_mask = valid_data == 0
            if zero_mask.any():
                valid_data_adjusted[zero_mask] = np.random.uniform(
                    0.001, 0.01, zero_mask.sum()
                )

            # Fit distribution
            try:
                if distribution == 'gamma':
                    params = gamma.fit(valid_data_adjusted, floc=0)
                    cdf_values = gamma.cdf(valid_data_adjusted, *params)
                else:
                    params = pearson3.fit(valid_data_adjusted)
                    cdf_values = pearson3.cdf(valid_data_adjusted, *params)

                # Convert to SPI
                spi_values = stats.norm.ppf(cdf_values)

                # Handle extreme values
                spi_values = np.clip(spi_values, -3.5, 3.5)

                # Save results
                temp_df = grid_data.iloc[scale - 1:].copy()
                temp_df[f'SPI_{scale}'] = spi_values
                temp_df['timescale'] = scale
                results.append(temp_df)

            except Exception as e:
                warnings.warn(f"Grid {grid_id}, scale {scale} fitting failed: {e}")
                continue

    if results:
        df_spi = pd.concat(results, ignore_index=True)
    else:
        df_spi = pd.DataFrame()

    print(f"✓ SPI calculation completed")
    print(f"  Time scales: {timescales}")
    print(f"  Distribution: {distribution}")

    if not df_spi.empty:
        spi_cols = [c for c in df_spi.columns if 'SPI' in c]
        print(f"  Output columns: {spi_cols}")

    return df_spi


def calculate_spi_batch(df: pd.DataFrame,
                        grid_ids: list = None,
                        **kwargs) -> pd.DataFrame:
    """
    Calculate SPI for multiple grid points in batch

    Parameters:
    -----------
    df : DataFrame
        Monthly precipitation data
    grid_ids : list
        Specific grid IDs to process (None for all)
    **kwargs : dict
        Additional parameters for calculate_spi

    Returns:
    --------
    DataFrame with SPI values
    """
    if grid_ids is None:
        grid_ids = df['grid_id'].unique()

    all_results = []

    for i, grid_id in enumerate(grid_ids):
        if (i + 1) % 50 == 0:
            print(f"Processing grid {i + 1}/{len(grid_ids)}")

        grid_data = df[df['grid_id'] == grid_id]
        grid_spi = calculate_spi(grid_data, **kwargs)

        if not grid_spi.empty:
            all_results.append(grid_spi)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

