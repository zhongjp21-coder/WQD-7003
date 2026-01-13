"""
Complete pipeline for monthly precipitation calculation with administrative boundaries and SPI calculation
"""

import json
import pandas as pd

from datetime import datetime
from pathlib import Path

from config import get_country_paths
from utils.data_io import read_precipitation_data, save_processed_data
from utils.spatial_utils import filter_geographic_region
from utils.time_utils import process_time_series
from functions.precipitation_prepare import robust_missing_value_handling
from functions.precipitation_prepare import calculate_monthly_precipitation, clean_data
from functions.precipitation_prepare import add_administrative_regions, add_geographic_regions
from functions.precipitation_prepare import calculate_spi
import os
import requests
from tqdm import tqdm

# =================config=================
BASE_URL = "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/"
DATA_RANGE = [2017, 2022]
SAVE_DIR = "./datasets/Malaysia/rawdata/history_precipitation"


# =========================================

def download_cpc_fixed_range():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"directory has been created: {SAVE_DIR}")

    start_year, end_year = DATA_RANGE
    print(f"In Progressing | Data source: {BASE_URL}")
    print(f"Time range: {start_year} to {end_year}")
    print(f"Saving location: {SAVE_DIR}")
    print("-" * 60)

    # 2. download data for each year
    for year in range(start_year, end_year + 1):
        file_name = f"precip.{year}.nc"
        file_url = f"{BASE_URL}{file_name}"
        save_path = os.path.join(SAVE_DIR, file_name)

        # check the file whether is existed
        if os.path.exists(save_path):
            print(f"[skip] file has existed: {file_name}")
            continue

        print(f"downloading: {file_name} ...")

        response = requests.get(file_url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        # Write file and display progress bar
        with open(save_path, 'wb') as file, tqdm(
                desc=file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    print("-" * 60)
    print("downloading completed.")


def run_monthly_boundary_spi(config: dict):
    """
    Run complete pipeline for monthly precipitation with administrative boundaries and SPI calculation

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    dict: Results containing monthly data, SPI data and paths
    """
    # Get country
    country = config.get('country', 'malaysia')

    # Generate run tag if not provided
    run_tag = config.get('run_tag')
    if run_tag is None:
        run_tag = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get file paths
    paths = get_country_paths(country, run_tag)

    print("=" * 60)
    print(f"MONTHLY BOUNDARY SPI PIPELINE")
    print("=" * 60)
    print(f"Country: {country.upper()}")
    print(f"Run tag: {run_tag}")
    print(f"Output directory: {paths['processed']}")
    print("=" * 60)

    # Get all parameters
    params = config.get('parameters', {})
    processing_options = config.get('processing_options', {})
    io_config = config.get('io_config', {})
    print(f"\n0. DOWNLOAD PRECIPITATION DATA FROM NOAA")
    download_cpc_fixed_range()

    print(f"\n1. READING DATA")
    print("-" * 40)

    # Determine data path
    data_path = io_config.get('data_path')
    if data_path is None:
        data_path = str(paths['rawdata'] / "history_precipitation")

    print(f"Data path: {data_path}")
    ds, missing_val, valid_range = read_precipitation_data(data_path)

    print(f"\n2. REGIONAL FILTERING")
    print("-" * 40)
    lat_range = tuple(params.get('lat_range', [0.5, 7.5]))
    lon_range = tuple(params.get('lon_range', [99.5, 120.5]))
    ds_sub = filter_geographic_region(ds, lat_range, lon_range)

    print(f"\n3. QUALITY CONTROL")
    print("-" * 40)
    qc_options = {
        'missing_val': missing_val,
        'valid_range': valid_range,
        'max_missing_ratio': params.get('max_missing_ratio', 0.3),
        'handle_negatives': processing_options.get('handle_negatives', 'zero'),
        'handle_high_values': processing_options.get('handle_high_values', 'nan'),
        'apply_interpolation': processing_options.get('apply_interpolation', True)
    }

    ds_clean, qc_report = robust_missing_value_handling(ds_sub, **qc_options)

    # Save quality control report
    report_path = paths['processed'] / "quality_control_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(qc_report, f, indent=2, default=str)
    print(f"✓ Quality control report saved to: {report_path}")

    print(f"\n4. TIME SERIES PROCESSING")
    print("-" * 40)
    dates = process_time_series(ds_clean)

    print(f"\n5. MONTHLY PRECIPITATION CALCULATION")
    print("-" * 40)
    df_monthly, grid_meta = calculate_monthly_precipitation(ds_clean, dates)

    print(f"\n6. REGION LABELING")
    print("-" * 40)
    print(paths['states_geojson'])
    if Path(paths['states_geojson']).exists():
        df_monthly = add_administrative_regions(
            df_monthly,
            str(paths['states_geojson']),
            grid_size=params.get('grid_size', 0.5)
        )
    else:
        print(f"Warning: Administrative boundaries file not found")
        df_monthly = add_geographic_regions(df_monthly)

    print(f"\n7. DATA CLEANING")
    print("-" * 40)
    df_clean = clean_data(df_monthly, min_valid_months=params.get('min_valid_months', 90))

    # Save cleaned monthly data
    output_format = io_config.get('output_format', 'csv')
    monthly_path = paths['processed'] / f"monthly_precipitation_cleaned.{output_format}"
    save_processed_data(df_clean, monthly_path, format=output_format)

    print(f"\n8. SPI CALCULATION")
    print("-" * 40)
    spi_params = {
        'timescales': params.get('timescales', [1, 3, 6, 12]),
        'distribution': params.get('distribution', 'gamma')
    }

    df_spi = calculate_spi(df_clean, **spi_params)

    # Save SPI results
    spi_path = paths['processed'] / f"spi_results.{output_format}"
    save_processed_data(df_spi, spi_path, format=output_format)

    # Save metadata
    metadata = {
        'country': country,
        'run_tag': run_tag,
        'pipeline': 'monthly_boundary_SPI',
        'parameters': params,
        'processing_options': processing_options,
        'grid_metadata': grid_meta,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = paths['processed'] / "pipeline_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("✓ MONTHLY BOUNDARY SPI PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Results saved to:")
    print(f"  Monthly data (cleaned): {monthly_path}")
    print(f"  SPI results: {spi_path}")
    print(f"  QC report: {report_path}")
    print(f"  Metadata: {metadata_path}")
    print("=" * 60)

    # Statistics for each state
    state_summary = df_spi.groupby('region').agg(
        valid_grid=('grid_id', 'nunique'),
        total_value=('monthly_precip', 'count')
    ).reset_index()

    # calc average precipitation
    avg_precip = df_spi.groupby('region')['monthly_precip'].mean().reset_index()
    state_summary = pd.merge(state_summary, avg_precip.rename(columns={'monthly_precip': 'average precipitation(mm)'}),
                             on='region')

    # calc SPI succeed percentage
    def spi_success_rate(group):
        spi_cols = ['SPI_1', 'SPI_3', 'SPI_6', 'SPI_12']
        valid_spi = group[spi_cols].notna().any(axis=1).sum()
        return valid_spi / len(group) * 100 if len(group) > 0 else 0

    spi_rates = df_spi.groupby('region').apply(spi_success_rate).reset_index(name='SPI caculate succeed %')
    state_summary = pd.merge(state_summary, spi_rates, on='region')

    # Format output
    state_summary = state_summary.round({
        'average precipitation(mm)': 2,
        'SPI caculate succeed %': 1
    })

    # ================= New: In-depth statistical analysis =================
    # 1. Calculate statistical metrics
    df_regional_stats = analyze_regional_drought_characteristics(df_spi)

    # 2. Save statistical results to CSV
    stats_path = paths['processed'] / "regional_drought_statistics.csv"
    df_regional_stats.to_csv(stats_path, index=False)
    print(f"✓ Regional drought statistics table saved: {stats_path}")

    img_save_dir = str(paths['processed'])  # Use your output directory
    # 1. Plot 'Extreme drought' frequency heatmap
    plot_drought_severity_heatmap(df_regional_stats, severity='Extreme', save_dir=img_save_dir)

    # 2. Plot 'Severe drought' frequency heatmap
    plot_drought_severity_heatmap(df_regional_stats, severity='Severe', save_dir=img_save_dir)

    # 4. Plot trend analysis heatmap (with asterisks)
    plot_trend_heatmap(df_regional_stats, save_dir=img_save_dir)

    # D. [Specified] Spatiotemporal evolution heatmap (SPI-3 only)
    plot_temporal_heatmap(df_spi, timescale="SPI_3", save_dir=img_save_dir)

    return {
        'monthly_data': df_clean,
        'spi_data': df_spi,
        'grid_metadata': grid_meta,
        'qc_report': qc_report,
        'metadata': metadata,
        'paths': paths
    }


import pandas as pd
import numpy as np
from scipy import stats


def analyze_regional_drought_characteristics(df_spi):
    """
    Perform in-depth statistical analysis of SPI data by state: drought frequency, intensity, and trend
    """
    print("=" * 60)
    print("Performing regional drought characteristics statistical analysis...")
    print("=" * 60)

    stats_list = []

    # Get all SPI columns (e.g., SPI_1, SPI_3...)
    spi_cols = [c for c in df_spi.columns if 'SPI_' in c]

    # Iterate by region (state) groups
    for region, group in df_spi.groupby('region'):
        region_stats = {'Region': region}
        print("analyzing...", group)
        for col in spi_cols:
            # Extract valid data
            valid_data = group[col].dropna()
            if len(valid_data) == 0:
                continue

            # --- 1. Drought frequency analysis (Frequency) - based on figure criteria ---
            total_months = len(valid_data)

            # 1. Mild drought: 0 to -0.99
            # Logic: less than or equal to 0 and greater than -1.0
            cnt_mild = ((valid_data <= 0) & (valid_data > -1.0)).sum()

            # 2. Moderate drought: -1.00 to -1.49
            # Logic: less than or equal to -1.0 and greater than -1.5
            cnt_mod = ((valid_data <= -1.0) & (valid_data > -1.5)).sum()

            # 3. Severe drought: -1.50 to -1.99 (corrected the figure's typo 1.50)
            # Logic: less than or equal to -1.5 and greater than -2.0
            cnt_sev = ((valid_data <= -1.5) & (valid_data > -2.0)).sum()

            # 4. Extreme drought: <= -2.00
            # Logic: less than or equal to -2.0
            cnt_ext = (valid_data <= -2.0).sum()

            # Calculate percentages and store them (keep 2 decimals)
            region_stats[f'{col}_Mild(%)'] = round((cnt_mild / total_months) * 100, 2)
            region_stats[f'{col}_Moderate(%)'] = round((cnt_mod / total_months) * 100, 2)
            region_stats[f'{col}_Severe(%)'] = round((cnt_sev / total_months) * 100, 2)
            region_stats[f'{col}_Extreme(%)'] = round((cnt_ext / total_months) * 100, 2)

            # --- 2. Extremes analysis ---
            # Historical minimum SPI value (driest)
            region_stats[f'{col}_Min'] = round(valid_data.min(), 2)

            # --- 3. Trend analysis (optimized for short-term 2017-2022 data) ---
            # With only 6 years of data, compute "SPI change per year"
            # Threshold > 24 months (at least ~2 years of valid data) to compute trend
            if len(valid_data) > 24:
                # 1. Extract dates
                dates = pd.to_datetime(group.loc[valid_data.index, 'time'])

                # 2. Convert dates to relative years
                # e.g., 2017-01-01 is 0.0 year, 2018-01-01 is 1.0 year
                start_date = dates.min()
                x_years = (dates - start_date).dt.days / 365.25

                # 3. Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_years, valid_data)

                # 4. Store results
                # Store slope directly, representing 'SPI value change per year'
                region_stats[f'{col}_Trend(per_year)'] = round(slope, 4)

                # Significance test (with small sample size, p-values may be large; reflects uncertainty)
                region_stats[f'{col}_Trend_Signif'] = p_value < 0.05

            else:
                region_stats[f'{col}_Trend(per_year)'] = np.nan
                region_stats[f'{col}_Trend_Signif'] = False

        stats_list.append(region_stats)

    # Convert to DataFrame
    df_stats = pd.DataFrame(stats_list)

    return df_stats


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_drought_severity_heatmap(df_stats, severity='Extreme', save_dir=None):
    """
    Plot a heatmap of the frequency of a specific drought severity level

    Parameters:
    -----------
    df_stats : pd.DataFrame
        Statistics generated by analyze_regional_drought_characteristics
    severity : str
        Drought level to visualize; options: 'Mild', 'Moderate', 'Severe', 'Extreme'
    """
    # Build column suffix, e.g., '_Extreme(%)'
    metric_suffix = f'_{severity}(%)'

    # Select relevant columns
    target_cols = [c for c in df_stats.columns if metric_suffix in c]

    if not target_cols:
        print(f"⚠️ No columns containing {metric_suffix} were found. Please check the data or the severity parameter.")
        print(f"Available column name samples: {df_stats.columns[:5]}")
        return

    # Prepare plotting data: set Region as index
    plot_data = df_stats.set_index('Region')[target_cols]

    # Simplify column names (remove suffix, keep SPI_1, SPI_3, etc.)
    plot_data.columns = [c.replace(metric_suffix, '') for c in plot_data.columns]

    # --- Sorting optimization ---
    # Sort by mean frequency across timescales so drier regions appear on top
    plot_data['mean'] = plot_data.mean(axis=1)
    plot_data = plot_data.sort_values('mean', ascending=False).drop(columns='mean')

    # --- Plotting ---
    # Choose colormap by severity: yellow-brown for milder, red for extreme
    if severity in ['Severe', 'Extreme']:
        cmap = 'Reds'
    else:
        cmap = 'YlOrBr'

    plt.figure(figsize=(10, len(plot_data) * 0.5 + 2))

    ax = sns.heatmap(plot_data,
                     annot=True,
                     cmap=cmap,
                     fmt='.1f',
                     linewidths=.5,
                     cbar_kws={'label': f'Frequency of {severity} Drought (%)'})

    plt.title(f'Regional Drought Analysis: {severity} Drought Frequency', fontsize=14)
    plt.ylabel('Region (State)', fontsize=12)
    plt.xlabel('SPI Timescale', fontsize=12)
    plt.tight_layout()

    if save_dir:
        import os
        filename = f"heatmap_{severity}_frequency.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"✓ Heatmap saved: {save_path}")


def plot_trend_heatmap(df_stats, save_dir=None):
    """
    Plot an SPI trend heatmap and mark statistical significance
    """
    # Select trend columns
    trend_cols = [c for c in df_stats.columns if 'Trend(per_year)' in c]

    if not trend_cols:
        print("⚠️ No trend data columns found")
        return

    # Prepare main data (slope)
    slope_data = df_stats.set_index('Region')[trend_cols]
    slope_data.columns = [c.replace('_Trend(per_year)', '') for c in slope_data.columns]  # Simplify column names

    # Prepare significance data (for asterisks)
    # Find corresponding Signif column
    annot_data = slope_data.copy()
    for col in slope_data.columns:  # col is "SPI_1"
        signif_col_name = f"{col}_Trend_Signif"
        # If significant (True), mark with "*", otherwise blank
        is_signif = df_stats.set_index('Region')[signif_col_name]
        annot_data[col] = is_signif.apply(
            lambda x: f"{x:.4f} *" if x is True else f"{x:.4f}")  # This is only for display; customizing heatmap annot is complex

    # --- Plotting ---
    plt.figure(figsize=(12, len(slope_data) * 0.5 + 2))

    # Use red-blue palette: red=drying (negative), blue=wetting (positive)
    # center=0 ensures 0 is white
    ax = sns.heatmap(slope_data,
                     annot=True,  # directly display values
                     cmap='RdBu',
                     center=0,
                     fmt='.3f',
                     linewidths=.5,
                     cbar_kws={'label': 'SPI Change per Year (Slope)'})

    # --- Manually add asterisk markers ---
    # Iterate cells and check significance
    for y in range(slope_data.shape[0]):
        for x in range(slope_data.shape[1]):
            region = slope_data.index[y]
            col_name = slope_data.columns[x]  # SPI_1, SPI_3...

            # Find corresponding Signif value
            original_col = f"{col_name}_Trend_Signif"
            is_significant = df_stats[df_stats['Region'] == region][original_col].values[0]

            if is_significant:
                # Draw a star slightly above the cell center
                plt.text(x + 0.5, y + 0.3, '*',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='black', fontsize=14, weight='bold')

    plt.title('Short-term SPI Trends (2017-2022)\n(* indicates statistical significance p<0.05)', fontsize=14)
    plt.ylabel('Region', fontsize=12)
    plt.tight_layout()

    if save_dir:
        import os
        save_path = os.path.join(save_dir, "heatmap_spi_trends.png")
        plt.savefig(save_path, dpi=300)
        print(f"✓ Trend heatmap saved: {save_path}")


def plot_temporal_heatmap(df_spi, timescale='SPI_3', save_dir=None):
    """
    Plot an SPI spatiotemporal evolution heatmap (optimized)
    Improvements:
    1. Time label formatting (YYYY-mm)
    2. Sort regions by mean SPI (driest on top)
    3. Add yearly separator lines
    """

    # --- 1. Data aggregation ---
    # Ensure 'time' column is datetime
    df_spi['time'] = pd.to_datetime(df_spi['time'])

    # Aggregate: compute average monthly SPI per region
    region_time_avg = df_spi.groupby(['region', 'time'])[timescale].mean().reset_index()

    # --- 2. Pivot ---
    pivot_df = region_time_avg.pivot(index='region', columns='time', values=timescale)

    # --- Optimization A: Smart sorting (by dryness) ---
    # Compute mean SPI per region and sort ascending (lower= drier)
    # This places the driest (reddest) regions at the top for visual emphasis
    mean_val = pivot_df.mean(axis=1).sort_values()
    pivot_df = pivot_df.reindex(mean_val.index)

    # --- 3. Plot ---
    plt.figure(figsize=(16, 8))  # slightly wider to fit the timeline

    # Draw heatmap
    ax = sns.heatmap(pivot_df, cmap='RdBu', center=0, vmin=-2.5, vmax=2.5,
                     cbar_kws={'label': f'{timescale} Value', 'shrink': 0.8})

    # --- Optimization B: Beautify x-axis labels ---
    # Get original x-axis tick positions (0, 1, 2...)
    # Extract corresponding time labels
    x_dates = pivot_df.columns

    new_ticks = []
    new_labels = []

    for i, date in enumerate(x_dates):
        # Strategy: show labels only for January and July each year
        if date.month in [1, 7]:
            new_ticks.append(i + 0.5)  # +0.5 to center labels on cells
            new_labels.append(date.strftime('%Y-%m'))  # Format as 2017-01

    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels, rotation=45, ha='right', fontsize=10)

    # --- Optimization C: Add yearly separator lines ---
    # Draw a vertical dashed line at each January
    for i, date in enumerate(x_dates):
        if date.month == 1 and i > 0:  # i>0 to avoid drawing on the far left
            plt.axvline(i, color='black', linestyle='--', linewidth=0.7, alpha=0.5)

    # Title and labels
    plt.title(f'Spatiotemporal Evolution of {timescale} (2017-2022)\n(Sorted by Mean Dryness: Driest Regions on Top)',
              fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Region', fontsize=12)

    plt.tight_layout()

    # Save logic
    if save_dir:
        # Dynamic filename to avoid overwriting other timescales
        filename = f"heatmap_temporal_{timescale}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300)
        print(f"✓ Optimized heatmap saved: {save_path}")