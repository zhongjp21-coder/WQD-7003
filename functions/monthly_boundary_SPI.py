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
    monthly_path = paths['processed']/f"monthly_precipitation_cleaned.{output_format}"
    save_processed_data(df_clean, monthly_path, format=output_format)

    print(f"\n8. SPI CALCULATION")
    print("-" * 40)
    spi_params = {
        'timescales': params.get('timescales', [1, 3, 6, 12]),
        'distribution': params.get('distribution', 'gamma')
    }

    df_spi = calculate_spi(df_clean, **spi_params)

    # Save SPI results
    spi_path = paths['processed']/ f"spi_results.{output_format}"
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

    metadata_path = paths['processed']/ "pipeline_metadata.json"
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

    # 统计各州指标
    print("df_spi",df_spi)
    print(df_spi.columns)
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

    # 格式化输出
    state_summary = state_summary.round({
        'average precipitation(mm)': 2,
        'SPI caculate succeed %': 1
    })

    return {
        'monthly_data': df_clean,
        'spi_data': df_spi,
        'grid_metadata': grid_meta,
        'qc_report': qc_report,
        'metadata': metadata,
        'paths': paths
    }