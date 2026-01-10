"""
CMIP6 Climate Analysis Module
Complete workflow for CMIP6 multi-model climate data analysis
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
import warnings

from utils.data_io import load_processed_data

warnings.filterwarnings('ignore')


def _read_excel_sheets(excel_path):
    """Read all sheets from CMIP6 Excel file"""
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names
    print(f"Found {len(sheet_names)} sheets: {sheet_names}")

    all_data = {}
    for sheet in sheet_names:
        df = xl.parse(sheet)
        print(f"  Sheet '{sheet}' shape: {df.shape}")
        all_data[sheet] = df

    return all_data, sheet_names


def _standardize_model_data(df, model_name):
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


def _calculate_spi_for_model(precip_series, scale=3, historical_years=(2015, 2024)):
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


def _extract_growing_season(data_series, growth_data, state):
    """Extract growing season data"""
    if data_series.empty:
        return pd.Series([], dtype=float)

    months = data_series.index.month

    seasons = growth_data[growth_data["State"] == state]["Growth_Months"]
    seasons = [int(x) for x in seasons.values[0].split(",")]
    mask = months.isin(seasons)

    growing_data = data_series[mask].copy()

    if growing_data.empty:
        return pd.Series([], dtype=float)

    # Calculate annual averages
    annual_means = []
    annual_years = []

    min_year = data_series.index.year.min()
    max_year = data_series.index.year.max()

    for year in range(min_year, max_year + 1):
        # if season_type == 'main':
        #     year_mask = ((data_series.index.year == year - 1) & data_series.index.month.isin([9, 10, 11, 12])) | \
        #                 ((data_series.index.year == year) & data_series.index.month.isin([1, 2]))
        # else:
        #     year_mask = (data_series.index.year == year) & data_series.index.month.isin([3, 4, 5, 6, 7, 8])

        year_mask = (data_series.index.year == year) & data_series.index.month.isin(seasons)

        year_data = data_series[year_mask]
        if len(year_data) >= 3:
            annual_means.append(year_data.mean())
            annual_years.append(year)

    return pd.Series(annual_means, index=annual_years, name='annual_spi')


def _perform_mann_kendall_test(annual_series, alpha=0.05):
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


def _load_and_process_data(excel_path):
    """Step 1: Load and process all model data"""
    all_data, sheet_names = _read_excel_sheets(excel_path)

    all_models_data = []

    for sheet in sheet_names:
        print(f"  Processing model: {sheet}")
        df = all_data[sheet]

        try:
            df_clean = _standardize_model_data(df, sheet)
            all_models_data.append(df_clean)
        except Exception as e:
            print(f"  Processing failed: {e}")
            continue

    if not all_models_data:
        raise ValueError("No data processed successfully!")

    combined_data = pd.concat(all_models_data, ignore_index=True)
    print(f"Combined total records: {len(combined_data)}")

    return combined_data


def _calculate_spi_for_all_models(data, historical_years, output_dir):
    """Step 2: Calculate SPI for all models"""
    spi_results = []

    for (state, model), group in data.groupby(['name', 'model']):
        group = group.set_index('date').sort_index()
        precip_series = group['precip']

        spi_series = _calculate_spi_for_model(precip_series, scale=3, historical_years=historical_years)

        if spi_series.isna().all():
            continue

        spi_df = pd.DataFrame({
            'date': spi_series.index,
            'state': state,
            'model': model,
            'precip': precip_series.reindex(spi_series.index),
            'spi_3month': spi_series.values
        })
        spi_results.append(spi_df)

    spi_all = pd.concat(spi_results, ignore_index=True)

    if spi_all.empty:
        raise ValueError("SPI calculation results empty!")

    # Save individual model results
    spi_all.to_csv(f'{output_dir}/spi_individual_models.csv', index=False)
    print(f"Individual model SPI saved: {output_dir}/spi_individual_models.csv")

    return spi_all


def _calculate_ensemble_statistics(spi_data, output_dir):
    """Step 3: Calculate multi-model ensemble statistics"""
    ensemble_stats = []

    for (state, date), group in spi_data.groupby(['state', 'date']):
        if len(group) >= 2:
            spi_values = group['spi_3month'].dropna()

            if len(spi_values) > 0:
                ensemble_stats.append({
                    'state': state,
                    'date': date,
                    'spi_median': np.median(spi_values),
                    'spi_p25': np.percentile(spi_values, 25),
                    'spi_p75': np.percentile(spi_values, 75),
                    'model_count': len(spi_values)
                })

    if not ensemble_stats:
        raise ValueError("Ensemble statistics calculation failed!")

    ensemble_df = pd.DataFrame(ensemble_stats)
    ensemble_df.to_csv(f'{output_dir}/spi_ensemble_stats.csv', index=False)
    print(f"Ensemble statistics saved: {len(ensemble_df)} records")

    return ensemble_df


def _extract_annual_growing_season(ensemble_data, growth_data, output_dir):
    """Step 4: Extract annual growing season SPI from ensemble"""
    annual_spi_results = []

    for state in ensemble_data['state'].unique():
        if state not in growth_data['State'].values:
            print(f"=====NO STATE===={state}")
            continue
        state_data = ensemble_data[ensemble_data['state'] == state].copy()
        state_data['date'] = pd.to_datetime(state_data['date'])
        state_data = state_data.set_index('date').sort_index()

        growing_series = _extract_growing_season(state_data['spi_median'], growth_data, state)

        if len(growing_series) > 0:
            for year, spi_value in growing_series.items():
                annual_spi_results.append({
                    'state': state,
                    'year': int(year),
                    'annual_spi': spi_value
                })

    if not annual_spi_results:
        raise ValueError("Growing season SPI extraction failed!")

    annual_spi_df = pd.DataFrame(annual_spi_results)
    annual_spi_df.to_csv(f'{output_dir}/annual_spi_ensemble.csv', index=False)
    print(f"Annual SPI saved: {len(annual_spi_df)} records")

    return annual_spi_df


def _perform_trend_analysis(annual_data, analysis_period, ensemble_data, output_dir):
    """Step 5: Perform trend analysis on annual data"""
    trend_results = []

    for state in annual_data['state'].unique():
        state_data = annual_data[annual_data['state'] == state].sort_values('year')

        # Analyze data from specified period
        future_data = state_data[(state_data['year'] >= analysis_period[0]) & (state_data['year'] <= analysis_period[1])]

        if len(future_data) < 8:
            print(f"  {state}: Insufficient data ({len(future_data)} < 8)")
            continue

        # Perform MK test
        mk_result = _perform_mann_kendall_test(future_data['annual_spi'].values)

        # Calculate uncertainty range
        if len(ensemble_data) > 0:
            state_ensemble = ensemble_data[ensemble_data['state'] == state]
            uncertainty = state_ensemble['spi_p75'].mean() - state_ensemble['spi_p25'].mean()
        else:
            uncertainty = np.nan

        trend_results.append({
            'state': state,
            'period': f'{analysis_period[0]}-{analysis_period[1]}',
            'n_years': len(future_data),
            'trend': mk_result['trend'],
            'p_value': mk_result['p'],
            'slope': mk_result['slope'],
            'mean_spi': future_data['annual_spi'].mean(),
            'uncertainty_range': uncertainty
        })
        print(f"  {state}: {mk_result['trend']}, p={mk_result['p']:.4f}, slope={mk_result['slope']:.6f}")

    if not trend_results:
        raise ValueError("Trend analysis produced no results!")

    trend_df = pd.DataFrame(trend_results)
    trend_df.to_csv(f'{output_dir}/trend_analysis_results.csv', index=False)
    print(f"Trend analysis saved: {len(trend_df)} records")

    return trend_df


def _generate_visualizations(trend_data, ensemble_data, growth_data, output_dir):
    """Step 6: Generate visualization charts"""
    sns.set_style("whitegrid")
    figures_dir = f'{output_dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Trend bar chart
    if not trend_data.empty and 'slope' in trend_data.columns:
        plt.figure(figsize=(12, 8))

        trend_sorted = trend_data.sort_values('slope')
        colors = []
        for _, row in trend_sorted.iterrows():
            if row['p_value'] < 0.05:
                colors.append('red' if row['slope'] < 0 else 'blue')
            else:
                colors.append('lightcoral' if row['slope'] < 0 else 'lightblue')

        plt.barh(range(len(trend_sorted)), trend_sorted['slope'], color=colors)
        plt.yticks(range(len(trend_sorted)), trend_sorted['state'])
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Significantly wetter (p<0.05)'),
            Patch(facecolor='red', label='Significantly drier (p<0.05)'),
            Patch(facecolor='lightblue', label='Non-significant wetter'),
            Patch(facecolor='lightcoral', label='Non-significant drier')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.xlabel('Trend Slope (SPI/year)', fontsize=12)
        plt.title('Drought Trends in Rice Growing Season (Multi-Model Median)', fontsize=14, pad=20)
        plt.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{figures_dir}/drought_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trend bar chart saved: {figures_dir}/drought_trends.png")

    # 2. Time series for representative states
    if not ensemble_data.empty:
        if "Kedah" in ensemble_data['state'].values:
            rep_states = ['Kedah', 'Kelantan', 'Johor', 'Selangor']
        else:
            rep_states = ['Guangxi', 'Hainan']

        for state in rep_states:
            state_data = ensemble_data[ensemble_data['state'] == state].copy()

            if len(state_data) > 0:
                state_data['date'] = pd.to_datetime(state_data['date'])
                state_data = state_data.set_index('date').sort_index()

                growing_median = _extract_growing_season(state_data['spi_median'], growth_data, state)

                if len(growing_median) > 0:
                    plt.figure(figsize=(12, 6))

                    years = [int(year) for year in growing_median.index]
                    spi_values = growing_median.values

                    plt.plot(years, spi_values, 'b-', linewidth=2, label='SPI Median')
                    plt.fill_between(years, -0.99, 0.99, alpha=0.1, color='gray', label='Normal Range')
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    plt.axhline(y=-1.5, color='red', linestyle=':', alpha=0.5, label='Moderate Drought')

                    if len(years) >= 8:
                        x = np.arange(len(years))
                        slope, intercept, _, _, _ = stats.linregress(x, spi_values)
                        trend_line = intercept + slope * x
                        plt.plot(years, trend_line, 'r--', linewidth=1.5,
                                 label=f'Trend (slope={slope:.4f}/year)')

                    plt.xlabel('Year', fontsize=12)
                    plt.ylabel('SPI (3-month)', fontsize=12)
                    plt.title(f'{state} - Rice Growing Season Drought Index', fontsize=14)
                    plt.legend(loc='best')
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(f'{figures_dir}/timeseries_{state}.png', dpi=300, bbox_inches='tight')
                    plt.close()

        print(f"Time series charts saved")

    print("Visualizations generated successfully!")


def _export_comprehensive_report(results, output_dir):
    """Step 7: Export comprehensive analysis report"""
    report_path = f'{output_dir}/climate_analysis_report.xlsx'

    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = []
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                summary_data.append({
                    'Dataset': key,
                    'Rows': len(value),
                    'Columns': len(value.columns)
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Data sheets
        for key, value in results.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                sheet_name = key[:31]  # Excel sheet name limit
                value.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Comprehensive report saved: {report_path}")


def _print_analysis_summary(trend_data, output_dir):
    """Print analysis summary"""
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 60)

    if not trend_data.empty:
        trend_counts = trend_data['trend'].value_counts()
        print(f"Trend distribution:")
        for trend, count in trend_counts.items():
            print(f"  {trend}: {count} states")

        significant = trend_data[trend_data['p_value'] < 0.05]
        print(f"\nSignificant trends (p < 0.05): {len(significant)} states")

        if len(significant) > 0:
            print("\nState trend details (sorted by slope):")
            for _, row in significant.sort_values('slope').iterrows():
                trend_symbol = "↑" if row['slope'] > 0 else "↓"
                print(f"  {row['state']}: {trend_symbol} slope={row['slope']:.6f}, p={row['p_value']:.4f}")

    print(f"\nResults saved to: {output_dir}")


def run_cmip_climate_analysis(config):
    """
    Execute complete CMIP6 climate analysis
    One-line function to call the complete workflow

    Args:
        config: Configuration dictionary with keys:
            - excel_path: Path to CMIP6 Excel file
            - output_dir: Output directory path
            - historical_years: Tuple of (start_year, end_year) for SPI calibration
            - analysis_period: Tuple of (start_year, end_year) for trend analysis

    Returns:
        dict: Analysis results and metadata
    """
    excel_path = f"./datasets/{config['country']}/rawdata/cmip6-4models.xlsx"
    growth_path = f"./datasets/{config['country']}/processed/{config['run_tag']}/growth_seasons/state_growth_months.csv"
    output_dir = config.get('output_dir', f"./datasets/{config['country']}/processed/{config['run_tag']}/cmip_analysis")
    historical_years = (2015, 2024)
    analysis_period = (2025, 2100)

    results = {
        'success': False,
        'output_directory': output_dir,
        'excel_path': excel_path,
        'config': config
    }

    try:
        print("=" * 60)
        print("CMIP6 Climate Analysis - Multi-Model Version")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Execute all analysis steps
        combined_data = _load_and_process_data(excel_path)
        results['combined_data'] = combined_data

        # Execute all analysis steps
        growth_data = pd.read_csv(growth_path)

        spi_data = _calculate_spi_for_all_models(combined_data, historical_years, output_dir)
        results['spi_individual'] = spi_data

        ensemble_data = _calculate_ensemble_statistics(spi_data, output_dir)
        results['ensemble_stats'] = ensemble_data

        annual_data = _extract_annual_growing_season(ensemble_data, growth_data, output_dir)
        results['annual_spi'] = annual_data

        trend_data = _perform_trend_analysis(annual_data, analysis_period, ensemble_data, output_dir)
        results['trend_analysis'] = trend_data

        _generate_visualizations(trend_data, ensemble_data, growth_data, output_dir)

        _export_comprehensive_report(results, output_dir)

        _print_analysis_summary(trend_data, output_dir)

        results['success'] = True
        results['analysis_complete'] = True

        return results

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

        results['error'] = str(e)
        return results
