"""
SYRS Yield Analysis Module
Complete workflow for Standardized Yield Residual Series calculation
All functions in one file for complete analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import utility functions
from utils.data_io import load_processed_data, save_processed_data
from utils.yield_analysis_utils import (
    calculate_yield_per_ha,
    validate_yield_data,
    linear_detrending,
    polynomial_detrending,
    moving_average_detrending,
    standardize_residuals,
    interpret_syrs_value,
    compare_detrending_methods
)


def _load_yield_data(file_path):
    """
    Load and prepare yield data from Excel file

    Args:
        file_path: Path to Excel file

    Returns:
        DataFrame: Cleaned yield data
    """
    print("=" * 80)
    print("Step 1: Loading Yield Data")
    print("=" * 80)

    # Load Excel data
    df = load_processed_data(file_path, format="xlsx")
    print(f"✓ Loaded data: {len(df)} rows")

    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

    # Calculate yield per hectare
    print("Calculating yield per hectare...")
    yield_values = []
    if 'yield_ton_per_ha' not in df.columns:
        for idx, row in df.iterrows():
            yield_val = calculate_yield_per_ha(
                row.get('production'),
                row.get('planted_area')
            )
            yield_values.append(yield_val)

        df['yield_ton_per_ha'] = yield_values

    # Remove rows with invalid yield
    df_clean = df.dropna(subset=['yield_ton_per_ha']).copy()
    print(f"✓ After cleaning: {len(df_clean)} rows")
    print(f"✓ Removed rows: {len(df) - len(df_clean)}")

    # Show data statistics by state
    print("\nState Data Statistics:")
    print(f"{'State':<20} {'Years':<10} {'Avg Yield (ton/ha)':<20}")
    print("-" * 50)

    for state in sorted(df_clean['state'].unique()):
        if state == 'Malaysia':  # Skip national data
            continue

        state_data = df_clean[df_clean['state'] == state]
        avg_yield = state_data['yield_ton_per_ha'].mean()
        print(f"{state:<20} {len(state_data):<10} {avg_yield:<20.3f}")

    return df_clean


def _calculate_single_state_syrs(state_data):
    """
    Calculate SYRS for a single state using all detrending methods

    Args:
        state_data: DataFrame for a single state

    Returns:
        tuple: (results_df, method_metrics)
    """
    if len(state_data) < 3:
        return None, None

    # Extract years and yields
    years = state_data['year'].values.reshape(-1, 1)
    yields = state_data['yield_ton_per_ha'].values

    # Validate data
    if not validate_yield_data(yields, min_years=3):
        return None, None

    results = state_data.copy()
    method_metrics = {}

    # Method 1: Linear detrending
    try:
        trend_linear, residuals_linear, model_linear, metrics_linear = linear_detrending(years, yields)
        syrs_linear = standardize_residuals(residuals_linear)

        results['trend_linear'] = trend_linear
        results['residual_linear'] = residuals_linear
        results['SYRS_linear'] = syrs_linear

        method_metrics['linear'] = metrics_linear
        method_metrics['linear']['method'] = 'linear'
    except Exception as e:
        print(f"  Linear detrending failed: {e}")
        method_metrics['linear'] = None

    # Method 2: Polynomial detrending
    try:
        trend_poly, residuals_poly, model_poly, metrics_poly = polynomial_detrending(years, yields, degree=2)
        syrs_poly = standardize_residuals(residuals_poly)

        results['trend_polynomial'] = trend_poly
        results['residual_polynomial'] = residuals_poly
        results['SYRS_polynomial'] = syrs_poly

        method_metrics['polynomial'] = metrics_poly
        method_metrics['polynomial']['method'] = 'polynomial'
    except Exception as e:
        print(f"  Polynomial detrending failed: {e}")
        method_metrics['polynomial'] = None

    # Method 3: Moving average detrending
    try:
        trend_ma, residuals_ma, metrics_ma = moving_average_detrending(yields)
        syrs_ma = standardize_residuals(residuals_ma)

        results['trend_moving_average'] = trend_ma
        results['residual_moving_average'] = residuals_ma
        results['SYRS_moving_average'] = syrs_ma

        method_metrics['moving_average'] = metrics_ma
        method_metrics['moving_average']['method'] = 'moving_average'
    except Exception as e:
        print(f"  Moving average detrending failed: {e}")
        method_metrics['moving_average'] = None

    return results, method_metrics


def _select_best_detrending_method(state_name, method_metrics, verbose=True):
    """
    Select best detrending method for a state

    Args:
        state_name: Name of the state
        method_metrics: Dictionary of method metrics
        verbose: Whether to print details

    Returns:
        str: Best method name
    """
    if verbose:
        print(f"\n【{state_name}】Method Selection")
        print(f"{'Method':<20} {'R²':>8} {'AIC':>10} {'MSE':>10} {'Residual Std':>12}")
        print("-" * 60)

    valid_methods = {}

    for method, metrics in method_metrics.items():
        if metrics is not None:
            valid_methods[method] = metrics

            if verbose:
                print(f"{method:<20} {metrics['r2']:>8.4f} {metrics['aic']:>10.2f} "
                      f"{metrics['mse']:>10.4f} {metrics['residual_std']:>12.4f}")

    if not valid_methods:
        return None

    # Select best method
    best_method = compare_detrending_methods(method_metrics)

    if verbose and best_method:
        print(f"\n✓ Selected: {best_method} (lowest AIC)")

    return best_method


def _process_all_states_syrs(df_clean, min_years=3):
    """
    Process SYRS calculation for all states

    Args:
        df_clean: Cleaned yield DataFrame
        min_years: Minimum years required

    Returns:
        tuple: (syrs_df, method_log_df)
    """
    print("\n" + "=" * 80)
    print("Step 2: Calculating SYRS for All States")
    print("=" * 80)

    all_results = []
    method_selection_log = []

    # Get all states (exclude Malaysia)
    states = [s for s in df_clean['state'].unique() if s != 'Malaysia']
    print(f"\nProcessing {len(states)} states\n")

    for state in sorted(states):
        print(f"\n{'=' * 60}")
        print(f"Processing: {state}")
        print(f"{'=' * 60}")

        # Extract state data
        state_data = df_clean[df_clean['state'] == state].sort_values('year').copy()

        # Check data sufficiency
        if len(state_data) < min_years:
            print(f"❌ Skipped: Insufficient data ({len(state_data)} years)")
            continue

        print(f"Data years: {state_data['year'].min():.0f} - {state_data['year'].max():.0f}")
        print(f"Data points: {len(state_data)}")
        print(f"Average yield: {state_data['yield_ton_per_ha'].mean():.3f} ton/ha")

        # Calculate SYRS with all methods
        results, method_metrics = _calculate_single_state_syrs(state_data)

        if results is None:
            print(f"❌ Skipped: Calculation failed")
            continue

        # Select best method
        best_method = _select_best_detrending_method(state, method_metrics, verbose=True)

        if best_method is None:
            print(f"❌ Skipped: No valid method")
            continue

        # Use best method results
        results['state'] = state
        results['selected_method'] = best_method
        results['SYRS'] = results[f'SYRS_{best_method}']
        results['trend_yield'] = results[f'trend_{best_method}']
        results['residual'] = results[f'residual_{best_method}']

        # Add method metrics
        if best_method in method_metrics and method_metrics[best_method] is not None:
            for key, value in method_metrics[best_method].items():
                if key != 'method':  # Don't duplicate method column
                    results[f'method_{key}'] = value

        # Interpret SYRS values
        climate_info = results['SYRS'].apply(interpret_syrs_value)
        results['climate_category'], results['climate_description'] = zip(*climate_info)

        all_results.append(results)

        # Log method selection
        if best_method in method_metrics and method_metrics[best_method] is not None:
            metrics = method_metrics[best_method]
            method_selection_log.append({
                'state': state,
                'selected_method': best_method,
                'r2': metrics.get('r2', np.nan),
                'aic': metrics.get('aic', np.nan),
                'mse': metrics.get('mse', np.nan),
                'residual_std': metrics.get('residual_std', np.nan),
                'n_samples': len(results),
                'n_params': metrics.get('n_params', np.nan)
            })

        # Show SYRS preview
        print(f"\nSYRS Results Preview:")
        print(f"{'Year':<8} {'Actual Yield':<12} {'Trend Yield':<12} {'Residual':<10} {'SYRS':<10} {'Category'}")
        print("-" * 65)

        preview_cols = ['year', 'yield_ton_per_ha', 'trend_yield', 'residual', 'SYRS', 'climate_category']
        preview_data = results[preview_cols].head(5)

        for _, row in preview_data.iterrows():
            print(f"{row['year']:<8.0f} {row['yield_ton_per_ha']:<12.3f} "
                  f"{row['trend_yield']:<12.3f} {row['residual']:<10.3f} "
                  f"{row['SYRS']:<10.2f} {row['climate_category'][:15]}")

    if all_results:
        syrs_df = pd.concat(all_results, ignore_index=True)
        method_log_df = pd.DataFrame(method_selection_log)
        return syrs_df, method_log_df
    else:
        return pd.DataFrame(), pd.DataFrame()


def _analyze_syrs_results(syrs_df, method_log_df):
    """
    Analyze SYRS calculation results

    Args:
        syrs_df: SYRS results DataFrame
        method_log_df: Method selection log DataFrame
    """
    print("\n" + "=" * 80)
    print("Step 3: Results Analysis")
    print("=" * 80)

    # 1. Method selection statistics
    print("\n【Method Selection Statistics】")
    if not method_log_df.empty:
        method_counts = method_log_df['selected_method'].value_counts()

        for method, count in method_counts.items():
            pct = count / len(method_log_df) * 100
            print(f"  {method:<20} {count:>3} states ({pct:>5.1f}%)")
    else:
        print("  No method selection data")

    # 2. State SYRS statistics
    print("\n【State SYRS Statistics】")
    if not syrs_df.empty:
        print(f"{'State':<20} {'Years':<10} {'Avg SYRS':<12} {'Min SYRS':<12} {'Max SYRS':<12} {'Method'}")
        print("-" * 85)

        for state in sorted(syrs_df['state'].unique()):
            state_data = syrs_df[syrs_df['state'] == state]
            method = state_data['selected_method'].iloc[0]

            print(f"{state:<20} {len(state_data):<10} "
                  f"{state_data['SYRS'].mean():<12.3f} "
                  f"{state_data['SYRS'].min():<12.3f} "
                  f"{state_data['SYRS'].max():<12.3f} "
                  f"{method}")
    else:
        print("  No SYRS data")

    # 3. Climate category distribution
    print("\n【Climate Category Distribution】")
    if not syrs_df.empty:
        climate_counts = syrs_df['climate_category'].value_counts()

        for category, count in climate_counts.items():
            pct = count / len(syrs_df) * 100
            print(f"  {category:<25} {count:>4} years ({pct:>5.1f}%)")
    else:
        print("  No climate category data")

    # 4. Extreme years identification
    print("\n【Extreme Climate Years (|SYRS| > 1.5)】")
    if not syrs_df.empty:
        extreme = syrs_df[abs(syrs_df['SYRS']) > 1.5].sort_values('SYRS', ascending=False)

        if len(extreme) > 0:
            print(f"{'State':<20} {'Year':<8} {'SYRS':<10} {'Category'}")
            print("-" * 60)

            for _, row in extreme.head(10).iterrows():  # Show top 10
                print(f"{row['state']:<20} {row['year']:<8.0f} "
                      f"{row['SYRS']:<10.2f} {row['climate_category']}")
        else:
            print("  No extreme years found")
    else:
        print("  No data for extreme years analysis")


def _create_syrs_time_series_plot(syrs_df, output_path):
    """
    Create SYRS time series plot

    Args:
        syrs_df: SYRS results DataFrame
        output_path: Output file path
    """
    if syrs_df.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 10))

    states = sorted(syrs_df['state'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(states)))

    for i, state in enumerate(states):
        state_data = syrs_df[syrs_df['state'] == state].sort_values('year')
        ax.plot(state_data['year'], state_data['SYRS'],
                marker='o', label=state, color=colors[i], linewidth=2)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('SYRS', fontsize=14)
    ax.set_title('SYRS Time Series by State', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_syrs_boxplot(syrs_df, output_path):
    """
    Create SYRS distribution boxplot

    Args:
        syrs_df: SYRS results DataFrame
        output_path: Output file path
    """
    if syrs_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    states = sorted(syrs_df['state'].unique())
    syrs_by_state = [syrs_df[syrs_df['state'] == s]['SYRS'].values for s in states]

    bp = ax.boxplot(syrs_by_state, labels=states, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.set_ylabel('SYRS', fontsize=14)
    ax.set_title('SYRS Distribution by State', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_trend_comparison_plot(syrs_df, output_path):
    """
    Create yield trend comparison plot

    Args:
        syrs_df: SYRS results DataFrame
        output_path: Output file path
    """
    if syrs_df.empty:
        return

    example_states = sorted(syrs_df['state'].unique())[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, state in enumerate(example_states):
        if idx >= len(axes):
            break

        ax = axes[idx]
        state_data = syrs_df[syrs_df['state'] == state].sort_values('year')

        years = state_data['year'].values
        actual = state_data['yield_ton_per_ha'].values
        trend = state_data['trend_yield'].values

        ax.plot(years, actual, 'o-', color='blue', linewidth=2,
                markersize=6, label='Actual Yield')
        ax.plot(years, trend, '--', color='red', linewidth=2,
                label='Trend', alpha=0.7)

        # Mark extreme points
        extreme = state_data[abs(state_data['SYRS']) > 1.5]
        if len(extreme) > 0:
            for _, row in extreme.iterrows():
                color = 'green' if row['SYRS'] > 0 else 'orange'
                ax.scatter(row['year'], row['yield_ton_per_ha'],
                           s=200, c=color, marker='*',
                           edgecolors='black', linewidths=1.5, zorder=5)

        method = state_data['selected_method'].iloc[0]
        slope = state_data['method_slope'].iloc[0] if 'method_slope' in state_data.columns else np.nan

        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Yield (ton/ha)', fontsize=10)
        ax.set_title(f'{state}\nMethod: {method}, Slope: {slope:+.4f}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(example_states), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_syrs_heatmap(syrs_df, output_path):
    """
    Create SYRS heatmap

    Args:
        syrs_df: SYRS results DataFrame
        output_path: Output file path
    """
    if syrs_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    pivot = syrs_df.pivot_table(values='SYRS', index='state', columns='year')

    sns.heatmap(pivot, cmap='RdYlGn', center=0, vmin=-2, vmax=2,
                annot=True, fmt='.2f', linewidths=0.5,
                cbar_kws={'label': 'SYRS'}, ax=ax)

    ax.set_title('SYRS Heatmap (State × Year)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('State', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_syrs_visualizations(syrs_df, output_dir):
    """
    Generate all SYRS visualization charts

    Args:
        syrs_df: SYRS results DataFrame
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("Step 4: Generating Visualizations")
    print("=" * 80)

    # Create figures directory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Generate plots
    print("Creating SYRS time series plot...")
    _create_syrs_time_series_plot(syrs_df, os.path.join(figures_dir, 'syrs_time_series.png'))
    print(f"✓ Saved: {figures_dir}/syrs_time_series.png")

    print("Creating SYRS boxplot...")
    _create_syrs_boxplot(syrs_df, os.path.join(figures_dir, 'syrs_boxplot.png'))
    print(f"✓ Saved: {figures_dir}/syrs_boxplot.png")

    print("Creating trend comparison plot...")
    _create_trend_comparison_plot(syrs_df, os.path.join(figures_dir, 'syrs_trends_comparison.png'))
    print(f"✓ Saved: {figures_dir}/syrs_trends_comparison.png")

    print("Creating SYRS heatmap...")
    _create_syrs_heatmap(syrs_df, os.path.join(figures_dir, 'syrs_heatmap.png'))
    print(f"✓ Saved: {figures_dir}/syrs_heatmap.png")

    print("\n✓ All visualizations generated successfully!")


def _export_syrs_results(syrs_df, method_log_df, output_dir):
    """
    Export SYRS analysis results

    Args:
        syrs_df: SYRS results DataFrame
        method_log_df: Method selection log DataFrame
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("Step 5: Exporting Results")
    print("=" * 80)

    # Create output directories
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Export detailed SYRS data
    output_cols = ['state', 'year', 'planted_area', 'production',
                   'yield_ton_per_ha', 'trend_yield', 'residual', 'SYRS',
                   'climate_category', 'climate_description',
                   'selected_method', 'method_r2', 'method_aic',
                   'method_slope', 'method_residual_std']

    # Filter available columns
    available_cols = [col for col in output_cols if col in syrs_df.columns]
    syrs_export = syrs_df[available_cols].copy()

    detailed_path = os.path.join(output_dir, 'syrs_detailed.csv')
    save_processed_data(syrs_export, detailed_path)
    print(f"✓ Detailed SYRS data: {detailed_path}")

    # 2. Export method selection log
    if not method_log_df.empty:
        method_log_path = os.path.join(tables_dir, 'method_selection_log.csv')
        save_processed_data(method_log_df, method_log_path)
        print(f"✓ Method selection log: {method_log_path}")

    # 3. Export state summary statistics
    if not syrs_df.empty:
        summary_data = []

        for state in sorted(syrs_df['state'].unique()):
            state_data = syrs_df[syrs_df['state'] == state]

            summary_data.append({
                'state': state,
                'years': len(state_data),
                'method': state_data['selected_method'].iloc[0],
                'avg_yield': state_data['yield_ton_per_ha'].mean(),
                'trend_slope': state_data['method_slope'].iloc[0] if 'method_slope' in state_data.columns else np.nan,
                'avg_SYRS': state_data['SYRS'].mean(),
                'std_SYRS': state_data['SYRS'].std(),
                'min_SYRS': state_data['SYRS'].min(),
                'max_SYRS': state_data['SYRS'].max(),
                'favorable_years': (state_data['SYRS'] > 0.5).sum(),
                'normal_years': ((state_data['SYRS'] >= -0.5) & (state_data['SYRS'] <= 0.5)).sum(),
                'unfavorable_years': (state_data['SYRS'] < -0.5).sum()
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(tables_dir, 'syrs_summary_by_state.csv')
        save_processed_data(summary_df, summary_path)
        print(f"✓ State summary statistics: {summary_path}")

    # 4. Export comprehensive Excel report
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    report_path = os.path.join(reports_dir, 'syrs_complete_report.xlsx')

    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # Sheet 1: Detailed SYRS
        syrs_export.to_excel(writer, sheet_name='Detailed_SYRS', index=False)

        # Sheet 2: Summary statistics
        if not syrs_df.empty:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 3: Method selection
        if not method_log_df.empty:
            method_log_df.to_excel(writer, sheet_name='Method_Selection', index=False)

        # Sheet 4: SYRS matrix
        if not syrs_df.empty:
            pivot = syrs_df.pivot_table(values='SYRS', index='state', columns='year')
            pivot.to_excel(writer, sheet_name='SYRS_Matrix')

    print(f"✓ Comprehensive report: {report_path}")


def run_complete_syrs_analysis(config):
    """
    Execute complete SYRS yield analysis
    One-line function to call the complete workflow

    Args:
        config: Configuration dictionary with keys:
            - input_file: Path to yield data Excel file
            - output_dir: Output directory path
            - min_years: Minimum years of data required
            - exclude_national: Whether to exclude national data

    Returns:
        dict: Analysis results and metadata
                'input_file': './datasets/malaysia/rawdata/crops_state_part.xlsx',
        'output_dir': './datasets/malaysia/processed/syrs_analysis',
        'min_data_years': 3,
        'exclude_national': True  # 排除全国数据
    """
    # Set default configuration
    # config = config or {}

    input_file = f'./datasets/{config["country"]}/rawdata/crops_state.xlsx'
    output_dir = f'./datasets/{config["country"]}/processed/{config["run_tag"]}/syrs_analysis'
    min_years = config.get('min_years', 3)
    exclude_national = config.get('exclude_national', True)

    results = {
        'success': False,
        'output_directory': output_dir,
        'input_file': input_file,
        'config': config
    }

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load and prepare data
        df_clean = _load_yield_data(input_file)

        if len(df_clean) == 0:
            print("\n❌ Error: No valid data found")
            results['error'] = 'No valid data'
            return results

        # Exclude national data if requested
        if exclude_national and 'Malaysia' in df_clean['state'].unique():
            df_clean = df_clean[df_clean['state'] != 'Malaysia'].copy()
            print(f"\n✓ Excluded national data (Malaysia)")

        # Step 2: Process SYRS for all states
        syrs_df, method_log_df = _process_all_states_syrs(df_clean, min_years)

        if syrs_df.empty:
            print("\n❌ Error: No SYRS calculated for any state")
            results['error'] = 'No SYRS calculated'
            return results

        # Store results
        results['syrs_data'] = syrs_df
        results['method_log'] = method_log_df
        results['states_processed'] = len(syrs_df['state'].unique())
        results['years_covered'] = f"{syrs_df['year'].min()}-{syrs_df['year'].max()}"

        # Step 3: Analyze results
        _analyze_syrs_results(syrs_df, method_log_df)

        # Step 4: Generate visualizations
        _generate_syrs_visualizations(syrs_df, output_dir)

        # Step 5: Export results
        _export_syrs_results(syrs_df, method_log_df, output_dir)

        # draw production graph
        # convert date
        states = df_clean['state'].unique()

        # 3. set figure
        plt.figure(figsize=(14, 8))

        # 4. draw
        for state in ["Kedah", "Perlis", "Selangor" ]:
            state_df = df_clean[df_clean['state'] == state]
            plt.plot(state_df['year'], state_df['yield_ton_per_ha'],
                     marker='o', linewidth=2, markersize=6, label=state)

        # 5. setting
        plt.title('Rice Production Per Year by State (2017-2022)', fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Production Per Year (tons/ha)', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/production_by_state.png', dpi=300, bbox_inches='tight')
        plt.close()


        # Final summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"✓ Processed {results['states_processed']} states")
        print(f"✓ Time period: {results['years_covered']}")
        print(f"✓ Results saved to: {output_dir}")
        print(f"\nMain output files:")
        print(f"  1. tables/syrs_detailed.csv - Detailed SYRS data")
        print(f"  2. tables/syrs_summary_by_state.csv - State summaries")
        print(f"  3. reports/syrs_complete_report.xlsx - Excel report")
        print(f"  4. figures/ - Visualization charts")

        results['success'] = True
        results['analysis_complete'] = True

        return results

    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

        results['error'] = str(e)
        return results