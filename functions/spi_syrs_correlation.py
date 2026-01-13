"""
SPI-SYRS Correlation Analysis Module
Complete workflow from data loading to result export
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# Import utility functions
from utils.data_io import load_processed_data, save_data_file
from utils.correlation_utils import (
    parse_growth_months_string,
    filter_by_growing_months,
    aggregate_annual_spi,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    determine_significance_level,
    validate_correlation_data,
    calculate_summary_statistics
)


def load_all_required_data(config, spi_scale='SPI_3', ):
    """
    Load all required datasets for analysis

    Args:
        spi_scale: SPI time scale ('SPI_1', 'SPI_3', 'SPI_6', 'SPI_12')

    Returns:
        tuple: (growth_months_df, spi_df, syrs_df, spi_scale)
    """
    print("=" * 80)
    print("SPI-SYRS Correlation Analysis")
    print("=" * 80)

    # Define file paths
    growth_months_path = f"./datasets/{config['country']}/processed/{config['run_tag']}/growth_seasons/state_growth_months.csv"
    spi_data_path = f"./datasets/{config['country']}/processed/{config['run_tag']}/spi_results.csv"
    syrs_data_path = f"./datasets/{config['country']}/processed/{config['run_tag']}/syrs_analysis/syrs_detailed.csv"

    # Load growth months data
    print(f"\n✓ Loading growth season data: {growth_months_path}")
    growth_months_df = load_processed_data(growth_months_path)
    print(f"  Loaded {len(growth_months_df)} states")

    # Load SPI data
    print(f"\n✓ Loading SPI data: {spi_data_path}")
    spi_df = load_processed_data(spi_data_path)

    # Convert time column
    if 'time' in spi_df.columns:
        spi_df['time'] = pd.to_datetime(spi_df['time'])
        spi_df['Year'] = spi_df['time'].dt.year
        spi_df['Month'] = spi_df['time'].dt.month

    # Standardize column names
    spi_df = spi_df.rename(columns={ 'region':'State'})
    print(f"  Loaded {len(spi_df)} records")
    print(f"  Time range: {spi_df['Year'].min()} - {spi_df['Year'].max()}")
    print(f"  States: {spi_df['State'].nunique()}")

    # Check if SPI scale exists
    if spi_scale not in spi_df.columns:
        available_scales = [col for col in spi_df.columns if col.startswith('SPI_')]
        raise ValueError(f"SPI scale '{spi_scale}' not found. Available: {available_scales}")

    # Remove missing SPI values
    spi_df = spi_df.dropna(subset=[spi_scale])
    print(f"  After removing missing values: {len(spi_df)} records")

    # Load SYRS data
    print(f"\n✓ Loading SYRS data: {syrs_data_path}")
    syrs_df = load_processed_data(syrs_data_path)
    syrs_df = syrs_df.rename(columns={"state": "State", "year": "Year"})
    print(f"  Loaded {len(syrs_df)} records")

    return growth_months_df, spi_df, syrs_df, spi_scale


def preprocess_and_filter_data(growth_months_df, spi_df, spi_scale):
    """
    Preprocess data and filter SPI by growing season

    Args:
        growth_months_df: Growth months DataFrame
        spi_df: SPI DataFrame
        spi_scale: SPI time scale column name

    Returns:
        DataFrame: Filtered SPI data for growing season
    """
    print("\n" + "=" * 80)
    print("Step 2: Filter SPI by Growing Season")
    print("=" * 80)

    filtered_data = []

    for _, state_row in growth_months_df.iterrows():
        state = state_row['State']
        growth_months = parse_growth_months_string(state_row['Growth_Months'])

        if not growth_months:
            print(f"⚠️  {state}: No growth months, skipping")
            continue

        # Filter SPI data for this state and growth months
        state_spi = filter_by_growing_months(
            spi_df, state, growth_months,
            state_col='State', month_col='Month'
        )

        if len(state_spi) == 0:
            print(f"⚠️  {state}: No SPI data found, skipping")
            continue

        filtered_data.append(state_spi)
        print(f"✓ {state: <20} Growth months: {growth_months} | SPI records: {len(state_spi)}")

    if not filtered_data:
        raise ValueError("No matching SPI data found after filtering")

    spi_growing = pd.concat(filtered_data, ignore_index=True)
    print(f"\nTotal SPI records: {len(spi_df)} → Growing season SPI: {len(spi_growing)}")

    return spi_growing


def calculate_annual_spi_statistics(spi_growing, spi_scale):
    """
    Calculate annual SPI statistics

    Args:
        spi_growing: Filtered SPI data
        spi_scale: SPI time scale column name

    Returns:
        DataFrame: Annual SPI statistics
    """
    print("\n" + "=" * 80)
    print("Step 3: Calculate Annual SPI Statistics")
    print("=" * 80)

    # Calculate annual statistics
    spi_annual = aggregate_annual_spi(
        spi_growing,
        state_col='State',
        year_col='Year',
        spi_col=spi_scale,
        precip_col='monthly_precip'
    )

    print(f"✓ Calculated annual statistics for {len(spi_annual)} state-year combinations")
    print("\nSample data:")
    print(spi_annual.head())

    return spi_annual


def merge_spi_and_syrs(spi_annual, syrs_df):
    """
    Merge SPI and SYRS datasets

    Args:
        spi_annual: Annual SPI statistics DataFrame
        syrs_df: SYRS DataFrame

    Returns:
        DataFrame: Merged dataset
    """
    print("\n" + "=" * 80)
    print("Step 4: Merge SPI and SYRS Data")
    print("=" * 80)

    # Merge datasets
    merged = spi_annual.merge(
        syrs_df[['State', 'Year', 'SYRS', 'yield_ton_per_ha', 'trend_yield']],
        on=['State', 'Year'],
        how='inner'
    )

    print(f"✓ Merged data: {len(merged)} records")
    print(f"✓ States: {merged['State'].nunique()}")

    if len(merged) == 0:
        print("\n❌ Error: SPI and SYRS data cannot be matched")
        print("SPI states:", sorted(spi_annual['State'].unique()))
        print("SYRS states:", sorted(syrs_df['State'].unique()))
        raise ValueError("No matching data between SPI and SYRS")

    # Show data count by state
    print("\nData count by state:")
    state_counts = merged.groupby('State').size().sort_values(ascending=False)
    for state, count in state_counts.items():
        print(f"  {state: <20} {count} years")

    return merged


def perform_correlation_analysis(merged_data, spi_scale):
    """
    Perform correlation analysis between SPI and SYRS

    Args:
        merged_data: Merged SPI-SYRS DataFrame
        spi_scale: SPI time scale column name

    Returns:
        DataFrame: Correlation results
    """
    print("\n" + "=" * 80)
    print(f"Step 5: Correlation Analysis (using {spi_scale})")
    print("=" * 80)

    correlation_results = []
    print("==========*********sorted(merged_data['State'].unique()):",sorted(merged_data['State'].unique()))
    for state in sorted(merged_data['State'].unique()):
        state_data = merged_data[merged_data['State'] == state]

        # Skip if insufficient data
        if len(state_data) < 3:
            print(f"⚠️  {state}: Insufficient data (n={len(state_data)})")
            continue

        spi_values = state_data['SPI_mean'].values
        syrs_values = state_data['SYRS'].values

        # Validate data
        if not validate_correlation_data(spi_values, syrs_values, min_samples=3):
            print(f"⚠️  {state}: Invalid data for correlation analysis")
            continue

        # Calculate Pearson correlation
        r_pearson, p_pearson = calculate_pearson_correlation(spi_values, syrs_values)

        # Calculate Spearman correlation
        r_spearman, p_spearman = calculate_spearman_correlation(spi_values, syrs_values)

        # Add to results
        correlation_results.append({
            'State': state,
            'N': len(state_data),
            'Pearson_r': r_pearson,
            'Pearson_p': p_pearson,
            'Pearson_sig': determine_significance_level(p_pearson),
            'Spearman_r': r_spearman,
            'Spearman_p': p_spearman,
            'Spearman_sig': determine_significance_level(p_spearman),
            'SPI_mean': np.mean(spi_values),
            'SPI_std': np.std(spi_values),
            'SYRS_mean': np.mean(syrs_values),
            'SYRS_std': np.std(syrs_values)
        })

    # Create results DataFrame
    corr_df = pd.DataFrame(correlation_results).sort_values('Pearson_r', ascending=False)

    # Display results
    print(f"\n{'State':<20} {'N':<6} {'Pearson_r':<12} {'P_value':<10} {'Sig':<6} "
          f"{'Spearman_r':<12} {'P_value':<10} {'Sig':<6}")
    print("-" * 90)

    for _, row in corr_df.iterrows():
        print(f"{row['State']:<20} {row['N']:<6} "
              f"{row['Pearson_r']:<12.4f} {row['Pearson_p']:<10.4f} {row['Pearson_sig']:<6} "
              f"{row['Spearman_r']:<12.4f} {row['Spearman_p']:<10.4f} {row['Spearman_sig']:<6}")

    return corr_df

def generate_visualization_outputs(corr_df, merged_data, spi_scale, output_dir):
    """
    Generate visualization outputs (Enhanced with graded color bars)
    """
    print("\n" + "=" * 80)
    print("Step 6: Generate Visualizations")
    print("=" * 80)

    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    generated_figures = []

    # =================================================================================
    # [NEW] 0. All States Scatter Grid (Plot SPI-SYRS relationship scatter plots by state)
    # =================================================================================
    print("Generating all states scatter grid...")
    fig_path = os.path.join(figures_dir, f'{spi_scale}_all_states_scatter_grid.png')

    # Get all states and sort
    states = sorted(merged_data['State'].unique())
    n_states = len(states)

    # Dynamically compute grid layout (4 per row)
    n_cols = 4
    n_rows = (n_states + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # Flatten for iteration

    for idx, state in enumerate(states):
        ax = axes[idx]
        state_data = merged_data[merged_data['State'] == state]

        # Get correlation info for this state (for title and color)
        state_corr = corr_df[corr_df['State'] == state].iloc[0]
        r_val = state_corr['Pearson_r']
        p_val = state_corr['Pearson_p']
        is_sig = state_corr['Pearson_sig'] != 'ns'

        # 1. Plot scatter points
        ax.scatter(state_data['SPI_mean'], state_data['SYRS'],
                   alpha=0.6, s=40, edgecolors='black', linewidth=0.5, color='skyblue')

        # 2. Plot regression trend line
        if len(state_data) > 1:
            # Linear fit
            z = np.polyfit(state_data['SPI_mean'], state_data['SYRS'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(state_data['SPI_mean'].min(), state_data['SPI_mean'].max(), 100)

            # Use green line for positive correlation, red for negative
            line_color = 'green' if r_val > 0 else 'red'
            # Significant lines: thicker and solid; non-significant: dashed
            line_style = '-' if is_sig else '--'
            line_width = 2 if is_sig else 1

            ax.plot(x_range, p(x_range), linestyle=line_style, color=line_color, linewidth=line_width, alpha=0.8)

        # 3. Set title and labels
        # If significant, make title bold and change color
        title_color = 'black'
        title_weight = 'normal'
        if is_sig:
            title_color = 'darkblue'
            title_weight = 'bold'

        ax.set_title(f"{state}\nr={r_val:.2f}, p={p_val:.3f}",
                     fontsize=11, color=title_color, fontweight=title_weight)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)

        # Show axis labels only on the leftmost column and bottom row to avoid clutter
        if idx % n_cols == 0:
            ax.set_ylabel('Yield Anomaly (SYRS)')
        if idx >= (n_rows - 1) * n_cols:  # Simple check for the last row
            ax.set_xlabel(f'{spi_scale}')

    # Hide extra empty subplots
    for i in range(n_states, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Generated all states scatter grid: {fig_path}")
    generated_figures.append(fig_path)

    # --- 1. Correlation bar chart (Enhanced Color Mapping) ---
    fig_path = os.path.join(figures_dir, f'{spi_scale}_correlation_bars.png')

    fig, axes = plt.subplots(2, 1, figsize=(14, 11)) # Slightly increase height to accommodate the colorbar

    # Set colormap and normalization
    # Use RdYlGn colormap: Red (negative) -> Yellow (zero) -> Green (positive)
    cmap = plt.get_cmap('RdYlGn')
    # TwoSlopeNorm ensures 0 value corresponds to the center of the colormap (light yellow/gray)
    # Range set to -1 to 1, which is the theoretical range of correlation coefficients
    norm = mcolors.TwoSlopeNorm(vmin=-0.8, vcenter=0., vmax=0.8) # Setting vmin/vmax to 0.8 makes extremes less dark; 1.0 also works

    # --- Subplot 1: Pearson correlation bars ---
    ax1 = axes[0]
    r_values_pearson = corr_df['Pearson_r'].values
    # Map colors according to r values
    bar_colors_pearson = cmap(norm(r_values_pearson))

    bars1 = ax1.bar(range(len(corr_df)), r_values_pearson, color=bar_colors_pearson,
                    edgecolor='grey', linewidth=0.5, alpha=0.9)

    # Add significance markers
    for i, (_, row) in enumerate(corr_df.iterrows()):
        if row['Pearson_sig'] != 'ns':
            # Decide the position of the star based on correlation direction (above or below the bar)
            va = 'bottom' if row['Pearson_r'] > 0 else 'top'
            # Add a small offset to prevent the star from sticking to the bar
            offset = 0.02 if row['Pearson_r'] > 0 else -0.02
            ax1.text(i, row['Pearson_r'] + offset, row['Pearson_sig'],
                     ha='center', va=va, fontsize=11, fontweight='bold', color='black')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xticks(range(len(corr_df)))
    ax1.set_xticklabels(corr_df['State'], rotation=45, ha='right')
    ax1.set_ylabel('Pearson Correlation (r)', fontsize=12)
    ax1.set_title(f'SPI-SYRS Pearson Correlation Magnitude ({spi_scale})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    # Set y-axis range to make the chart more symmetric
    ax1.set_ylim(-0.85, 0.85)

    # --- Subplot 2: Spearman correlation bars ---
    ax2 = axes[1]
    r_values_spearman = corr_df['Spearman_r'].values
    # Map colors according to r values
    bar_colors_spearman = cmap(norm(r_values_spearman))

    bars2 = ax2.bar(range(len(corr_df)), r_values_spearman, color=bar_colors_spearman,
                    edgecolor='grey', linewidth=0.5, alpha=0.9)

    # Add significance markers
    for i, (_, row) in enumerate(corr_df.iterrows()):
        if row['Spearman_sig'] != 'ns':
            va = 'bottom' if row['Spearman_r'] > 0 else 'top'
            offset = 0.02 if row['Spearman_r'] > 0 else -0.02
            ax2.text(i, row['Spearman_r'] + offset, row['Spearman_sig'],
                     ha='center', va=va, fontsize=11, fontweight='bold', color='black')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xticks(range(len(corr_df)))
    ax2.set_xticklabels(corr_df['State'], rotation=45, ha='right')
    ax2.set_ylabel('Spearman Correlation (ρ)', fontsize=12)
    ax2.set_title(f'SPI-SYRS Spearman Correlation Magnitude ({spi_scale})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(-0.85, 0.85)

    # --- Add Global Colorbar (Key step) ---
    # Create a ScalarMappable to generate the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # An empty array is sufficient

    # Add the colorbar to the right side of the figure, matching the height of the two subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Correlation Strength & Direction\n(Red=Negative, Green=Positive, Darker=Stronger)', fontsize=11)
    # Set ticks on the colorbar
    cbar.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
    cbar.set_ticklabels(['Strong Neg', 'Weak Neg', 'Neutral', 'Weak Pos', 'Strong Pos'])

    # Adjust layout to accommodate the colorbar
    plt.subplots_adjust(right=0.9, hspace=0.5)

    # Save image
    # Note: Do not use plt.tight_layout() because it may disrupt the manually set colorbar position
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Generated enhanced correlation bar chart: {fig_path}")
    generated_figures.append(fig_path)

    # --- 2. Scatter plots for significant states (keep unchanged) ---
    sig_states = corr_df[corr_df['Pearson_sig'] != 'ns'].head(6)

    if len(sig_states) > 0:
        fig_path = os.path.join(figures_dir, f'{spi_scale}_significant_states_scatter.png')

        n_plots = len(sig_states)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (_, state_row) in enumerate(sig_states.iterrows()):
            state = state_row['State']
            state_data = merged_data[merged_data['State'] == state]

            ax = axes[idx]
            # Set base scatter color according to correlation sign; add some transparency
            point_color = 'green' if state_row['Pearson_r'] > 0 else 'red'

            ax.scatter(state_data['SPI_mean'], state_data['SYRS'],
                       s=100, alpha=0.5, color=point_color, edgecolors='black', linewidths=0.5)

            # Add trend line
            z = np.polyfit(state_data['SPI_mean'], state_data['SYRS'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(state_data['SPI_mean'].min(), state_data['SPI_mean'].max(), 100)
            # Darken the trend line color
            line_color = 'darkgreen' if state_row['Pearson_r'] > 0 else 'darkred'
            ax.plot(x_line, p(x_line), linestyle='--', linewidth=2.5, color=line_color, alpha=0.8)

            ax.set_xlabel(f'{spi_scale} (Growing Season)', fontsize=11)
            ax.set_ylabel('SYRS (Yield Anomaly)', fontsize=11)
            ax.set_title(
                f'{state}\nr={state_row["Pearson_r"]:.3f}, p={state_row["Pearson_p"]:.4f} {state_row["Pearson_sig"]}',
                fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)

        for idx in range(len(sig_states), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Generated scatter plots: {fig_path}")
        generated_figures.append(fig_path)

    return generated_figures
# def generate_visualization_outputs(corr_df, merged_data, spi_scale, output_dir):
#     """
#     Generate visualization outputs
#
#     Args:
#         corr_df: Correlation results DataFrame
#         merged_data: Merged SPI-SYRS data
#         spi_scale: SPI time scale
#         output_dir: Output directory path
#
#     Returns:
#         list: Paths to generated figures
#     """
#     print("\n" + "=" * 80)
#     print("Step 6: Generate Visualizations")
#     print("=" * 80)
#
#     # Create output directory if it doesn't exist
#     figures_dir = os.path.join(output_dir, 'figures')
#     os.makedirs(figures_dir, exist_ok=True)
#
#     generated_figures = []
#
#     # 1. Correlation bar chart
#     fig_path = os.path.join(figures_dir, f'{spi_scale}_correlation_bars.png')
#
#     fig, axes = plt.subplots(2, 1, figsize=(14, 10))
#
#     # Set colormap and normalization
#     # Use RdYlGn colormap: Red (negative) -> Yellow (zero) -> Green (positive)
#     cmap = plt.get_cmap('RdYlGn')
#     # TwoSlopeNorm ensures 0 value corresponds to the center of the colormap (light yellow/gray)
#     # Range set to -1 to 1, which is the theoretical range of correlation coefficients
#     norm = mcolors.TwoSlopeNorm(vmin=-0.8, vcenter=0., vmax=0.8)  # Setting vmin/vmax to 0.8 makes extremes less dark; 1.0 also works
#
#     # Pearson correlation bars
#     ax1 = axes[0]
#     colors = ['green' if r > 0 else 'red' for r in corr_df['Pearson_r']]
#     bars = ax1.bar(range(len(corr_df)), corr_df['Pearson_r'], color=colors, alpha=0.7)
#
#     # Add significance markers
#     for i, (_, row) in enumerate(corr_df.iterrows()):
#         if row['Pearson_sig'] != 'ns':
#             ax1.text(i, row['Pearson_r'], row['Pearson_sig'],
#                      ha='center', va='bottom' if row['Pearson_r'] > 0 else 'top',
#                      fontsize=10, fontweight='bold')
#
#     ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
#     ax1.set_xticks(range(len(corr_df)))
#     ax1.set_xticklabels(corr_df['State'], rotation=45, ha='right')
#     ax1.set_ylabel('Pearson r')
#     ax1.set_title(f'SPI-SYRS Pearson Correlation ({spi_scale})')
#     ax1.grid(True, alpha=0.3, axis='y')
#
#     # Spearman correlation bars
#     ax2 = axes[1]
#     colors = ['green' if r > 0 else 'red' for r in corr_df['Spearman_r']]
#     ax2.bar(range(len(corr_df)), corr_df['Spearman_r'], color=colors, alpha=0.7)
#
#     # Add significance markers
#     for i, (_, row) in enumerate(corr_df.iterrows()):
#         if row['Spearman_sig'] != 'ns':
#             ax2.text(i, row['Spearman_r'], row['Spearman_sig'],
#                      ha='center', va='bottom' if row['Spearman_r'] > 0 else 'top',
#                      fontsize=10, fontweight='bold')
#
#     ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
#     ax2.set_xticks(range(len(corr_df)))
#     ax2.set_xticklabels(corr_df['State'], rotation=45, ha='right')
#     ax2.set_ylabel('Spearman ρ')
#     ax2.set_title(f'SPI-SYRS Spearman Correlation ({spi_scale})')
#     ax2.grid(True, alpha=0.3, axis='y')
#
#     plt.tight_layout()
#     plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"✓ Generated correlation bar chart: {fig_path}")
#     generated_figures.append(fig_path)
#
#     # 2. Scatter plots for significant states (top 6)
#     sig_states = corr_df[corr_df['Pearson_sig'] != 'ns'].head(6)
#
#     if len(sig_states) > 0:
#         fig_path = os.path.join(figures_dir, f'{spi_scale}_significant_states_scatter.png')
#
#         n_plots = len(sig_states)
#         n_cols = min(3, n_plots)
#         n_rows = (n_plots + n_cols - 1) // n_cols
#
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
#
#         # Flatten axes array for easy iteration
#         if n_plots == 1:
#             axes = np.array([axes])
#         axes = axes.flatten()
#
#         for idx, (_, state_row) in enumerate(sig_states.iterrows()):
#             state = state_row['State']
#             state_data = merged_data[merged_data['State'] == state]
#
#             ax = axes[idx]
#             ax.scatter(state_data['SPI_mean'], state_data['SYRS'],
#                        s=100, alpha=0.6, edgecolors='black', linewidths=1)
#
#             # Add trend line
#             z = np.polyfit(state_data['SPI_mean'], state_data['SYRS'], 1)
#             p = np.poly1d(z)
#             x_line = np.linspace(state_data['SPI_mean'].min(), state_data['SPI_mean'].max(), 100)
#             ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
#
#             # Add correlation info
#             ax.set_xlabel(f'{spi_scale} (Growing Season)')
#             ax.set_ylabel('SYRS')
#             ax.set_title(
#                 f'{state}\nr={state_row["Pearson_r"]:.3f}, p={state_row["Pearson_p"]:.4f} {state_row["Pearson_sig"]}')
#             ax.grid(True, alpha=0.3)
#             ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
#             ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
#
#         # Hide unused subplots
#         for idx in range(len(sig_states), len(axes)):
#             axes[idx].axis('off')
#
#         plt.tight_layout()
#         plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#         plt.close()
#
#         print(f"✓ Generated scatter plots: {fig_path}")
#         generated_figures.append(fig_path)
#
#     return generated_figures


def export_analysis_results(corr_df, merged_data, spi_scale, output_dir, summary_stats):
    """
    Export analysis results to files

    Args:
        corr_df: Correlation results DataFrame
        merged_data: Merged SPI-SYRS data
        spi_scale: SPI time scale
        output_dir: Output directory path
        summary_stats: Summary statistics dictionary

    Returns:
        dict: Paths to exported files
    """
    print("\n" + "=" * 80)
    print("Step 7: Export Results")
    print("=" * 80)

    # Create directories
    tables_dir = os.path.join(output_dir, 'tables')
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    exported_files = {}

    # 1. Save correlation results to CSV
    csv_path = os.path.join(tables_dir, f'{spi_scale}_correlation_results.csv')
    save_data_file(corr_df, csv_path)
    exported_files['correlation_csv'] = csv_path
    print(f"✓ Correlation results: {csv_path}")

    # 2. Save merged data to CSV
    merged_path = os.path.join(tables_dir, f'{spi_scale}_merged_data.csv')
    save_data_file(merged_data, merged_path)
    exported_files['merged_csv'] = merged_path
    print(f"✓ Merged data: {merged_path}")

    # 3. Save to Excel report
    excel_path = os.path.join(reports_dir, f'{spi_scale}_correlation_report.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Correlation results
        corr_df.to_excel(writer, sheet_name='Correlation', index=False)

        # Merged data
        merged_data.to_excel(writer, sheet_name='Merged_Data', index=False)

        # Significant results
        sig_pearson = corr_df[corr_df['Pearson_sig'] != 'ns']
        sig_pearson.to_excel(writer, sheet_name='Significant_Pearson', index=False)

        # Summary statistics
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    exported_files['excel_report'] = excel_path
    print(f"✓ Excel report: {excel_path}")

    # 4. Save summary statistics as text
    summary_path = os.path.join(tables_dir, f'{spi_scale}_summary.txt')

    with open(summary_path, 'w') as f:
        f.write(f"SPI-SYRS Correlation Analysis Summary ({spi_scale})\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total states analyzed: {summary_stats.get('total_states', 0)}\n")
        f.write(f"States with significant Pearson correlation: {summary_stats.get('pearson_significant', 0)} "
                f"({summary_stats.get('pearson_significant_pct', 0):.1f}%)\n")
        f.write(f"States with significant Spearman correlation: {summary_stats.get('spearman_significant', 0)} "
                f"({summary_stats.get('spearman_significant_pct', 0):.1f}%)\n")
        f.write(f"Positive correlations: {summary_stats.get('positive_correlation', 0)} "
                f"({summary_stats.get('positive_correlation_pct', 0):.1f}%)\n")
        f.write(f"Negative correlations: {summary_stats.get('negative_correlation', 0)} "
                f"({summary_stats.get('negative_correlation_pct', 0):.1f}%)\n")
        f.write(f"Mean Pearson correlation: {summary_stats.get('pearson_mean', 0):.3f}\n")
        f.write(
            f"Pearson correlation range: {summary_stats.get('pearson_min', 0):.3f} to {summary_stats.get('pearson_max', 0):.3f}\n")

    exported_files['summary_txt'] = summary_path
    print(f"✓ Summary statistics: {summary_path}")

    return exported_files


def run_complete_spi_syrs_analysis(config, spi_scale='SPI_3',):
    """
    Execute complete SPI-SYRS correlation analysis workflow

    Args:
        spi_scale: SPI time scale ('SPI_1', 'SPI_3', 'SPI_6', 'SPI_12')
        output_dir: Output directory path (default: ./datasets/malaysia/results/syrs_correlation)

    Returns:
        dict: Analysis results and metadata
    """
    # Set default output directory
    output_dir = f"./datasets/{config['country']}/processed/{config['run_tag']}/spi_syrs_correlation"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Load all required data
        growth_months_df, spi_df, syrs_df, spi_scale = load_all_required_data(config, spi_scale)

        # Step 2: Preprocess and filter data
        spi_growing = preprocess_and_filter_data(growth_months_df, spi_df, spi_scale)

        # Step 3: Calculate annual SPI statistics
        spi_annual = calculate_annual_spi_statistics(spi_growing, spi_scale)

        # Step 4: Merge SPI and SYRS data
        merged_data = merge_spi_and_syrs(spi_annual, syrs_df)

        # Step 5: Perform correlation analysis
        corr_df = perform_correlation_analysis(merged_data, spi_scale)

        # Calculate summary statistics
        summary_stats = calculate_summary_statistics(corr_df)

        # Display summary
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)

        print(f"\nTotal states: {summary_stats.get('total_states', 0)}")
        print(f"Pearson significant: {summary_stats.get('pearson_significant', 0)} states "
              f"({summary_stats.get('pearson_significant_pct', 0):.1f}%)")
        print(f"Spearman significant: {summary_stats.get('spearman_significant', 0)} states "
              f"({summary_stats.get('spearman_significant_pct', 0):.1f}%)")
        print(f"Pearson correlation range: {summary_stats.get('pearson_min', 0):.3f} to "
              f"{summary_stats.get('pearson_max', 0):.3f}")
        print(f"Mean Pearson correlation: {summary_stats.get('pearson_mean', 0):.3f}")
        print(f"Positive correlations: {summary_stats.get('positive_correlation', 0)} states "
              f"({summary_stats.get('positive_correlation_pct', 0):.1f}%)")
        print(f"Negative correlations: {summary_stats.get('negative_correlation', 0)} states "
              f"({summary_stats.get('negative_correlation_pct', 0):.1f}%)")

        # Step 6: Generate visualizations
        figures = generate_visualization_outputs(corr_df, merged_data, spi_scale, output_dir)

        # Step 7: Export results
        exported_files = export_analysis_results(corr_df, merged_data, spi_scale, output_dir, summary_stats)

        # Return results
        results = {
            'success': True,
            'spi_scale': spi_scale,
            'state_count': summary_stats.get('total_states', 0),
            'significant_states_count': summary_stats.get('pearson_significant', 0),
            'correlation_results': corr_df,
            'merged_data': merged_data,
            'summary_statistics': summary_stats,
            'output_directory': output_dir,
            'generated_figures': figures,
            'exported_files': exported_files,
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print("\n" + "=" * 80)
        print("✓ Analysis Complete!")
        print("=" * 80)

        return results

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'error': str(e),
            'spi_scale': spi_scale,
            'output_directory': output_dir
        }