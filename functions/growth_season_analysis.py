"""
Growth season analysis module for agricultural pattern analysis
"""

import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil, os;
from utils.data_io import load_processed_data, save_data_file
import warnings


def aggregate_state_weighted(state_df: pd.DataFrame,
                             months_to_use: List[str]) -> List[str]:
    """
    Area-weighted aggregation: select the most dominant growth stage for each month

    Parameters:
    -----------
    state_df : DataFrame
        DataFrame containing state-level growth data
    months_to_use : List[str]
        List of month column names to process

    Returns:
    --------
    List[str]: Unified growth pattern for the state
    """
    total_area = state_df['Area_ha'].sum()
    unified_pattern = []

    for month_col in months_to_use:
        # Count area for each growth stage in this month
        stage_weights = {}

        for _, row in state_df.iterrows():
            stage = row.get(month_col, '')

            # Handle missing values and NaN
            if pd.isna(stage):
                stage = ''

            stage = str(stage).strip()

            # Skip fallow periods (blank)
            if stage == '' or stage == 'nan':
                continue

            area = row['Area_ha']
            stage_weights[stage] = stage_weights.get(stage, 0) + area

        # If all clusters are fallow in this month
        if not stage_weights:
            unified_pattern.append('')
            continue

        # Select the most dominant stage (largest area)
        dominant_stage = max(stage_weights, key=stage_weights.get)
        dominant_area = stage_weights[dominant_stage]
        dominant_percentage = (dominant_area / total_area) * 100

        # If dominant stage percentage is too low (<30%), mark as mixed
        if dominant_percentage < 30:
            unified_pattern.append('Mixed')
        else:
            unified_pattern.append(dominant_stage)

    return unified_pattern


def count_complete_seasons(pattern: List[str]) -> List[Dict]:
    """
    Count complete growing seasons

    Rules: Complete sequence: T → V/V1/V2 → R → M

    Parameters:
    -----------
    pattern : List[str]
        Monthly growth pattern

    Returns:
    --------
    List[Dict]: Complete seasons found
    """
    seasons = []
    i = 0

    while i < len(pattern):
        stage = pattern[i]

        # Skip fallow and mixed
        if stage in ['', 'Mixed']:
            i += 1
            continue

        # Find planting indicator T
        if stage == 'T':
            season_start = i

            # Look for complete cycle in next 5-6 months
            for j in range(i + 1, min(i + 7, len(pattern))):
                if pattern[j] == 'M':
                    # Check intermediate stages
                    middle = pattern[i + 1:j]

                    # Must have V (including V1/V2/V3) and R
                    has_V = any(s in ['V', 'V1', 'V2', 'V3'] or
                                (isinstance(s, str) and s.startswith('V'))
                                for s in middle)
                    has_R = 'R' in middle

                    if has_V and has_R:
                        seasons.append({
                            'start': season_start,
                            'end': j,
                            'duration': j - season_start + 1,
                            'pattern': ' → '.join([str(s) for s in pattern[season_start:j + 1] if s])
                        })
                        i = j
                        break
            else:
                i += 1
        else:
            i += 1

    return seasons


def identify_planting_months(pattern: List[str]) -> List[int]:
    """
    Identify planting months (positions where T appears)

    Parameters:
    -----------
    pattern : List[str]
        Monthly growth pattern

    Returns:
    --------
    List[int]: Planting month indices (1-12)
    """
    months = []
    for i, stage in enumerate(pattern):
        if stage == 'T':
            months.append(i + 1)  # Convert to month number (1-12)
    return months


def calculate_active_coverage(pattern: List[str]) -> float:
    """
    Calculate active growing coverage percentage

    Parameters:
    -----------
    pattern : List[str]
        Monthly growth pattern

    Returns:
    --------
    float: Active coverage percentage
    """
    active = sum(1 for s in pattern if s not in ['', 'Mixed'])
    return (active / len(pattern) * 100) if pattern else 0


def extract_growth_months(result_df: pd.DataFrame, r_flag=0) -> pd.DataFrame:
    """
    Extract growth months for each state (months containing T, V series, or R)

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results DataFrame

    Returns:
    --------
    DataFrame: Growth months summary for each state
    """
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    growth_months_data = []

    for _, row in result_df.iterrows():
        state = row['State']

        # Use set to collect all growth months (T, V series, R)
        growth_months = set()

        for month_idx, month_name in enumerate(month_names, 1):
            col_name = f'{month_name}_2019'
            stage = row.get(col_name, '')

            # Check if it's a growth stage (exclude M and fallow)
            if r_flag == 1:
                if stage == "R":
                    growth_months.add(month_idx)
            else:
                if stage in ['T', 'V', 'V1', 'V2', 'V3', "R"]:
                    growth_months.add(month_idx)

        # Sort months
        growth_months_sorted = sorted(list(growth_months))

        # Convert to month names
        growth_month_names = [month_names[m - 1] for m in growth_months_sorted]

        growth_months_data.append({
            'State': state,
            'Growth_Months': ','.join(map(str, growth_months_sorted)) if growth_months_sorted else '-',
            'Growth_Months_Names': ','.join(growth_month_names) if growth_month_names else '-',
            'Growth_Months_Count': len(growth_months_sorted),
            'Growth_Start_Month': min(growth_months_sorted) if growth_months_sorted else None,
            'Growth_End_Month': max(growth_months_sorted) if growth_months_sorted else None
        })

    return pd.DataFrame(growth_months_data)


def analyze_season_distribution(result_df: pd.DataFrame) -> Dict:
    """
    Analyze season distribution across states

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results

    Returns:
    --------
    Dict: Season distribution statistics
    """
    season_dist = result_df['Seasons_2019'].value_counts().sort_index()

    distribution = {}
    for seasons, count in season_dist.items():
        states_with_n_seasons = result_df[result_df['Seasons_2019'] == seasons]['State'].tolist()
        distribution[seasons] = {
            'count': count,
            'percentage': (count / len(result_df) * 100),
            'states': states_with_n_seasons
        }

    return {
        'distribution': distribution,
        'mean': result_df['Seasons_2019'].mean(),
        'median': result_df['Seasons_2019'].median(),
        'min': result_df['Seasons_2019'].min(),
        'max': result_df['Seasons_2019'].max(),
        'total_states': len(result_df)
    }


def analyze_area_distribution(result_df: pd.DataFrame) -> Dict:
    """
    Analyze area distribution across states

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results

    Returns:
    --------
    Dict: Area distribution statistics
    """
    return {
        'total_area': result_df['Total_Area_ha'].sum(),
        'mean_area': result_df['Total_Area_ha'].mean(),
        'median_area': result_df['Total_Area_ha'].median(),
        'max_area': {
            'state': result_df.loc[result_df['Total_Area_ha'].idxmax(), 'State'],
            'area': result_df['Total_Area_ha'].max()
        },
        'min_area': {
            'state': result_df.loc[result_df['Total_Area_ha'].idxmin(), 'State'],
            'area': result_df['Total_Area_ha'].min()
        }
    }


def analyze_coverage_distribution(result_df: pd.DataFrame) -> Dict:
    """
    Analyze active coverage distribution across states

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results

    Returns:
    --------
    Dict: Coverage distribution statistics
    """
    return {
        'mean_coverage': result_df['Active_Coverage_%'].mean(),
        'max_coverage': {
            'state': result_df.loc[result_df['Active_Coverage_%'].idxmax(), 'State'],
            'coverage': result_df['Active_Coverage_%'].max()
        },
        'min_coverage': {
            'state': result_df.loc[result_df['Active_Coverage_%'].idxmin(), 'State'],
            'coverage': result_df['Active_Coverage_%'].min()
        }
    }


def analyze_planting_timing(result_df: pd.DataFrame) -> Dict:
    """
    Analyze planting timing distribution

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results

    Returns:
    --------
    Dict: Planting timing statistics
    """
    all_planting_months = []
    for _, row in result_df.iterrows():
        if row['Planting_Months'] != 'None':
            months = [int(m.strip()) for m in row['Planting_Months'].split(',')]
            all_planting_months.extend(months)

    month_freq = Counter(all_planting_months) if all_planting_months else {}

    return {
        'total_planting_events': len(all_planting_months),
        'month_frequency': month_freq,
        'most_common_month': max(month_freq, key=month_freq.get) if month_freq else None,
        'states_with_planting': sum(1 for m in result_df['Planting_Months'] if m != 'None')
    }


def create_unified_state_dataset(df: pd.DataFrame,
                                 analysis_year: int = 2019) -> pd.DataFrame:
    """
    Create unified state-level dataset from cluster data

    Parameters:
    -----------
    df : DataFrame
        Raw cluster-level growth data
    analysis_year : int
        Year for analysis (default: 2019)

    Returns:
    --------
    DataFrame: Unified state-level results
    """
    # Define month columns
    months_2019 = [f'{month}_2019' for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    months_2020 = [f'{month}_2020' for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    all_months = months_2019 + months_2020

    state_unified_data = []

    for state_name in df['State'].unique():
        state_df = df[df['State'] == state_name]

        num_clusters = len(state_df)
        total_area = state_df['Area_ha'].sum()

        # Aggregate patterns
        pattern_2019 = aggregate_state_weighted(state_df, months_2019)
        pattern_24 = aggregate_state_weighted(state_df, all_months)

        # Count seasons
        seasons_24 = count_complete_seasons(pattern_24)
        seasons_2019_complete = [s for s in seasons_24 if s['start'] < 12 and s['end'] < 12]

        # Identify planting months
        planting_months = identify_planting_months(pattern_2019)

        # Calculate coverage
        coverage = calculate_active_coverage(pattern_2019)

        # Assemble result
        result = {
            'No': len(state_unified_data) + 1,
            'State': state_name,
            'Cluster_Code': f'{state_name[:3]}-Unified',
            'Total_Area_ha': int(total_area),
            'Num_Original_Clusters': num_clusters,
            'Seasons_2019': len(seasons_2019_complete),
            'Planting_Months': ', '.join(map(str, planting_months)) if planting_months else 'None',
            'Active_Coverage_%': round(coverage, 1),
            'Pattern_2019_Text': ' '.join([s if s else '_' for s in pattern_2019])
        }

        # Add season details
        for i, season in enumerate(seasons_2019_complete, 1):
            result[f'Season{i}_Start'] = season['start'] + 1  # Convert to month number
            result[f'Season{i}_End'] = season['end'] + 1
            result[f'Season{i}_Duration'] = season['duration']
            result[f'Season{i}_Stages'] = season['pattern']

        # Add monthly detailed stages
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, month_name in enumerate(month_names):
            result[f'{month_name}_2019'] = pattern_2019[i] if pattern_2019[i] else ''

        # Add original cluster list
        cluster_list = []
        for _, cluster_row in state_df.iterrows():
            cluster_list.append(f"{cluster_row['Cluster_Code']}({cluster_row['Area_ha']:.0f}ha)")
        result['Original_Clusters'] = '; '.join(cluster_list)

        state_unified_data.append(result)

    # Create DataFrame and sort by area
    result_df = pd.DataFrame(state_unified_data)
    result_df = result_df.sort_values('Total_Area_ha', ascending=False)

    return result_df


def create_rainfall_matching_format(result_df: pd.DataFrame,
                                    state_coords: Dict) -> pd.DataFrame:
    """
    Create rainfall matching format data (state × month format)

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results
    state_coords : Dict
        Dictionary of state coordinates

    Returns:
    --------
    DataFrame: Rainfall matching data
    """
    matching_data = []

    for _, row in result_df.iterrows():
        state = row['State']
        coords = state_coords.get(state, {'lat': None, 'lon': None})

        for month_idx, month_name in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1):
            col_name = f'{month_name}_2019'
            growth_stage = row.get(col_name, '')

            matching_data.append({
                'State': state,
                'Year': 2019,
                'Month': month_idx,
                'Month_Name': month_name,
                'Growth_Stage': growth_stage if growth_stage else 'Fallow',
                'Area_ha': row['Total_Area_ha'],
                'Latitude': coords.get('lat'),
                'Longitude': coords.get('lon'),
                'Rainfall_mm': None  # To be filled
            })

    return pd.DataFrame(matching_data)


def create_growth_months_dataset(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create growth months summary dataset

    Parameters:
    -----------
    result_df : DataFrame
        State-aggregated results

    Returns:
    --------
    DataFrame: Growth months summary
    """
    growth_months_df = extract_growth_months(result_df)

    # Merge with main results
    result_with_months = result_df.merge(growth_months_df, on='State', how='left')

    return result_with_months


def run_growth_analysis_pipeline(config: dict,
                                 state_coords: Optional[Dict] = None,
                                 analysis_year: int = 2019) -> Dict:
    """
    Run complete growth season analysis pipeline

    Parameters:
    -----------
    input_file : str
        Path to input growth season data file
    output_dir : str
        Output directory for results
    state_coords : Dict, optional
        Dictionary of state coordinates
    analysis_year : int
        Analysis year (default: 2019)

    Returns:
    --------
    Dict: Analysis results and statistics
    """
    print("=" * 100)
    print("GROWTH SEASON ANALYSIS PIPELINE")
    print("=" * 100)

    country = config.get('country')
    if country == "china":
        folder = f'./datasets/{country}/processed/{config["run_tag"]}/growth_seasons'
        os.makedirs(folder, exist_ok=True);
        shutil.copy(f'./datasets/{country}/rawdata/state_growth_months.csv', folder)
        return

    # Load data
    print("\nLoading data...")
    df = load_processed_data(f'./datasets/{country}/rawdata/growing_season_multi_clus.xlsx', format='xlsx')

    print(f"Total clusters: {len(df)}")
    print(f"Number of states: {df['State'].nunique()}")
    print(f"Total area: {df['Area_ha'].sum():,.0f} ha")
    print(f"\nClusters per state:")
    print(df['State'].value_counts())

    # Create output directory
    output_path = Path(f'./datasets/{country}/processed/{config["run_tag"]}/growth_seasons')
    output_path.mkdir(parents=True, exist_ok=True)

    # Process data
    print("\n" + "=" * 100)
    print("Processing state aggregation...")
    print("=" * 100)

    result_df = create_unified_state_dataset(df, analysis_year)

    # Create growth months dataset
    growth_months_df_r = extract_growth_months(result_df, r_flag=1)  # r_flag=1 : stage ==r
    growth_months_df = extract_growth_months(result_df, )  # r_flag=0 : stage in ['T', 'V', 'V1', 'V2', 'V3', 'R']
    result_with_months = create_growth_months_dataset(result_df)

    # # Create rainfall matching data
    # if state_coords is None:
    #     # Default state coordinates for Malaysia
    #     state_coords = {
    #         'Selangor': {'lat': 3.07, 'lon': 101.52},
    #         'Perlis': {'lat': 6.44, 'lon': 100.20},
    #         'Kedah': {'lat': 6.12, 'lon': 100.37},
    #         'Kelantan': {'lat': 5.95, 'lon': 102.04},
    #         'Terengganu': {'lat': 5.31, 'lon': 103.13},
    #         'Pahang': {'lat': 3.81, 'lon': 103.33},
    #         'Perak': {'lat': 4.59, 'lon': 101.08},
    #         'Penang': {'lat': 5.41, 'lon': 100.33},
    #         'Johor': {'lat': 1.48, 'lon': 103.74},
    #         'Malacca': {'lat': 2.19, 'lon': 102.25},
    #         'Negeri Sembilan': {'lat': 2.73, 'lon': 101.94},
    #         'Langkawi': {'lat': 6.35, 'lon': 99.85}
    #     }
    #
    # rainfall_matching_df = create_rainfall_matching_format(result_df, state_coords)

    # Perform analyses
    season_stats = analyze_season_distribution(result_df)
    area_stats = analyze_area_distribution(result_df)
    coverage_stats = analyze_coverage_distribution(result_df)
    planting_stats = analyze_planting_timing(result_df)

    # Save results
    print("\n" + "=" * 100)
    print("Saving results...")
    print("=" * 100)

    # 1. Full unified data
    save_data_file(result_df, output_path / "state_unified_weighted.csv")
    print("✓ state_unified_weighted.csv - Complete data")

    # 2. Growth months summary
    save_data_file(growth_months_df, output_path / "state_growth_months.csv")
    print("✓ state_growth_months.csv - Growth months summary(stage T&V)")

    # 2. Growth months summary
    save_data_file(growth_months_df_r, output_path / "state_growth_months_r.csv")
    print("✓ state_growth_months_r.csv - Growth months summary(stage R)")

    # # 3. Rainfall matching format
    # save_data_file(rainfall_matching_df, output_path / "rainfall_growth_matching.csv")
    # print("✓ rainfall_growth_matching.csv - Rainfall matching format")

    # 4. Combined data with growth months
    save_data_file(result_with_months, output_path / "state_unified_with_growth_months.csv")
    print("✓ state_unified_with_growth_months.csv - Combined data")

    # 5. Simplified format (like Table S1)
    month_cols = [f'{m}_2019' for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    simplified_cols = ['No', 'State', 'Cluster_Code', 'Total_Area_ha'] + month_cols
    simplified_df = result_df[simplified_cols].copy()
    simplified_df = simplified_df.rename(columns={'Total_Area_ha': 'Area_ha'})
    save_data_file(simplified_df, output_path / "state_unified_table_format.csv")
    print("✓ state_unified_table_format.csv - Simplified format")

    # 6. Excel report
    with pd.ExcelWriter(output_path / "growth_analysis_summary.xlsx", engine='openpyxl') as writer:
        # Summary sheet
        summary_cols = ['No', 'State', 'Cluster_Code', 'Total_Area_ha',
                        'Num_Original_Clusters', 'Seasons_2019',
                        'Planting_Months', 'Active_Coverage_%']
        result_df[summary_cols].to_excel(writer, sheet_name='Summary', index=False)

        # Full data sheet
        result_df.to_excel(writer, sheet_name='Full_Data', index=False)

        # Monthly pattern sheet
        monthly_matrix = result_df[['State'] + month_cols].copy()
        monthly_matrix.to_excel(writer, sheet_name='Monthly_Pattern', index=False)

        # Growth months sheet
        growth_months_df.to_excel(writer, sheet_name='Growth_Months', index=False)

        # Statistics sheet
        stats_data = {
            'Metric': [
                'Total States',
                'Total Area (ha)',
                'Mean Area (ha)',
                'Mean Seasons per Year',
                'Median Seasons',
                'Max Seasons',
                'Min Seasons',
                'Mean Active Coverage (%)'
            ],
            'Value': [
                len(result_df),
                result_df['Total_Area_ha'].sum(),
                result_df['Total_Area_ha'].mean(),
                result_df['Seasons_2019'].mean(),
                result_df['Seasons_2019'].median(),
                result_df['Seasons_2019'].max(),
                result_df['Seasons_2019'].min(),
                result_df['Active_Coverage_%'].mean()
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)

        # Season distribution sheet
        season_dist_df = pd.DataFrame({
            'Seasons': season_stats['distribution'].keys(),
            'Number of States': [v['count'] for v in season_stats['distribution'].values()],
            'Percentage (%)': [v['percentage'] for v in season_stats['distribution'].values()],
            'States': [', '.join(v['states']) for v in season_stats['distribution'].values()]
        })
        season_dist_df.to_excel(writer, sheet_name='Season_Distribution', index=False)

    print("✓ growth_analysis_summary.xlsx - Excel analysis report")

    # Save metadata
    metadata = {
        'input_file': "./datasets/malaysia/rawdata/growing_season_multi_clus.xlsx",
        'analysis_year': analysis_year,
        'total_states': len(result_df),
        'total_area': result_df['Total_Area_ha'].sum(),
        'season_statistics': season_stats,
        'area_statistics': area_stats,
        'coverage_statistics': coverage_stats,
        'planting_statistics': planting_stats
    }

    save_data_file(metadata, output_path / "analysis_metadata.json", format='json')
    print("✓ analysis_metadata.json - Analysis metadata")

    # Print summary
    print("\n" + "=" * 100)
    print("ANALYSIS SUMMARY")
    print("=" * 100)

    print(f"\nSeason Distribution:")
    for seasons, data in season_stats['distribution'].items():
        print(f"  {seasons} season(s): {data['count']} states ({data['percentage']:.1f}%)")

    print(f"\nArea Statistics:")
    print(f"  Total area: {area_stats['total_area']:,.0f} ha")
    print(f"  Mean area per state: {area_stats['mean_area']:,.0f} ha")
    print(f"  Largest state: {area_stats['max_area']['state']} ({area_stats['max_area']['area']:,.0f} ha)")
    print(f"  Smallest state: {area_stats['min_area']['state']} ({area_stats['min_area']['area']:,.0f} ha)")

    print(f"\nCoverage Statistics:")
    print(f"  Mean active coverage: {coverage_stats['mean_coverage']:.1f}%")
    print(
        f"  Highest coverage: {coverage_stats['max_coverage']['state']} ({coverage_stats['max_coverage']['coverage']:.1f}%)")
    print(
        f"  Lowest coverage: {coverage_stats['min_coverage']['state']} ({coverage_stats['min_coverage']['coverage']:.1f}%)")

    print("\n" + "=" * 100)
    print("PROCESSING COMPLETE!")
    print("=" * 100)
    print(f"\nGenerated 6 files in: {output_path}")

    return {
        'result_df': result_df,
        'growth_months_df': growth_months_df,
        # 'rainfall_matching_df': rainfall_matching_df,
        'season_stats': season_stats,
        'area_stats': area_stats,
        'coverage_stats': coverage_stats,
        'planting_stats': planting_stats,
        'output_dir': str(output_path)
    }