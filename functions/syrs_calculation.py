"""
SYRS (Standardized Yield Residual Series) calculation module for crop yield analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.preprocessing import PolynomialFeatures  # ✅ 新增导入

from utils.statistical_utils import (
    linear_trend_fitting,
    polynomial_trend_fitting,
    moving_average_trend,
    calculate_residuals,
    standardize_values,
    calculate_goodness_of_fit,
    calculate_slope_intercept
)


def safe_yield_calculation(row: pd.Series,
                           production_col: str = 'production',
                           area_col: str = 'planted_area',
                           min_area: float = 0.01) -> Optional[float]:
    """
    Safe yield calculation with error handling for division by zero and outliers

    Parameters:
    -----------
    row : pd.Series
        Data row containing production and area
    production_col : str
        Production column name
    area_col : str
        Planted area column name
    min_area : float
        Minimum valid area (in hectares)

    Returns:
    --------
    float or None: Calculated yield or None if invalid
    """
    planted_area = row.get(area_col)
    production = row.get(production_col)

    # Check for missing values
    if pd.isna(planted_area) or pd.isna(production):
        return None

    # Check for valid area
    if planted_area < min_area:
        return None  # No meaningful planting area

    # Check for valid production
    if production < 0:
        return None  # Negative production indicates data error

    # Calculate yield
    yield_value = production / planted_area

    return yield_value


def get_default_crop_parameters(crop_type: str = 'rice') -> Dict:
    """
    Get default parameters for specific crop type

    Parameters:
    -----------
    crop_type : str
        Type of crop

    Returns:
    --------
    Dict: Crop parameters
    """
    # Default parameters for different crop types
    crop_defaults = {
        'rice': {
            'name': 'Rice',
            'yield_range': {'min': 1.0, 'max': 8.0},
            'min_area': 0.01,
            'min_years_required': 3,
            'syrs_thresholds': {
                'extremely_favorable': 2.0,
                'very_favorable': 1.5,
                'favorable': 0.5,
                'normal': -0.5,
                'unfavorable': -1.5,
                'very_unfavorable': -2.0
            }
        }
    }

    # Return parameters for specified crop, default to rice if not found
    return crop_defaults.get(crop_type.lower(), crop_defaults['rice'])


def validate_crop_data(df: pd.DataFrame,
                       crop_type: str = 'rice',
                       crop_params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Validate crop yield data based on crop type parameters

    Parameters:
    -----------
    df : pd.DataFrame
        Crop yield data
    crop_type : str
        Type of crop (for parameter lookup)
    crop_params : Dict, optional
        Crop-specific parameters

    Returns:
    --------
    pd.DataFrame: Validated data with quality flags
    """
    df_validated = df.copy()

    # Get crop parameters if not provided
    if crop_params is None:
        crop_params = get_default_crop_parameters(crop_type)

    # Calculate yield if not already present
    if 'yield_ton_per_ha' not in df_validated.columns:
        if 'production' in df_validated.columns and 'planted_area' in df_validated.columns:
            df_validated['yield_ton_per_ha'] = df_validated.apply(
                lambda row: safe_yield_calculation(row), axis=1
            )

    # Add quality flags
    df_validated['data_quality'] = 'Good'

    # Flag missing yield
    missing_yield = df_validated['yield_ton_per_ha'].isna()
    df_validated.loc[missing_yield, 'data_quality'] = 'Missing Yield'

    # Flag extreme values
    if 'yield_ton_per_ha' in df_validated.columns:
        valid_yield = df_validated['yield_ton_per_ha'].dropna()
        if len(valid_yield) > 0:
            yield_range = crop_params.get('yield_range', {'min': 1.0, 'max': 8.0})
            too_low = df_validated['yield_ton_per_ha'] < yield_range['min']
            too_high = df_validated['yield_ton_per_ha'] > yield_range['max']
            extreme_mask = too_low | too_high

            df_validated.loc[extreme_mask & ~missing_yield, 'data_quality'] = 'Extreme Yield'

    return df_validated


def clean_crop_yield_data(df: pd.DataFrame,
                          min_valid_years: int = 3) -> pd.DataFrame:
    """
    Clean crop yield data by removing states with insufficient data

    Parameters:
    -----------
    df : pd.DataFrame
        Crop yield data
    min_valid_years : int
        Minimum number of valid years required

    Returns:
    --------
    pd.DataFrame: Cleaned data
    """
    df_clean = df.copy()

    # Calculate yield per row if needed
    if 'yield_ton_per_ha' not in df_clean.columns:
        if 'production' in df_clean.columns and 'planted_area' in df_clean.columns:
            df_clean['yield_ton_per_ha'] = df_clean.apply(
                lambda row: safe_yield_calculation(row), axis=1
            )

    # Remove rows with missing yield
    df_clean = df_clean.dropna(subset=['yield_ton_per_ha']).copy()

    # Filter states with enough data
    if 'state' in df_clean.columns and 'year' in df_clean.columns:
        state_year_counts = df_clean.groupby('state')['year'].nunique()
        valid_states = state_year_counts[state_year_counts >= min_valid_years].index
        df_clean = df_clean[df_clean['state'].isin(valid_states)].copy()

    return df_clean


def calculate_syrs_robust(state_data: pd.DataFrame,
                          method: str = 'linear',
                          crop_params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
    """
    Robust SYRS calculation with comprehensive error handling

    Parameters:
    -----------
    state_data : pd.DataFrame
        State-level crop yield data
    method : str
        Detrending method: 'linear', 'polynomial', or 'moving_average'
    crop_params : Dict, optional
        Crop-specific parameters for validation

    Returns:
    --------
    pd.DataFrame or None: SYRS results or None if calculation fails
    """
    # ========== Data Validation ==========

    # Check data quantity
    if len(state_data) < 3:
        warnings.warn(f"Insufficient data: Only {len(state_data)} years, minimum 3 required")
        return None

    # Check yield data validity
    yields = state_data['yield_ton_per_ha'].values

    if np.all(np.isnan(yields)):
        warnings.warn("All yield data are NaN")
        return None

    if len(yields[~np.isnan(yields)]) < 3:
        warnings.warn(f"Only {len(yields[~np.isnan(yields)])} valid yield points")
        return None

    # Check yield variability
    valid_yields = yields[~np.isnan(yields)]
    yield_std = np.std(valid_yields, ddof=1)
    yield_mean = np.mean(valid_yields)

    if yield_std < 1e-6:
        warnings.warn(f"Yield shows no variation (std={yield_std:.6f})")
        return None

    if yield_mean < 0.1:
        warnings.warn(f"Average yield abnormally low ({yield_mean:.3f} ton/ha)")
        return None

    # ========== Prepare Data ==========

    years = state_data['year'].values
    yields = state_data['yield_ton_per_ha'].values

    # Remove NaN values for trend fitting
    valid_mask = ~np.isnan(yields)
    years_clean = years[valid_mask]
    yields_clean = yields[valid_mask]

    if len(yields_clean) < 3:
        warnings.warn(f"Only {len(yields_clean)} valid data points after cleaning")
        return None

    # ========== Detrending ==========

    try:
        if method == 'linear':
            # Linear trend
            model, trend_clean = linear_trend_fitting(years_clean, yields_clean)
            slope, intercept = calculate_slope_intercept(years_clean, yields_clean)

            # Predict trend for all years (including potential NaN positions)
            trend = slope * years + intercept

        elif method == 'polynomial':
            # Polynomial trend (degree 2)
            model, trend_clean = polynomial_trend_fitting(years_clean, yields_clean, degree=2)

            # Predict for all years
            poly = PolynomialFeatures(degree=2)
            years_poly_all = poly.fit_transform(years.reshape(-1, 1))
            trend = model.predict(years_poly_all)
            slope, intercept = calculate_slope_intercept(years_clean, yields_clean)

        elif method == 'moving_average':
            # Moving average trend
            window = min(5, len(yields_clean))
            trend = moving_average_trend(yields, window=window, center=True)
            slope, intercept = calculate_slope_intercept(years_clean, yields_clean)

        else:
            raise ValueError(f"Unknown detrending method: {method}")

    except Exception as e:
        warnings.warn(f"Trend fitting failed: {str(e)}")
        return None

    # ========== Calculate Residuals ==========

    residuals = calculate_residuals(yields, trend)

    # Check residual validity
    valid_residuals = residuals[~np.isnan(residuals)]

    if len(valid_residuals) < 2:
        warnings.warn("Too few valid residuals")
        return None

    # ========== Standardize to SYRS ==========

    residual_std = np.std(valid_residuals, ddof=1)

    if residual_std < 1e-6:
        # Residual std close to zero, data perfectly fits trend
        warnings.warn(f"Residual standard deviation near zero ({residual_std:.6f}), SYRS set to 0")
        syrs = np.zeros_like(residuals)
        residual_std = 0.001  # Small non-zero value to avoid division by zero
    else:
        syrs = standardize_values(residuals)

    # ========== Calculate Fit Metrics ==========

    fit_metrics = calculate_goodness_of_fit(yields, trend)
    r_squared = fit_metrics['r_squared']

    # ========== Assemble Results ==========

    result = state_data.copy()
    result['trend_yield'] = trend
    result['residual'] = residuals
    result['SYRS'] = syrs
    result['trend_slope'] = slope
    result['trend_intercept'] = intercept
    result['r_squared'] = r_squared
    result['residual_std'] = residual_std
    result['detrending_method'] = method

    # Add data quality assessment
    if len(valid_yields) == len(yields):
        result['data_quality'] = 'Good'
    else:
        result['data_quality'] = f'Partial ({len(valid_yields)}/{len(yields)} valid)'

    return result


def interpret_syrs_values(syrs_value: float,
                          thresholds: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Interpret SYRS values into climate categories

    Parameters:
    -----------
    syrs_value : float
        SYRS value to interpret
    thresholds : Dict, optional
        Custom threshold values

    Returns:
    --------
    category : str
        Climate category
    description : str
        Climate description
    """
    if thresholds is None:
        # Default thresholds
        thresholds = {
            'extremely_favorable': 2.0,
            'very_favorable': 1.5,
            'favorable': 0.5,
            'normal': -0.5,
            'unfavorable': -1.5,
            'very_unfavorable': -2.0
        }

    if pd.isna(syrs_value):
        return "Unknown", "Missing data"

    if syrs_value > thresholds['extremely_favorable']:
        return "Extremely Favorable", "Exceptional climate conditions"
    elif syrs_value > thresholds['very_favorable']:
        return "Very Favorable", "Very good climate conditions"
    elif syrs_value > thresholds['favorable']:
        return "Favorable", "Good climate conditions"
    elif syrs_value > thresholds['normal']:
        return "Normal", "Average climate conditions"
    elif syrs_value > thresholds['unfavorable']:
        return "Unfavorable", "Poor climate conditions"
    elif syrs_value > thresholds['very_unfavorable']:
        return "Very Unfavorable", "Very poor climate conditions"
    else:
        return "Extremely Unfavorable", "Extreme climate stress"


def process_all_states_syrs(df: pd.DataFrame,
                            method: str = 'linear',
                            crop_type: str = 'rice',
                            crop_params: Optional[Dict] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Process SYRS calculation for all states

    Parameters:
    -----------
    df : pd.DataFrame
        Crop yield data for all states
    method : str
        Detrending method
    crop_type : str
        Crop type for parameter reference
    crop_params : Dict, optional
        Crop-specific parameters

    Returns:
    --------
    results_df : pd.DataFrame
        SYRS results for successful states
    skipped_states : List[Dict]
        Information about skipped states
    """
    results = []
    skipped_states = []

    # Get default crop parameters if not provided
    if crop_params is None:
        crop_params = get_default_crop_parameters(crop_type)

    # Filter out national aggregate data if present
    states = df[df['state'] != 'Malaysia']['state'].unique()

    print(f"\nProcessing SYRS calculation")
    print(f"Method: {method}")
    print(f"Crop type: {crop_type}")
    print("=" * 80)

    for state in states:
        state_data = df[df['state'] == state].sort_values('year').copy()

        print(f"\nProcessing: {state}")
        print("-" * 40)

        # Pre-check data quality
        if len(state_data) == 0:
            print(f"  ❌ Skipped: No data")
            skipped_states.append({'state': state, 'reason': 'No data'})
            continue

        if len(state_data) < 3:
            print(f"  ❌ Skipped: Insufficient data ({len(state_data)} years < 3)")
            skipped_states.append({'state': state, 'reason': f'Insufficient data ({len(state_data)} years)'})
            continue

        # Check yield validity
        valid_yields = state_data['yield_ton_per_ha'].dropna()

        if len(valid_yields) == 0:
            print(f"  ❌ Skipped: All yield data invalid")
            skipped_states.append({'state': state, 'reason': 'All yields invalid'})
            continue

        if valid_yields.std() < 0.01:
            print(f"  ⚠️  Warning: Yield shows little variation (std={valid_yields.std():.4f})")
            print(f"  Yield range: {valid_yields.min():.3f} - {valid_yields.max():.3f} ton/ha")

        # Attempt SYRS calculation
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                syrs_result = calculate_syrs_robust(state_data, method=method, crop_params=crop_params)

                # Display any warnings
                if w:
                    for warning in w:
                        print(f"  ⚠️  {warning.message}")

            if syrs_result is None:
                print(f"  ❌ Skipped: SYRS calculation failed")
                skipped_states.append({'state': state, 'reason': 'SYRS calculation failed'})
                continue

            # Successfully calculated
            syrs_result['method'] = method
            syrs_result['crop_type'] = crop_type
            results.append(syrs_result)

            # Display summary
            print(f"  ✅ Success")
            print(f"    Years: {syrs_result['year'].min():.0f}-{syrs_result['year'].max():.0f}")
            print(f"    Average yield: {syrs_result['yield_ton_per_ha'].mean():.3f} ton/ha")
            print(f"    Trend slope: {syrs_result['trend_slope'].iloc[0]:+.4f} ton/ha/year")
            print(f"    R²: {syrs_result['r_squared'].iloc[0]:.3f}")
            print(f"    Residual std: {syrs_result['residual_std'].iloc[0]:.3f}")
            print(f"    SYRS range: [{syrs_result['SYRS'].min():.2f}, {syrs_result['SYRS'].max():.2f}]")

        except Exception as e:
            print(f"  ❌ Skipped: Error occurred - {str(e)}")
            skipped_states.append({'state': state, 'reason': f'Error: {str(e)}'})
            continue

    # Summary
    print("\n" + "=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Total states: {len(states)}")
    print(f"Successfully processed: {len(results)} states")
    print(f"Skipped: {len(skipped_states)} states")

    if skipped_states:
        print("\nSkipped states and reasons:")
        for item in skipped_states:
            print(f"  - {item['state']:20} {item['reason']}")

    if results:
        return pd.concat(results, ignore_index=True), skipped_states
    else:
        print("\n⚠️  Warning: No states successfully calculated SYRS")
        return pd.DataFrame(), skipped_states


def analyze_syrs_results(syrs_df: pd.DataFrame,
                         thresholds: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze SYRS results and add climate interpretation

    Parameters:
    -----------
    syrs_df : pd.DataFrame
        SYRS calculation results
    thresholds : Dict, optional
        Climate threshold values

    Returns:
    --------
    df_analyzed : pd.DataFrame
        Enhanced SYRS results with analysis
    stats_df : pd.DataFrame
        State-level statistics
    """
    if syrs_df.empty:
        return syrs_df, pd.DataFrame()

    df_analyzed = syrs_df.copy()

    # Add climate interpretation
    df_analyzed[['climate_category', 'climate_description']] = df_analyzed.apply(
        lambda row: pd.Series(interpret_syrs_values(row['SYRS'], thresholds)),
        axis=1
    )

    # Calculate additional statistics
    if 'state' in df_analyzed.columns:
        state_stats = []
        for state in df_analyzed['state'].unique():
            state_data = df_analyzed[df_analyzed['state'] == state]

            stats = {
                'state': state,
                'mean_syrs': state_data['SYRS'].mean(),
                'std_syrs': state_data['SYRS'].std(),
                'min_syrs': state_data['SYRS'].min(),
                'max_syrs': state_data['SYRS'].max(),
                'favorable_years': (state_data['SYRS'] > 0.5).sum(),
                'unfavorable_years': (state_data['SYRS'] < -0.5).sum(),
                'extreme_years': (abs(state_data['SYRS']) > 2.0).sum(),
                'mean_yield': state_data['yield_ton_per_ha'].mean(),
                'yield_trend': state_data['trend_slope'].iloc[0] if len(state_data) > 0 else 0.0
            }
            state_stats.append(stats)

        # Create separate stats DataFrame
        stats_df = pd.DataFrame(state_stats)

        return df_analyzed, stats_df

    return df_analyzed, pd.DataFrame()


def run_syrs_analysis_pipeline(config,
                               crop_type: str = 'rice',
                               method: str = 'linear',
                               crop_params: Optional[Dict] = None,
                               thresholds: Optional[Dict] = None) -> Dict:
    """
    Complete SYRS analysis pipeline

    Parameters:
    -----------
    input_file : str
        Path to crop yield data file
    output_dir : str
        Output directory for results
    crop_type : str
        Type of crop being analyzed
    method : str
        Detrending method for SYRS calculation
    crop_params : Dict, optional
        Crop-specific parameters
    thresholds : Dict, optional
        Climate threshold values

    Returns:
    --------
    Dict: Analysis results and metadata
    """
    import json
    from pathlib import Path
    from utils.data_io import load_processed_data, save_data_file

    print("=" * 80)
    print("SYRS ANALYSIS PIPELINE")
    print("=" * 80)

    # Create output directory
    output_path = Path(f"./datasets/{config['country']}/processed/{config['run_tag']}/syrs")
    output_path.mkdir(parents=True, exist_ok=True)

    # Get default parameters if not provided
    if crop_params is None:
        crop_params = get_default_crop_parameters(crop_type)

    if thresholds is None:
        thresholds = crop_params.get('syrs_thresholds', {})

    # Load data
    print("\nLoading data...")
    df = load_processed_data(f"./datasets/{config['country']}/rawdata/crops_state.xlsx", format='xlsx')

    # Preprocess date if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

    print(f"Total records: {len(df)}")
    print(f"States: {df['state'].nunique()}")

    # Validate and clean data
    print("\nValidating and cleaning data...")
    df_validated = validate_crop_data(df, crop_type, crop_params)
    df_clean = clean_crop_yield_data(df_validated, min_valid_years=3)

    print(f"Original records: {len(df)}")
    print(f"Valid records: {len(df_clean)}")
    print(f"Removed records: {len(df) - len(df_clean)}")

    # Calculate SYRS
    print("\nCalculating SYRS...")
    syrs_results, skipped_states = process_all_states_syrs(
        df_clean,
        method=method,
        crop_type=crop_type,
        crop_params=crop_params
    )

    if syrs_results.empty:
        print("\n❌ No SYRS results generated")
        return {'success': False, 'error': 'No results generated'}

    # Analyze results
    print("\nAnalyzing SYRS results...")
    syrs_analyzed, syrs_stats = analyze_syrs_results(syrs_results, thresholds)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # 1. Detailed SYRS results
    save_data_file(syrs_analyzed, output_path / "syrs_detailed.csv")
    print("✓ syrs_detailed.csv - Detailed SYRS results")

    # 2. SYRS statistics
    if not syrs_stats.empty:
        save_data_file(syrs_stats, output_path / "syrs_statistics.csv")
        print("✓ syrs_statistics.csv - State-level SYRS statistics")

    # 3. Cleaned yield data
    save_data_file(df_clean, output_path / "cleaned_yield_data.csv")
    print("✓ cleaned_yield_data.csv - Cleaned yield data")

    # 4. Skipped states log
    if skipped_states:
        skipped_df = pd.DataFrame(skipped_states)
        save_data_file(skipped_df, output_path / "skipped_states.csv")
        print("✓ skipped_states.csv - Log of skipped states")

    # 5. Metadata
    metadata = {
        'crop_type': crop_type,
        'detrending_method': method,
        'total_states_processed': syrs_analyzed['state'].nunique(),
        'total_records': len(syrs_analyzed),
        'years_range': {
            'start': int(syrs_analyzed['year'].min()),
            'end': int(syrs_analyzed['year'].max())
        },
        'skipped_states_count': len(skipped_states),
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'crop_parameters': crop_params,
        'thresholds': thresholds
    }

    save_data_file(metadata, output_path / "analysis_metadata.json", format='json')
    print("✓ analysis_metadata.json - Analysis metadata")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(f"\nSummary:")
    print(f"  States analyzed: {syrs_analyzed['state'].nunique()}")
    print(f"  Years per state: {syrs_analyzed.groupby('state')['year'].nunique().mean():.1f}")
    print(f"  Average SYRS: {syrs_analyzed['SYRS'].mean():.3f}")
    print(f"  SYRS range: [{syrs_analyzed['SYRS'].min():.2f}, {syrs_analyzed['SYRS'].max():.2f}]")

    return {
        'success': True,
        'syrs_results': syrs_analyzed,
        'syrs_stats': syrs_stats,
        'skipped_states': skipped_states,
        'metadata': metadata,
    }