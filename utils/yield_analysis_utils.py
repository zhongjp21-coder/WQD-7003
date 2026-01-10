"""
Yield Analysis Utilities
Basic tools for SYRS calculation and yield trend analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')


def calculate_yield_per_ha(production, planted_area, min_area=0.001):
    """
    Safely calculate yield per hectare

    Args:
        production: Production value
        planted_area: Planted area in hectares
        min_area: Minimum valid area threshold

    Returns:
        float: Yield in ton/ha, or NaN if invalid
    """
    if pd.isna(production) or pd.isna(planted_area):
        return np.nan

    if planted_area < min_area:
        return np.nan

    yield_val = production / planted_area

    # Validate yield range (0-15 ton/ha)
    if yield_val > 15 or yield_val < 0:
        return np.nan

    return yield_val


def validate_yield_data(yield_series, min_years=3):
    """
    Validate yield data for SYRS calculation

    Args:
        yield_series: Series of yield values
        min_years: Minimum number of valid years

    Returns:
        bool: True if data is valid
    """
    if len(yield_series) < min_years:
        return False

    # Check for variability
    if np.std(yield_series) < 0.01:
        return False

    # Check for too many missing values
    valid_count = np.sum(~np.isnan(yield_series))
    if valid_count < min_years:
        return False

    return True


def linear_detrending(years, yields):
    """
    Perform linear detrending

    Args:
        years: Array of years
        yields: Array of yield values

    Returns:
        tuple: (trend, residuals, model, metrics)
    """
    years_reshaped = years.reshape(-1, 1)

    model = LinearRegression()
    model.fit(years_reshaped, yields)
    trend = model.predict(years_reshaped)
    residuals = yields - trend

    # Calculate metrics
    metrics = _calculate_model_metrics(yields, residuals, trend, n_params=2)
    metrics['slope'] = model.coef_[0]
    metrics['intercept'] = model.intercept_

    return trend, residuals, model, metrics


def polynomial_detrending(years, yields, degree=2):
    """
    Perform polynomial detrending

    Args:
        years: Array of years
        yields: Array of yield values
        degree: Polynomial degree

    Returns:
        tuple: (trend, residuals, model, metrics)
    """
    years_reshaped = years.reshape(-1, 1)

    poly = PolynomialFeatures(degree=degree)
    years_poly = poly.fit_transform(years_reshaped)

    model = LinearRegression()
    model.fit(years_poly, yields)
    trend = model.predict(years_poly)
    residuals = yields - trend

    # Calculate metrics
    n_params = degree + 1
    metrics = _calculate_model_metrics(yields, residuals, trend, n_params=n_params)
    metrics['degree'] = degree

    return trend, residuals, model, metrics


def moving_average_detrending(yields, window=None):
    """
    Perform moving average detrending

    Args:
        yields: Array of yield values
        window: Moving window size

    Returns:
        tuple: (trend, residuals, metrics)
    """
    if window is None:
        window = min(5, len(yields))

    series = pd.Series(yields)
    trend = series.rolling(window=window, center=True, min_periods=1).mean().values
    residuals = yields - trend

    # Calculate metrics
    n_params = window  # Approximate degrees of freedom
    metrics = _calculate_model_metrics(yields, residuals, trend, n_params=n_params)
    metrics['window'] = window

    return trend, residuals, metrics


def _calculate_model_metrics(yields, residuals, trend, n_params):
    """
    Calculate model evaluation metrics

    Args:
        yields: Actual yield values
        residuals: Residual values
        trend: Trend values
        n_params: Number of model parameters

    Returns:
        dict: Model metrics
    """
    n = len(yields)

    # Calculate R²
    ss_tot = np.sum((yields - np.mean(yields)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calculate MSE and AIC
    mse = ss_res / n
    aic = n * np.log(mse + 1e-10) + 2 * n_params

    # Residual statistics
    residual_std = np.std(residuals, ddof=1)

    return {
        'r2': r2,
        'mse': mse,
        'aic': aic,
        'residual_std': residual_std,
        'n_samples': n,
        'n_params': n_params
    }


def standardize_residuals(residuals):
    """
    Standardize residuals to create SYRS

    Args:
        residuals: Residual values

    Returns:
        array: Standardized yield residual series (SYRS)
    """
    residual_std = np.std(residuals, ddof=1)

    if residual_std > 1e-6:
        syrs = residuals / residual_std
    else:
        syrs = np.zeros_like(residuals)

    return syrs


def interpret_syrs_value(syrs_value):
    """
    Interpret SYRS value for climate significance

    Args:
        syrs_value: SYRS value

    Returns:
        tuple: (category, description)
    """
    if pd.isna(syrs_value):
        return "Unknown", "Missing data"
    elif syrs_value > 2.0:
        return "Extremely Favorable", "Excellent climate conditions"
    elif syrs_value > 1.5:
        return "Very Favorable", "Very favorable conditions"
    elif syrs_value > 0.5:
        return "Favorable", "Favorable conditions"
    elif syrs_value > -0.5:
        return "Normal", "Normal conditions"
    elif syrs_value > -1.5:
        return "Unfavorable", "Unfavorable conditions"
    elif syrs_value > -2.0:
        return "Very Unfavorable", "Very unfavorable conditions"
    else:
        return "Extremely Unfavorable", "Extremely unfavorable conditions"


def compare_detrending_methods(metrics_dict):
    """
    Compare different detrending methods based on metrics

    Args:
        metrics_dict: Dictionary of method metrics

    Returns:
        str: Best method name
    """
    # Filter out failed methods
    valid_methods = {k: v for k, v in metrics_dict.items() if v is not None}

    if not valid_methods:
        return None

    # Select method with lowest AIC (corrected for sample size)
    best_method = min(valid_methods.items(),
                      key=lambda x: x[1]['aic'])[0]

    return best_method