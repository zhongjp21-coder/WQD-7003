"""
Statistical utilities for trend analysis and data processing
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple, Optional, Dict, List
import warnings


def linear_trend_fitting(x: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, np.ndarray]:
    """
    Fit linear trend to data

    Parameters:
    -----------
    x : np.ndarray
        Independent variable (e.g., years)
    y : np.ndarray
        Dependent variable (e.g., yield)

    Returns:
    --------
    model : LinearRegression
        Fitted linear model
    trend : np.ndarray
        Trend values
    """
    x_reshaped = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_reshaped, y)
    trend = model.predict(x_reshaped)

    return model, trend


def polynomial_trend_fitting(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Tuple[LinearRegression, np.ndarray]:
    """
    Fit polynomial trend to data

    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    degree : int
        Polynomial degree

    Returns:
    --------
    model : LinearRegression
        Fitted polynomial model
    trend : np.ndarray
        Trend values
    """
    x_reshaped = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x_reshaped)

    model = LinearRegression()
    model.fit(x_poly, y)
    trend = model.predict(x_poly)

    return model, trend


def moving_average_trend(y: np.ndarray, window: int = 5, center: bool = True) -> np.ndarray:
    """
    Calculate moving average trend

    Parameters:
    -----------
    y : np.ndarray
        Time series data
    window : int
        Moving window size
    center : bool
        Whether to center the window

    Returns:
    --------
    np.ndarray: Moving average trend
    """
    series = pd.Series(y)
    trend = series.rolling(window=window, center=center, min_periods=1).mean()
    trend = trend.fillna(method='bfill').fillna(method='ffill').values

    return trend


def calculate_residuals(y: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """
    Calculate residuals from trend

    Parameters:
    -----------
    y : np.ndarray
        Original data
    trend : np.ndarray
        Trend values

    Returns:
    --------
    np.ndarray: Residuals
    """
    residuals = y - trend
    return residuals


def standardize_values(values: np.ndarray, ddof: int = 1) -> np.ndarray:
    """
    Standardize values (z-score normalization)

    Parameters:
    -----------
    values : np.ndarray
        Values to standardize
    ddof : int
        Delta degrees of freedom for std calculation

    Returns:
    --------
    np.ndarray: Standardized values
    """
    std = np.nanstd(values, ddof=ddof)

    if std < 1e-6:  # Avoid division by near-zero
        return np.zeros_like(values)

    standardized = (values - np.nanmean(values)) / std
    return standardized


def detect_outliers_iqr(values: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers using IQR method

    Parameters:
    -----------
    values : np.ndarray
        Data values
    threshold : float
        IQR multiplier threshold

    Returns:
    --------
    np.ndarray: Boolean mask of outliers
    """
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1

    if iqr < 1e-6:
        return np.zeros_like(values, dtype=bool)

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outliers = (values < lower_bound) | (values > upper_bound)
    return outliers


def calculate_goodness_of_fit(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate goodness of fit metrics

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    Dict: Fit metrics
    """
    # Remove NaN values for calculation
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return {
            'r_squared': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'explained_variance': np.nan
        }

    # R-squared
    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)

    if ss_tot > 1e-10:
        r_squared = 1 - (ss_res / ss_tot)
    else:
        r_squared = 0

    # MAE and RMSE
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))

    # Explained variance
    explained_variance = 1 - np.var(y_true_clean - y_pred_clean) / np.var(y_true_clean)

    return {
        'r_squared': max(0, min(1, r_squared)),
        'mae': mae,
        'rmse': rmse,
        'explained_variance': explained_variance
    }


def calculate_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate slope and intercept for linear relationship

    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable

    Returns:
    --------
    slope : float
    intercept : float
    """
    if len(x) < 2:
        return 0.0, np.nanmean(y) if len(y) > 0 else 0.0

    slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0
    intercept = y[0] - slope * x[0]

    return slope, intercept


def mann_kendall_test(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Perform Mann-Kendall trend test

    Parameters:
    -----------
    x : np.ndarray
        Time values
    y : np.ndarray
        Data values

    Returns:
    --------
    Dict: Test results
    """
    try:
        # Perform Mann-Kendall test
        tau, p_value = stats.kendalltau(x, y, nan_policy='omit')

        # Determine trend direction
        if p_value < 0.05:
            if tau > 0:
                trend = "increasing"
            elif tau < 0:
                trend = "decreasing"
            else:
                trend = "no trend"
        else:
            trend = "no significant trend"

        return {
            'tau': tau,
            'p_value': p_value,
            'trend': trend,
            'significant': p_value < 0.05
        }

    except Exception as e:
        warnings.warn(f"Mann-Kendall test failed: {e}")
        return {
            'tau': np.nan,
            'p_value': np.nan,
            'trend': "test failed",
            'significant': False
        }