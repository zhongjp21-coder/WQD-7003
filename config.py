"""
Configuration settings for precipitation analysis pipeline
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
UTILS_DIR = BASE_DIR / "utils"
FUNCTIONS_DIR = BASE_DIR / "functions"

# Default parameters
DEFAULT_PARAMS = {
    "malaysia": {
        "lat_range": (0.5, 7.5),
        "lon_range": (99.5, 120.5),
        "grid_size": 0.5,
        "timescales": [1, 3, 6, 12],
        "distribution": "gamma",
        "min_valid_months": 90,
        "max_missing_ratio": 0.3
    }
}

# Processing options
PROCESSING_OPTIONS = {
    "handle_negatives": "zero",  # 'zero', 'nan', or 'interpolate'
    "handle_high_values": "nan",  # 'nan', 'cap', or 'keep'
    "apply_interpolation": True,
    "completeness_threshold": 0.8  # Minimum data completeness for monthly aggregation
}


def get_country_paths(country: str, run_tag: str = None):
    """Get file paths for a specific country and run tag"""
    country_dir = DATASETS_DIR / country.lower()

    if run_tag:
        processed_dir = country_dir / "processed" / run_tag
        results_dir = country_dir / "results" / run_tag
        # Create directories if they don't exist
        processed_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        processed_dir = country_dir / "processed"
        results_dir = country_dir / "results"

    return {
        "rawdata": country_dir / "rawdata",
        "processed": processed_dir,
        "results": results_dir,
        "states_geojson": f"{country_dir}/rawdata/gadm41_{country}_1.json"
    }
