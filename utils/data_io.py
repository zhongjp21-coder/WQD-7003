"""
Data I/O utilities for reading and writing precipitation data
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Union


def read_precipitation_data(data_path: str) -> Tuple[xr.Dataset, float, Tuple[float, float]]:
    """
    Read raw precipitation data from NetCDF files

    Parameters:
    -----------
    data_path : str
        Path to data file or directory

    Returns:
    --------
    ds : xr.Dataset
        Raw dataset
    missing_val : float
        Missing value indicator
    valid_range : Tuple[float, float]
        Valid data range
    """
    path = Path(data_path)

    # Support multiple formats
    if path.suffix == '.nc':
        ds = xr.open_dataset(data_path)
    elif path.is_dir():
        # If directory, read all nc files
        ds = xr.open_mfdataset(str(path / "*.nc"), combine='by_coords')


    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")

    # Extract metadata
    precip_var = 'precip' if 'precip' in ds else 'precipitation'
    missing_val = ds[precip_var].attrs.get('missing_value', -999.0)

    # Get valid range from attributes or use defaults
    valid_min = ds[precip_var].attrs.get('valid_min', 0.0)
    valid_max = ds[precip_var].attrs.get('valid_max', 2000.0)  # mm/day
    valid_range = (valid_min, valid_max)

    print(f"✓ Data loaded successfully")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Missing value indicator: {missing_val}")
    print(f"  Valid range: {valid_range}")

    return ds, missing_val, valid_range


def save_processed_data(data, filepath: str, format: str = 'csv'):
    """
    Save processed data in specified format

    Parameters:
    -----------
    data : DataFrame or Dataset
        Data to save
    filepath : str
        Output file path
    format : str
        Output format ('csv', 'netcdf', 'parquet')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        if hasattr(data, 'to_csv'):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
    elif format == 'netcdf':
        if hasattr(data, 'to_netcdf'):
            data.to_netcdf(filepath)
        else:
            raise ValueError("Data must be xarray Dataset for NetCDF format")
    elif format == 'parquet':
        if hasattr(data, 'to_parquet'):
            data.to_parquet(filepath, index=False)
        else:
            pd.DataFrame(data).to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Data saved to: {filepath}")


def load_processed_data(filepath: str, format: str = 'csv'):
    """
    Load processed data

    Parameters:
    -----------
    filepath : str
        Input file path
    format : str
        Input format ('csv', 'netcdf', 'parquet')
    """
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'netcdf':
        return xr.open_dataset(filepath)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'xlsx':
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_data_file(data: Union[pd.DataFrame, xr.Dataset],
                   file_path: str,
                   format: Optional[str] = None,
                   **kwargs):
    """
    Save data to file based on format or file extension

    Parameters:
    -----------
    data : DataFrame or Dataset
        Data to save
    file_path : str
        Output file path
    format : str, optional
        Output format ('csv', 'excel', 'parquet', 'netcdf', 'json')
        If None, determined from file extension
    **kwargs : dict
        Additional arguments passed to save function
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format is None:
        format = path.suffix.lower().lstrip('.')

    # Save based on format
    if format in ['csv', 'txt']:
        if hasattr(data, 'to_csv'):
            data.to_csv(file_path, **kwargs)
        else:
            pd.DataFrame(data).to_csv(file_path, **kwargs)

    elif format in ['xlsx', 'xls', 'excel']:
        if hasattr(data, 'to_excel'):
            data.to_excel(file_path, **kwargs)
        else:
            pd.DataFrame(data).to_excel(file_path, **kwargs)

    elif format == 'parquet':
        if hasattr(data, 'to_parquet'):
            data.to_parquet(file_path, **kwargs)
        else:
            pd.DataFrame(data).to_parquet(file_path, **kwargs)

    elif format == 'netcdf':
        if hasattr(data, 'to_netcdf'):
            data.to_netcdf(file_path, **kwargs)
        else:
            raise ValueError("Data must be xarray Dataset for NetCDF format")

    elif format == 'json':
        if hasattr(data, 'to_json'):
            data.to_json(file_path, **kwargs)
        else:
            pd.DataFrame(data).to_json(file_path, **kwargs)

    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Data saved to: {file_path}")