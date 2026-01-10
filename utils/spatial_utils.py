"""
Spatial utilities for geographic data processing
"""

import numpy as np
import xarray as xr
from typing import Tuple
import geopandas as gpd
from shapely.geometry import box
from scipy.spatial import KDTree


def filter_geographic_region(ds: xr.Dataset,
                             lat_range: Tuple[float, float],
                             lon_range: Tuple[float, float]) -> xr.Dataset:
    """
    Filter dataset to specific geographic region

    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset
    lat_range : Tuple[float, float]
        Latitude range (min_lat, max_lat)
    lon_range : Tuple[float, float]
        Longitude range (min_lon, max_lon)

    Returns:
    --------
    ds_filtered : xr.Dataset
        Filtered dataset
    """
    # Auto-detect dimension names
    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'

    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    print(f"Original {lat_name} range: [{lat_values.min():.2f}, {lat_values.max():.2f}]")
    print(f"Original {lon_name} range: [{lon_values.min():.2f}, {lon_values.max():.2f}]")

    # Check coordinate order
    lat_ascending = lat_values[0] < lat_values[-1]
    lon_ascending = lon_values[0] < lon_values[-1]

    print(f"{lat_name} direction: {'Ascending' if lat_ascending else 'Descending'}")
    print(f"{lon_name} direction: {'Ascending' if lon_ascending else 'Descending'}")

    # Apply correct slicing based on order
    if lat_ascending:
        lat_slice = slice(lat_range[0], lat_range[1])
    else:
        lat_slice = slice(lat_range[1], lat_range[0])

    if lon_ascending:
        lon_slice = slice(lon_range[0], lon_range[1])
    else:
        lon_slice = slice(lon_range[1], lon_range[0])

    # Perform filtering
    ds_filtered = ds.sel({
        lat_name: lat_slice,
        lon_name: lon_slice
    })

    n_lat = ds_filtered[lat_name].size
    n_lon = ds_filtered[lon_name].size
    n_grids = n_lat * n_lon

    print(f"✓ Region filtering completed")
    print(f"  Latitude range: {ds_filtered[lat_name].min().values:.2f} to {ds_filtered[lat_name].max().values:.2f}")
    print(f"  Longitude range: {ds_filtered[lon_name].min().values:.2f} to {ds_filtered[lon_name].max().values:.2f}")
    print(f"  Grid points: {n_grids}")

    return ds_filtered


def create_grid_polygons(df, lat_col='lat', lon_col='lon', grid_size=0.5):
    """
    Create polygon geometries for grid points

    Parameters:
    ----------
    df : DataFrame
        Input data with lat/lon columns
    lat_col : str
        Latitude column name
    lon_col : str
        Longitude column name
    grid_size : float
        Grid cell size in degrees

    Returns:
    --------
    GeoDataFrame with polygon geometries
    """
    import geopandas as gpd

    df = df.copy()
    df['geometry'] = df.apply(
        lambda r: box(
            r[lon_col] - grid_size / 2,
            r[lat_col] - grid_size / 2,
            r[lon_col] + grid_size / 2,
            r[lat_col] + grid_size / 2
        ),
        axis=1
    )

    return gpd.GeoDataFrame(df, crs='EPSG:4326')