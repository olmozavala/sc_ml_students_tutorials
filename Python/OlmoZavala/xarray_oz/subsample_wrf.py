import numpy as np
import pandas as pd
import seaborn as sns
from img_viz.eoa_viz import EOAImageVisualizer
import matplotlib.pyplot as plt
from os.path import join
from io_xarray import data_summary

import xarray as xr

if __name__ == "__main__":
    viz_obj = EOAImageVisualizer()

    file_name = "/home/olmozavala/wrfout_d01_2020-08-27_00.nc"
    ds = xr.load_dataset(file_name)

    x = 1
    # # viz_obj.plot_3d_data_xarray_map(ds, var_names=['air'], timesteps=[0], title='2D example')
    # # -------- Modifying values of a region with a subset of the dimensions (there are more options)
    # # http://xarray.pydata.org/en/stable/indexing.html#more-advanced-indexing
    # lon = ds.coords["lon"]
    # lat = ds.coords["lat"]
    # # ds['air'].loc[dict(lon=lon[(lon > 220) & (lon < 260)], lat=lat[(lat > 20) & (lat < 60)])] = 290
    # viz_obj.plot_3d_data_xarray_map(ds, var_names=['air'], timesteps=[0], title='Modified data 2D example')
    #
    # # -------- Changeging an index
    # air = ds['air']
    # air_small = air[:20, : 20]