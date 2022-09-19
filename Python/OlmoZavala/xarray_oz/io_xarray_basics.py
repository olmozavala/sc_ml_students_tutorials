import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
import xarray as xr

def plot_1d_data_np(X, Ys,  title='', labels=[], file_name_prefix='', wide_ratio=1):
    """
    Plots multiple lines in a single plot.
    """
    plt.figure(figsize=[8*wide_ratio, 8])
    for i, y in enumerate(Ys):
        style = F"-.{_COLORS[i%len(_COLORS)]}"
        if len(labels) > 0:
            assert len(labels) == len(Ys)
            plt.plot(X, y, style, label=labels[i])
        else:
            plt.plot(X, y, style)

    if len(labels) > 0:
        plt.legend()

    plt.grid(True)
    plt.title(title, fontsize=10)
    file_name = F'{file_name_prefix}'
    # pylab.savefig(join(self._output_folder, F'{file_name}.png'), bbox_inches='tight')
    plt.show()

_COLORS = ['y', 'r', 'c', 'b', 'g', 'w', 'k', 'y', 'r', 'c', 'b', 'g', 'w', 'k']

data_path = join('..', 'data')
file_name = join(data_path, 'GFSTodayReduced.nc')
ds = xr.load_dataset(file_name)

## ------------- Data summary ---------------------
print("------------- Data summary ---------------------")
print(ds.head())
df = ds.to_dataframe()
print(df.describe())

## # In this example we have two variables (temp_surf, tmax) with two dimensions each (time:731, location:3)
X = range(len(ds["time"]))
# http://xarray.pydata.org/en/stable/indexing.html
# --- access by index (single var, all times)----
Y = ds["temp_surf"][:,0]
plot_1d_data_np(X, [Y], title="Single var and dim")

## --- access by name (single var, all times)----
Y = ds["temp_surf"].loc[:,"IA"]
plot_1d_data_np(X, [Y], title="Single var and dim (by name)")
# --- access dimension by name (single var, all times)----
Y = ds.sel(location="IA")["temp_surf"]
plot_1d_data_np(X, [Y], title="Single var and dim (location by name)")

## -- Grouping Awesome stuff
# Mean by dimension
Y = ds.mean(dim='location')["temp_surf"]
plot_1d_data_np(X, [Y], title="Mean temp_surf from all locations")

## --------------------- Automatic Masking (cool stuff) -----------
# http://xarray.pydata.org/en/stable/indexing.html#masking-with-where
Y = ds.where(ds.time.isin(pd.date_range("2001-11-01", "2001-12-31", name="time"))).sel(location="IA")["temp_surf"]
plot_1d_data_np(X, [Y], title="Masked dimension (few dot somewhere)")

# --------------------- INTERPOLATION Select non existing dimension values by interpolating --------------------
# --- Nearest neighb, not really interpolation.
method = 'nearest'  # It can be 'nearest, pad, backfill,
Y1 = ds.sel(lat=[1], method=method)["ozmax"]
Y2 = ds.sel(lat=[1.5], method=method)["ozmax"]
plot_1d_data_np(X, [Y1, Y2], title="Interpolating by dim NNeigh")

# --- Real interpolation
method = 'linear'  # 'linear, nearest, zero, slinear, quadratic, cubic'
Y2i = ds.interp(lat=[1.5], method=method)["ozmax"]
plot_1d_data_np(X, [Y1, Y2, Y2i], labels=['time1', 'time 1.5', 'time 1.5 NN'], title="Interpolating by dim Linear")

def adding_and_dropping(ds):
    # Drop dimensions
    data_summary(ds.drop_dims('lat'))

def create_datasets():
    # Fro crating single variables with dimensions (data arrays)
    temp = xr.DataArray([[1, 2], [3, 4]], dims=['lat', 'lon'])

    # Creates a date range
    times = pd.date_range("2000-01-01", "2001-12-31", name="time")
    annual_cycle = np.sin(2 * np.pi * (times.dayofyear.values / 365.25 - 0.28))

    base = 10 + 15 * annual_cycle.reshape(-1, 1)
    tmin_values = base + 3 * np.random.randn(annual_cycle.size, 3)
    tmax_values = base + 10 + 3 * np.random.randn(annual_cycle.size, 3)

    ds = xr.Dataset(
        {
            "temp_surf": (("time", "location"), tmin_values),
            "tmax": (("time", "location"), tmax_values),
            "ozmax": (("time", "lat"), tmax_values),
        },
        {"time": times, "location": ["IA", "IN", "IL"], "lat": [1, 2, 3]},
    )

    data_summary(ds)
    # --- Access a single variable
    access_data(ds)

    # --- Add and drop stuff
    adding_and_dropping(ds)

    # df = ds.to_dataframe()
    # sns.pairplot(df.reset_index(), vars=ds.data_vars)

def edit_datasets():
    ds = xr.tutorial.open_dataset("air_temperature")
    data_summary(ds)

    # -------- Modifying values of a region with a subset of the dimensions (there are more options)
    # http://xarray.pydata.org/en/stable/indexing.html#more-advanced-indexing
    lon = ds.coords["lon"]
    lat = ds.coords["lat"]

    # -------- Changing an index
    air = ds['air']
    air_small = air[:20, : 20]

    air_small.to_netcdf("out.nc")

def read_write():
    data_path = join('..','data')
    file_name = join(data_path,'GFSTodayReduced.nc')
    ds = xr.load_dataset(file_name)
    only_temp = ds['temp_surf']
    # Shows temp at time 0
    plt.imshow(only_temp[0])
    plt.show()

    new_ds = only_temp.to_dataset()
    data_summary(new_ds)

    new_ds.to_netcdf(join(data_path,'OnlyTemp.nc'))
    #  --------- Reading multiple netcdf -------------
    ds = xr.open_mfdataset("/home/olmozavala/Dropbox/TestData/netCDF/HYCOM/S_T_U_V_Surface/*.nc")
    data_summary(ds)

def crop_xr():
    data_path = join('..', 'data')
    file_name = join(data_path, 'GFSTodayReduced.nc')
    ds = xr.load_dataset(file_name)
    ds_crop = ds.sel(lat=slice(24, 30), lon=slice(-84, -78))  # Cropping by value
    # Shows temp at time 0
    plt.imshow(ds_crop['water_u'])
    plt.show()


if __name__ == "__main__":
    data_path = join('..', 'data')
    file_name = join(data_path, 'GFSTodayReduced.nc')
    ds = xr.load_dataset(file_name)

    # create_datasets()
    # edit_datasets()
    # read_write()
    crop_xr()