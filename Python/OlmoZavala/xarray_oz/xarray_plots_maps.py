import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

in_file = "./test_data/hycom_glby_930_2021052912_t000_uv3z.nc"
ds = xr.load_dataset(in_file)
print(ds.info())

## ---------- Basic (it is by variable name)
ds.surf_u.plot()
plt.show()

## ---------- FIgure Size
ds.surf_u.plot(aspect=2.5, size=5)
plt.show()

## ---------- Override axis
# TODO MISSING HOW TO OVERRIDE AXIS
ds.surf_u.plot()
plt.title("My new title")
plt.show()

## ---------- Multiple plots
fig, axs = plt.subplots(1,2, figsize=(12,6))
ds.surf_u.plot(ax=axs[0])
ds.surf_v.plot(ax=axs[1])
plt.show()

## ---------- Colorbar / colormaps
fig, axs = plt.subplots(1,2, figsize=(12,6))
p1 = ds.surf_u.plot(ax=axs[0], cmap='viridis')
p2 = ds.surf_v.plot(ax=axs[1], cmap='Blues_r')
plt.show()

## -------- Advanced maps
p = ds.surf_u.plot(subplot_kws=dict(projection=ccrs.PlateCarree(), facecolor="gray"),
               cmap='inferno')
p.axes.set_global()
p.axes.coastlines()
plt.show()

##

