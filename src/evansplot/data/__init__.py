import xarray
import importlib

def ndvi_time():
    return xarray.open_dataset(importlib.resources.files(__name__).joinpath('ndvi_time.nc'))

def gcm():
    return xarray.open_dataset(importlib.resources.files(__name__).joinpath('gcm_co2_pre.nc'))

def pr_mme_change():
    return xarray.open_dataset(importlib.resources.files(__name__).joinpath('pr_mme_change_sresa2.nc'))
    