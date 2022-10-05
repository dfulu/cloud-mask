from dask import delayed
import dask
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from datadrivencloud.data import get_array

def add_coord(dataset, value, name):
    c = xr.DataArray(value, dims=['chip_id'], coords=[dataset.chip_id], name=name)
    return dataset.assign_coords({name:c})

def tiffs_to_xarray(root_path="/driven-data/cloud-cover/train_{}", df_meta_path="/driven-data/cloud-cover/train_metadata.csv", n=None):
    df = pd.read_csv(df_meta_path).iloc[:n]
    xy = np.arange(512)
    band = ['B02', 'B03', 'B04', 'B08', 'cloud_mask']
    xs = []

    for i, row in tqdm(df.iterrows()):
        xs += [dask.array.from_delayed(delayed(get_array)(row.chip_id, root_path=root_path), shape=(5, 512, 512), dtype=np.int16)]
    xs = dask.array.array(xs)
    da = xr.DataArray(xs, dims=['chip_id', 'band', 'x', 'y'], coords=[df.chip_id.values, band, xy, xy], name='images')
    ds = da.to_dataset()
    ds = add_coord(ds, df.location.values, 'location')
    ds = add_coord(ds, df.datetime.values, 'datetime')
    return ds