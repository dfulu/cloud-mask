from . utils import plot_xarray_chip
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings


def create_image_figure(x, pred, y, n=1, random_image=False):
    assert n<=x.shape[0], f"cannot plot {n} images when validation batch size is {x.shape[0]}"
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    pred = pred.cpu().numpy()
    xy = np.arange(512)
    band = ['B02', 'B03', 'B04', 'B08', 'cloud_mask']
    chip = np.arange(x.shape[0])
    ds_y = xr.DataArray(np.concatenate([x, y], axis=1), 
                      dims=['chip', 'band', 'x', 'y'], coords=[chip, band, xy, xy], name='images')
    ds_pred = xr.DataArray(np.concatenate([x, pred], axis=1), 
                      dims=['chip', 'band', 'x', 'y'], coords=[chip, band, xy, xy], name='images')
    fig, axes = plt.subplots(n,4, figsize=(12, 3*n), sharex=True, sharey=True)
    axes = axes.reshape(n,4)
    selection = np.arange(x.shape[0])
    if random_image:
        selection = np.random.permutation(selection)
    for i, c in zip(range(n), selection[:n]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            plot_xarray_chip(ds_y.isel(chip=c), ax=axes[i, 0], cloud_contour=True, nodata=-torch.inf, infrared_alpha=False)
            axes[i, 1].contourf(ds_y.isel(chip=c).sel(band='cloud_mask'), cmap='Blues', levels=[0.5,1], vmin=0, vmax=1)
            axes[i, 1].contour(ds_y.isel(chip=c).sel(band='cloud_mask'), cmap='Blues', vmin=0, vmax=1, linewidths=.5)
            axes[i, 1].set_aspect(axes[i, 0].get_aspect())

            plot_xarray_chip(ds_pred.isel(chip=c), ax=axes[i, 2], cloud_contour=True, nodata=-torch.inf, infrared_alpha=False)
            axes[i, 3].contourf(ds_pred.isel(chip=c).sel(band='cloud_mask'), cmap='Blues', levels=[0.5,1], vmin=0, vmax=1)
            axes[i, 3].contour(ds_pred.isel(chip=c).sel(band='cloud_mask'), cmap='Blues', vmin=0, vmax=1, linewidths=.5)
            axes[i, 3].set_aspect(axes[i, 0].get_aspect())
    return fig