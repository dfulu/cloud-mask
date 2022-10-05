import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import warnings
import xarray as xr



def normalize_data(data, pixel_max=255, c=10.0, th=0.125):
    max_val = data.max()
    min_val = data.min()
    range_val = max_val - min_val
    norm = (data - min_val) / range_val
    norm = 1 / (1 + np.exp(c * (th - norm)))
    return norm * pixel_max

def plot_xarray_chip(chip, ax=None, cloud_contour=True, infrared_alpha=False, nodata=1, c=10.0, th=0.125):

    r = chip.sel(band='B04')
    g = chip.sel(band='B03')
    b = chip.sel(band='B02')
    
    a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    
    im = np.full(b.shape+(4,), 0)

    pixel_max = 255
    alpha=.3
    linewidths=.5
    
    im[:, :, 0] = normalize_data(r, pixel_max, c, th).astype(np.uint8)
    im[:, :, 1] = normalize_data(g, pixel_max, c, th).astype(np.uint8)
    im[:, :, 2] = normalize_data(b, pixel_max, c, th).astype(np.uint8)
    im[:, :, 3] = a.astype(np.uint8)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(im)
        if infrared_alpha:
            ax.imshow(normalize_data(chip.sel(band='B08'), pixel_max, c, th).astype(np.uint8))
            alpha=1
            linewidths=3
        if cloud_contour:
            ax.contour(chip.sel(band='cloud_mask'), cmap='gist_gray', vmin=.5, vmax=5, alpha=.3, linewidths=.5)
    return ax