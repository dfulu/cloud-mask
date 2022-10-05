from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import dask
from dask import delayed
from dask.diagnostics import ProgressBar


def location_train_test_split(ds, test_frac=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ds_locs = ds.location.groupby(ds.location).count()
    shuffled_loc_names = np.random.permutation(ds_locs.location)
    cumsum = np.cumsum(ds_locs.sel(location=shuffled_loc_names))
    cumsum = cumsum/cumsum[-1]
    i = (cumsum>test_frac).argmax().item()
    print(f"{i} test locations and {cumsum.shape[0]-i} train locations")
    test_locations = cumsum.isel(location=slice(None, i+1))
    test_mask = np.isin(ds.location, test_locations.location)
    test_set = ds.isel(chip_id=test_mask)
    train_set = ds.isel(chip_id=~test_mask)
    return train_set, test_set


def df_location_train_test_split(df, test_frac=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    df_locs = df.location.groupby(df.location).count()
    shuffled_loc_names = np.random.permutation(df_locs.index)
    cumsum = np.cumsum(df_locs.loc[shuffled_loc_names])
    cumsum = cumsum/cumsum[-1].item()
    i = (cumsum>test_frac).argmax().item()
    print(f"{i} test locations and {cumsum.shape[0]-i} train locations")
    test_locations = cumsum.iloc[:i+1]
    test_mask = np.isin(df.location, test_locations.index)
    df_test = df.iloc[test_mask]
    df_train = df.iloc[~test_mask]
    return df_train, df_test


def get_array(chip_id, root_path="/driven-data/cloud-cover/train_{}", append_label=True):
    band = ['B02', 'B03', 'B04', 'B08']
    xs=[]
    for b in band:
        xs += [np.array(Image.open(f"{root_path.format('features')}/{chip_id}/{b}.tif"))]
    if append_label:
        xs += [np.array(Image.open(f"{root_path.format('labels')}/{chip_id}.tif"))]
    return np.array(xs).astype(np.int16)


class ImageDataset(Dataset):
    
    def __init__(self, ds, augmentation=None):
        self.ds = ds
        self.augmentation = augmentation
        self._image_bands = [b for b in ds.band.values if b!='cloud_mask']
    
    def __len__(self):
        return self.ds.chip_id.shape[0]

    def __getitem__(self, idx):
        # need to set single-threaded scheduler so no clash with DataLoader scheduler
        da = self.ds.images.isel(chip_id=idx).compute(scheduler='single-threaded')
        image = da.sel(band=self._image_bands).values
        label = da.sel(band='cloud_mask').values[None, ...]
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        if self.augmentation:
            image, label = self.augmentation(image, label)
        return image, label


class tiffImageDataset(Dataset):
    
    def __init__(self, df, augmentation=None, root_path="/driven-data/cloud-cover/train_{}", return_label=True):
        self.df = df
        self.augmentation = augmentation
        self._image_bands = ['B02', 'B03', 'B04', 'B08']
        self.root_path = root_path
        self.return_label = return_label
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # need to set single-threaded scheduler so no clash with DataLoader scheduler
        chip_id = self.df.iloc[idx].chip_id
        x = get_array(chip_id, self.root_path, append_label=self.return_label)
        if self.return_label:
            image = x[:-1]
            label = x[-1:]
            label = torch.from_numpy(label).float()
        else:
            image = x
        image = torch.from_numpy(image).float()
        if self.return_label:
            if self.augmentation:
                image, label = self.augmentation(image, label)
            return image, label
        else:
            return image
    
class preloadedTiffImageDataset(tiffImageDataset):
    
    def __init__(self, df, augmentation=None, root_path="/driven-data/cloud-cover/train_{}", return_label=True):
        super().__init__(df, augmentation=augmentation, root_path=root_path, return_label=return_label)
        self.preload()
        
    def preload(self):
        xs = []
        for idx in range(len(self)):
            chip_id = self.df.iloc[idx].chip_id
            xs += [
                dask.array.from_delayed(
                    delayed(get_array)(chip_id, self.root_path, append_label=self.return_label), 
                    shape=(4+self.return_label, 512, 512), 
                    dtype=np.int16,
                )
            ]
        xs = dask.array.array(xs)
        with ProgressBar(dt=10):
            self.xs = xs.compute(scheduler='threads')
            
    def __getitem__(self, idx):
        x = self.xs[idx]
        if self.return_label:
            image = x[:-1]
            label = x[-1:]
            label = torch.from_numpy(label).float()
        else:
            image = x
        image = torch.from_numpy(image).float()
        
        if self.return_label:
            if self.augmentation:
                image, label = self.augmentation(image, label)
            return image, label
        else:
            return image
        