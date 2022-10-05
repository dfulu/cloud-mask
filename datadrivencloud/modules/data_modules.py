from typing import Optional, List, Union

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import train_test_split

from dask.diagnostics import ProgressBar

from .. data import (
    location_train_test_split, ImageDataset,
    df_location_train_test_split, tiffImageDataset,
    preloadedTiffImageDataset
)
from .. tiffs_to_xarray import tiffs_to_xarray
 
    
class CloudDataModule(pl.LightningDataModule):
    def __init__(
            self,
            df_meta_path,
            **kwargs,
        ):
        super().__init__()
        self.df_meta_path = df_meta_path
        self.num_workers = kwargs.get("num_workers", 2)
        self.batch_size = kwargs.get("batch_size", 16)
        self.data_split_seed = kwargs.get("data_split_seed", None)
        self.split_by_location = kwargs.get("split_by_location", True)
        self.val_frac = kwargs.get("val_frac", 0.2)
        self.augmentation = kwargs.get("augmentation", None)
        
        
    def load_split_df(self):
        df = pd.read_csv(self.df_meta_path)
        
        if self.split_by_location:
            df_train, df_val = df_location_train_test_split(
                df, 
                self.val_frac, 
                self.data_split_seed
            )
        else:
            df_train, df_val = train_test_split(
                df, 
                test_size=self.val_frac, 
                random_state=self.data_split_seed, 
                shuffle=True
            )
            
        return df_train, df_val

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )
    
class xarrayDataModule(CloudDataModule):
    def __init__(
            self,
            zarr_store_path: Union[str,None],
            df_meta_path: str,
            root_path: Union[str,None]=None,
            preload_val=False,
            preload_train=False,
            **kwargs,
        ):
        super().__init__(df_meta_path, **kwargs)
        self.zarr_store_path = zarr_store_path
        self.preload_val = preload_val
        self.preload_train = preload_train
        self.root_path = root_path


    def setup(self, stage: Optional[str] = None):
        # split dataset
        if stage not in (None, "fit"):
            raise NotImplementedError
            
        df_train, df_val = self.load_split_df()
        
        if self.zarr_store_path:
            ds = xr.open_zarr(self.zarr_store_path).astype(np.int16)
        else:
            assert self.root_path is not None
            print('constructing xarray dataset from tifs')
            ds = tiffs_to_xarray(root_path=self.root_path, df_meta_path=self.df_meta_path)
        
        ds.location.load()

        ds_train = ds.sel(chip_id=df_train.chip_id.values)
        ds_val = ds.sel(chip_id=df_val.chip_id.values)

        if self.preload_val:
            with ProgressBar(dt=1):
                ds_val = ds_val.compute()
        if self.preload_train:
            with ProgressBar(dt=1):
                ds_train = ds_train.compute()
        
        self.train_dataset = ImageDataset(
            ds_train,
            augmentation=self.augmentation,
        )
        self.val_dataset = ImageDataset(
            ds_val,
        )

    
class tiffDataModule(CloudDataModule):
    def __init__(
            self,
            df_meta_path: str,
            preload_val=False,
            preload_train=False,
            root_path="/driven-data/cloud-cover/train_{}",
            **kwargs,
        ):
        super().__init__(df_meta_path, **kwargs)
        
        self.preload_val = preload_val
        self.preload_train = preload_train
        self.root_path = root_path

    def setup(self, stage: Optional[str] = None):
        # split dataset
        if stage not in (None, "fit"):
            raise NotImplementedError
        
        df_train, df_val = self.load_split_df()

        if self.preload_train:
            self.train_dataset = preloadedTiffImageDataset(
                df_train,
                augmentation=self.augmentation,
                root_path=self.root_path,
            )
        else:
            self.train_dataset = tiffImageDataset(
                df_train,
                augmentation=self.augmentation,
                root_path=self.root_path,
            )

        if self.preload_val:
            self.val_dataset = preloadedTiffImageDataset(
                df_val,
                root_path=self.root_path,
            )
        else:
            self.val_dataset = tiffImageDataset(
                df_val,
                root_path=self.root_path,
            )
    