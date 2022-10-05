from datadrivencloud.modules import CloudModel, xarrayDataModule, tiffDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
import kornia.augmentation as K
import torch
from pytorch_lightning.loggers import TensorBoardLogger



import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--all_data', action='store_true')
parser.add_argument('--uplus', action='store_true')
parser.add_argument('--robust', action='store_true')
parser.add_argument('--dev', action='store_true')
parser.add_argument('--seed', type=int)
parser.add_argument('--loss_name', type=str)
parser.add_argument('--backbone', type=str)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--run_name', type=str, default='default')
parser.add_argument('--preproc', type=str, default='LogTransform')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--tune_lr', action='store_true')
parser.add_argument('--load_train', action='store_true')
parser.add_argument('--split_by_location', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)

accumulate_grad_batches = {5:2, 40:4, 70:8, 100:4}
main_train_epochs = 100
swa_anneal_epochs = 5
swa_post_anneal_epochs = 10
total_epochs = main_train_epochs + swa_anneal_epochs + swa_post_anneal_epochs

###############################################################################

if args.all_data:
    path = "../data/train_metadata_filtered.csv"
else:
    path = "../data/train_metadata.csv"

    
# init data
augmentation = K.AugmentationSequential(
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),
    data_keys=["input", "mask"],  # Just to define the future input here.
    return_transform=False,
    same_on_batch=False,
    keepdim=True,
)

    
"""
datamodule = xarrayDataModule(
    zarr_store_path = '../data/train_zarr',
    df_meta_path = path,
    val_frac=0.1,
    data_split_seed=args.seed,
    num_workers=0 if args.load_train else 4,
    batch_size=8,
    augmentation=augmentation,
    preload_val=not args.dev,
    preload_train=not args.dev and args.load_train,
)
"""

datamodule = tiffDataModule(
    df_meta_path=path,
    val_frac=0.1,
    data_split_seed=args.seed,
    split_by_location=args.split_by_location,
    num_workers=0 if args.load_train else 4,
    batch_size=8,
    augmentation=augmentation,
    preload_val=not args.dev,
    preload_train=not args.dev and args.load_train,
    root_path="../data/train_{}",
)
#"""

datamodule.setup()

#steps_per_epoch = len(datamodule.train_dataloader())//accumulate_grad_batches

###############################################################################


model = CloudModel(
    hparams = dict(
        backbone=args.backbone,
        lr=args.lr,
        loss_name = args.loss_name, #['bce', 'lovasz', 'dice', 'focal2', 'dicebce', 'sce', 'dicesce']
        transform = args.preproc, # ['Div2000', 'LogTransform', 'QuarterPower']
        bands=[0,1,2,3],
        uplus=args.uplus,
        robust=args.robust,
        seed=args.seed,
        #steps_per_epoch=steps_per_epoch,
    )
)


# callbacks

callbacks = []
callbacks += [ModelCheckpoint(
    monitor="val_accumulated_iou", 
    mode="max",
    verbose=True,
    save_top_k=1,
    save_last=True,
)]

callbacks += [LearningRateMonitor()]

if total_epochs>main_train_epochs:
    callbacks += [StochasticWeightAveraging(
        swa_lrs=args.lr/500.,
        swa_epoch_start=main_train_epochs, 
        annealing_epochs=swa_anneal_epochs, 
        annealing_strategy='cos', 
    )]


logger = TensorBoardLogger('../lightning_logs', name=args.run_name)

# train
trainer = pl.Trainer(
    gpus= 0 if args.cpu else [args.gpu,],
    fast_dev_run = args.dev,
    log_every_n_steps = 50,
    default_root_dir='../.',
    callbacks = callbacks,
    max_epochs=total_epochs,
    logger=logger,
    accumulate_grad_batches=accumulate_grad_batches,
    val_check_interval=1.0,
    auto_lr_find=args.tune_lr,
    #precision=16,
)

trainer.main_train_epochs = main_train_epochs

if args.tune_lr:
    trainer.tune(model, datamodule=datamodule)
trainer.fit(model, datamodule=datamodule)