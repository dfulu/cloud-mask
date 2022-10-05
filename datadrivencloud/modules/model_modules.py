from typing import Optional, List, Union

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import kornia
from torch import nn

#if True:
try:
    from .. losses import (
        SCELoss, DiceLoss, DiceBCELoss, 
        IoULoss, LovaszHingeLoss, iou_metric, FocalLoss,
        DiceSCELoss, Reverse, Symmetric
    )
    from .. loggers import create_image_figure
    from .. transforms import Div2000, LogTransform, QuarterPower, Visual
#try:
#    pass
except ImportError:
    # this should only happen when uploading submission code
    from transforms import Div2000, LogTransform,  QuarterPower, Visual
    
    class Dummy:
        def __init__(self, *args, **kwags):
            return
    
    SCELoss=DiceLoss=DiceBCELoss=IoULoss=LovaszHingeLoss=FocalLoss=DiceSCELoss=Reverse=Symmetric = Dummy
    iou_metric = lambda x: x


class CloudModel(pl.LightningModule):
    def __init__(
        self,
        hparams: dict = {},
    ):
        """
        Instantiate the CloudModel class based on the pl.LightningModule
        (https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).


        """
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # optional modeling params
        self.backbone = self.hparams.get("backbone", "resnet34")
        self.weights = self.hparams.get("weights", None)
        #self.learning_rate = self.hparams.get("lr", 1e-3)
        self.patience = self.hparams.get("patience", 5)
        self.plot_n_images = self.hparams.get("plot_n_images", 6)
        self.bands = self.hparams.get("bands", [0,1,2,3])
        loss_name = self.hparams.get("loss_name", 'bce')
        self.set_loss(loss_name)
        self.model = self._prepare_model()
        self.transform = self._get_transform()
        self.seed = self.hparams.get("seed", 0)
        
        self.sample_val_image_from = 0
        self.max_val_batch_index = 1

    ## Required LightningModule methods ##
    
    def set_loss(self, name):
        losses = {
            'lovasz': LovaszHingeLoss(),
            'bce': F.binary_cross_entropy,
            'jaccard': IoULoss(),
            'dice':smp.utils.losses.DiceLoss(),
            'focal2':FocalLoss(alpha=0.5, gamma=2.0),
            'dicebce':DiceBCELoss(alpha=0.2), # higher alpha = more BCE
            'dicebce50':DiceBCELoss(alpha=0.5), # higher alpha = more BCE
            'revdicebce':Reverse(DiceBCELoss(alpha=0.2)), # higher alpha = more BCE,
            'symdicebce':Symmetric(DiceBCELoss(alpha=0.2)), # higher alpha = more BCE
            'symlovasz':Symmetric(LovaszHingeLoss()),
            'sce':SCELoss(),
            'dicesce':DiceSCELoss(),
            
        }
        assert name in losses
        self.loss_name = name
        self.loss_fn = losses[name]
        
    
    def forward(self, image: torch.Tensor):
        return self.model(self.transform(image[:, self.bands]))

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step.
        """

        # Load images and labels
        x, y = batch

        # Forward pass
        preds = self.forward(x)

        # Log batch loss
        loss = self.loss_fn(preds, y)
                
        self.log("training_loss", loss, on_epoch=True, on_step=False)
            
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step.
        """
        
        # Load images and labels
        x, y = batch

        # Forward pass & softmax
        preds = self.forward(x)

        # Log batch IOU
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss) # mean automatically taken over validation set 
        
        # Accumulate IoU over validation set
        self.log("val_iou_loss", iou_metric(preds, y)) # mean automatically taken over validation set 
        preds_bool = (preds>0.5).bool()
        y_bool = y.bool()
        intersection = torch.logical_and(preds_bool, y_bool).sum()
        union = torch.logical_or(preds_bool, y_bool).sum()
        
        global_step = self.trainer.global_step
        epoch = self.trainer.current_epoch
        val_batches = self.trainer.num_val_batches[0]
        if batch_idx==0:
            # plot same images each val step so can compare across time
            self.logger.experiment.add_figure(
                'stable_val_examples', 
                create_image_figure(x, preds, y, self.plot_n_images, False), 
                global_step
            )
        if batch_idx==(epoch%val_batches):
            # plot random images each val step so can getter better coverage
            self.logger.experiment.add_figure(
                'random_val_examples', 
                create_image_figure(x, preds, y, self.plot_n_images, True), 
                global_step
            )

        return intersection, union
    
    def validation_epoch_end(self, val_outs):
        total_intersection = 0
        total_union = 0
        for intersection, union in val_outs:
            total_intersection += intersection
            total_union += union
        self.log("val_accumulated_iou", total_intersection/total_union)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.hparams["lr"], amsgrad=True)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.main_train_epochs) #, T_max=10,
        #sch = torch.optim.lr_scheduler.OneCycleLR(
        #    opt, self.hparams["lr"], epochs=self.trainer.main_train_epochs,
        #    steps_per_epoch=self.hparams["steps_per_epoch"],
        #    three_phase=True, pct_start=0.4,
        #    anneal_strategy='linear',
        #)
        #sch = torch.optim.lr_scheduler.CyclicLR(opt, 
        #    base_lr=self.hparams["lr"]/64, max_lr=self.hparams["lr"], 
        #    step_size_up=(self.hparams["steps_per_epoch"]*self.trainer.main_train_epochs)//8, 
        #    mode='triangular2', cycle_momentum=False)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
            factor=0.5,
            patience=self.patience,
            threshold=2e-4,
        )
        sch = {"scheduler": sch, "monitor": "training_loss"}
        return [opt], [sch]

    ## Convenience Methods ##
    def _get_transform(self):
        transforms = {
            'Div2000':Div2000(), 
            'LogTransform':LogTransform(),  
            'QuarterPower':QuarterPower(),
            'Visual':Visual(),
        }
        transform = self.hparams.get("transform", "Div2000")
        assert transform in transforms
        return transforms[transform]
        

    def _prepare_model(self):
        if self.hparams.get("uplus", False):
            architecture = smp.UnetPlusPlus
        else:
            architecture = smp.Unet
        model = architecture(
            encoder_name=self.backbone,
            encoder_weights=self.weights,
            in_channels=len(self.bands),
            activation='sigmoid',
            classes=1,
        )
        if self.hparams.get("robust", False):
            model = RobustModel(model, alpha=-5, beta=-5)
        return model
    
class RobustModel(nn.Module):
    def __init__(self, model, alpha, beta):
        super(RobustModel, self).__init__()
        self.model = model
        self.alpha = nn.parameter.Parameter(torch.Tensor([alpha]))
        self.beta = nn.parameter.Parameter(torch.Tensor([beta]))

    def forward(self, x):
        P = self.model(x)
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        P_robust = alpha+(1-alpha-beta)*P
        return P_robust