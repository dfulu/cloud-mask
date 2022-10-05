### Training experiments
 - [ ] Use [`kornia`](https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.BinaryFocalLossWithLogits) to train with [focal loss](https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7).
 - [x] **Use [`kornia`](https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.DiceLoss) to train with [dice loss]().**
 - [x] Ditch the IR channel - now back in
 - [x] Try DiceBCE loss which uses BCE to help smooth things out
 - [x] Try focal loss which helps network focus on hard examples
 - [ ] Try training with symmetric CE - robust to noisey labels (picking loss)
 - [ ] Retrain v24 with all channels and to completion and SWA (picking loss)
 - [ ] Same as 39-40 but with combo hinge and bce loss
 - [ ] Tain new network 39-40 on uncleaned data (picking cleanliness)
 - [ ] Same as 39-40 but efficientnet-b1 (picking backbone)
 - [ ] same as 39-40 but Unet++ (picking architecture)
 - [ ] Train many of same model with SWA (final sub)
 - [ ] x nise.same factr fr all bands
 - [ ] Morphological Operations e.g. https://medium.com/swlh/image-processing-with-python-morphological-operations-26b7006c0359

### Network structure
- Try dropout bayesian UNet?
- Train with multiple heads each predicting same mask buyt subject to different losses?
- leaky relus rather than relu?
- Try DeepLab architecture
- [ ] Train network to estimate how much of field is cloud - use to set threshold, or just to catch high mask fields. Add classifier headf at bottom of unet?

### Training procedure
- Accumulate loss in training steps for smoother graphs
- Make a script to continue training a network


### Dataset work
- [x] Remove the datapoints with clearly wrong masks
- [ ] Make a dataloader that will just keep the whole dataset in RAM

### Evaluation
- [x] Write a notebook which will allow me to see where the network is underperforming. Ideally using the same code as in `/submission`.
- Are the tiffs from the same day at same locations? Have they just chunked bigger swatches into these tiffs?
- Have a look at scenes from same locations. Do they look the same?
- [x] Need to know TP, TN, FN, FP

### Post processing
- Let's say I train on something like `lovasz_hinge` loss. If I implemented a dropout bayesian NN I could use the uncertainties from that to maximise my submissions.
    - sample using dropout
    - use sample vosting to establish a probabilitity
    - tune threshold based on this probability
    - or do something more analytically grounded to maximise expected IoU