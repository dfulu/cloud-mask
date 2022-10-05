# Compiling submission

This folder is dedicated to packaging up a code submission for the competition. See [this page](https://www.drivendata.org/competitions/83/cloud-cover/page/412) for more info on the submission requirements.

The main point of entry is to run the line e.g.

```
bash construct_submission.sh ../lightning_logs/version_18/checkpoints/epoch=35-step=23795.ckpt
```

The checkpoint, and all the code files needed to recreate and run the model object are packed into a `submission.zip` file. 
An end to end test is also run to test the checkpointed model on a small local supply of data. The generated image `local_test_results.png` shows the predictions made by the model alongside the label. 

--------------------------------

### Convenience functions 

Add a chip from the local train set to the created directory of test data. This encludes metadata recompiling.
```
bash add_chip.sh <chip_id>
```

Search through all the local test data and compile the needed info into metadata CSV.
```
python create_metadata_csv.py
```