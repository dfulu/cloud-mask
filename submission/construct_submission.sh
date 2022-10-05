#!/bin/bash
# e.g.
# bash construct_submission.sh TESTSUB ../lightning_logs/version_18/checkpoints/epoch=35-step=23795.ckpt

checkpoints=${@:2}
subname=$1

echo '----------Building----------'
mkdir submission_src
mkdir submission_src/checkpoints
echo SUBMISSION: $subname > submission_src/submission_summary.txt

# copy source files
cp src/main.py submission_src/.
cp src/apply_model.py submission_src/.
cp ../datadrivencloud/transforms.py submission_src/.
cp ../datadrivencloud/data.py submission_src/.
cp ../datadrivencloud/modules/model_modules.py submission_src/.
cp configs/reduction.yaml submission_src/.

# strip and copy model files
cp ../datadrivencloud/modules/model_modules.py .
cp ../datadrivencloud/transforms.py .
cp src/apply_model.py .
python strip_model_checkpoints.py $checkpoints --outdir submission_src/checkpoints >> submission_src/submission_summary.txt 
rm model_modules.py transforms.py apply_model.py

# test
echo '----------Testing-----------'
cp configs/local_config.yaml submission_src/config.yaml
cd submission_src
python main.py --cpu
cd ..
python official_scoring.py

# clean up and compile
echo '----------Cleaning----------'
rm -r ../data/submission_test/predictions/*
rm submission_src/config.yaml
cp configs/config.yaml submission_src/config.yaml

cd submission_src
zip -r ../submission.zip ./*
cd ..
rm -r submission_src

echo Done.