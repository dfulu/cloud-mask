cp -r /driven-data/cloud-cover/train_features/$1 ../data/submission_test/data/test_features/.
cp /driven-data/cloud-cover/train_labels/$1.tif ../data/submission_test/data/test_labels/.

# update metedata file
python create_metadata_csv.py