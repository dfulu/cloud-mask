import yaml
import pandas as pd
import glob

# load config
with open('local_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

chip_directory = f"{config['ROOT_DIRECTORY']}/data/test_features"

feature_dirs = glob.glob(f"{chip_directory}/*")
feature_dirs = [f.split('/')[-1] for f in feature_dirs]
df = pd.DataFrame({'chip_id':feature_dirs})
df.to_csv(f"{config['ROOT_DIRECTORY']}/data/test_metadata.csv")