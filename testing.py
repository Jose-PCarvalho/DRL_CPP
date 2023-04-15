import yaml

with open('configs/training.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())    # load the config file

