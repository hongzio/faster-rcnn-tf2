import os

import yaml


class Config:
    def __init__(self, additional_config_path=None):
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default.yml')
        with open(default_config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        if additional_config_path is not None:
            with open(additional_config_path, 'r') as f:
                self.config.update(yaml.load(f, Loader=yaml.FullLoader))

    def __getitem__(self, item):
        return self.config[item]
