import yaml
import os

class Config:
    def __init__(self, additional_config_path=None):
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default.yml')
        self.config = yaml.load(open(default_config, 'r'))
        if additional_config_path is not None:
            self.config.update(yaml.load(open(additional_config_path, 'r')))

    def __getitem__(self, item):
        return self.config[item]
