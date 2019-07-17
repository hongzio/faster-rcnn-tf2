import yaml

class Config:
    def __init__(self, additional_config_path=None):
        self.config = yaml.load(open('config/default.yml', 'r'))
        if additional_config_path is not None:
            self.config.update(yaml.load(open(additional_config_path, 'r')))

    def __getitem__(self, item):
        return self.config[item]
