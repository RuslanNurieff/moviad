import os

import yaml

class BenchmarkConfig:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_config(self):
        return self.config

    def get_config_value(self, key):
        return self.config[key]

    def get_config_value_or_default(self, key, default):
        return self.config.get(key, default)


class DatasetConfig:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.realiad_root_path = self.convert_path(self.config['datasets']['realiad']['root_path'])
        self.realiad_json_root_path = self.convert_path(self.config['datasets']['realiad']['json_root_path'])
        self.visa_root_path = self.convert_path(self.config['datasets']['visa']['root_path'])
        self.visa_csv_path = self.convert_path(self.config['datasets']['visa']['csv_path'])
        self.mvtec_root_path = self.convert_path(self.config['datasets']['mvtec']['root_path'])

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def convert_path(self, path):
        return os.path.normpath(path)