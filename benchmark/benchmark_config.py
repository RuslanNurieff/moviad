import os
import json
import yaml


class BackboneRunConfig:
    def __init__(self, backbone_info):
        self.name = list(backbone_info.keys())[0]
        self.ad_layers = backbone_info[self.name]

    def get_name(self):
        return self.name

    def get_ad_layers(self):
        return self.ad_layers


class DatasetRunConfig:
    def __init__(self, dataset_info):
        self.type = dataset_info['type']
        self.class_name = dataset_info['class']

    def get_type(self):
        return self.type

    def get_class_name(self):
        return self.class_name


class RunConfig:
    """
    Configuration for a single run of the benchmark.
    """

    model: str
    dataset_type: str
    class_name: str
    backbone: str
    ad_layers: list
    contamination: float

    def __init__(self, model, dataset_type, class_name, backbone, ad_layers, contamination):
        self.model = model
        self.dataset_type = dataset_type
        self.class_name = class_name
        self.backbone = backbone
        self.ad_layers = ad_layers
        self.contamination = contamination


class BenchmarkRun:
    def __init__(self, benchmark_section):
        self.model = benchmark_section['model']
        self.datasets = [DatasetRunConfig(dataset) for dataset in benchmark_section['datasets']]
        self.backbones = [BackboneRunConfig(backbone) for backbone in benchmark_section['backbones']]
        self.contamination = benchmark_section.get('contamination', [0])

    def get_runs(self):
        runs = []
        for dataset in self.datasets:
            for backbone in self.backbones:
                for contamination in self.contamination:
                    runs.append(RunConfig(self.model, dataset.get_type(), dataset.get_class_name(), backbone.get_name(),
                                          backbone.get_ad_layers(), contamination))
        return runs

    def get_datasets(self):
        return self.datasets

    def get_backbones(self):
        return self.backbones

    def get_model(self):
        return self.model


class BenchmarkConfig:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.benchmark_runs = [BenchmarkRun(benchmark) for benchmark in self.config['benchmark']]

    def load_config(self, config_file):
        assert os.path.exists(config_file), f"Config file {config_file} does not exist"
        ext = os.path.splitext(config_file)[1].lower()
        if ext == '.json':
            return self.load_json_config(config_file)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

    def load_json_config(self, config_file):
        with open(config_file, 'r') as file:
            return json.load(file)

    def get_benchmark_runs(self):
        return self.benchmark_runs

