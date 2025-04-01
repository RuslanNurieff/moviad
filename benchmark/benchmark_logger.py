import os
from abc import abstractmethod
import pandas as pd
from benchmark_config import RunConfig, BenchmarkConfig
import inspect

class BenchmarkLogger:
    """
    Abstract class for logging benchmark runs. This class is responsible for
    managing the logging of benchmark runs, including saving results and
    updating the status of runs.
    """

    @abstractmethod
    def update(self, run: RunConfig, state: str, error: str = ""):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def run_exists(self, run):
        pass

class WandbLogger(BenchmarkLogger):
    """
    Concrete implementation of BenchmarkLogger that logs benchmark runs to Weights & Biases (wandb).
    """

    def __init__(self):
        import wandb
        self.wandb = wandb

    def update(self, run: RunConfig, state: str, error: str = "") -> None:
        self.wandb.log({
            "model": run.model,
            "dataset_type": run.dataset_type,
            "class_name": run.class_name,
            "backbone": run.backbone,
            "ad_layers": str(run.ad_layers),
            "contamination": run.contamination,
            "state": state,
            "error": error
        })

    def save(self):
        pass

    def run_exists(self, run: RunConfig):
        return False


class CsvLogger(BenchmarkLogger):
    """
    Concrete implementation of BenchmarkLogger that logs benchmark runs to a CSV file.
    """

    def __init__(self, csv_file):
        self.columns = [attr_name for attr_name, _ in inspect.get_annotations(RunConfig).items()
              if not attr_name.startswith('__')]
        self.additional_columns = ["runs", "state", "error"]
        self.csv_file = csv_file
        self.df = pd.DataFrame(columns=self.columns + self.additional_columns)
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.DataFrame(columns=self.columns + self.additional_columns)
            self.df.to_csv(csv_file, index=False)


    def existing_row(self, run: RunConfig):
        filter_dict = {
            "model": run.model,
            "dataset_type": run.dataset_type,
            "class_name": run.class_name,
            "backbone": run.backbone,
            "ad_layers": str(run.ad_layers),
            "contamination": run.contamination
        }

        filter_expr = pd.Series(True, index=self.df.index)
        for col, val in filter_dict.items():
            filter_expr = filter_expr & (self.df[col] == val)

        existing_row = self.df[filter_expr]
        return existing_row

    def update(self, run: RunConfig, state: str, error: str = "") -> None:
        existing_row = self.existing_row(run)

        if not existing_row.empty:
            self.df.loc[existing_row.index, "runs"] += 1
            self.df.loc[existing_row.index, "state"] = state
            self.df.loc[existing_row.index, "error"] = error
        else:
            new_row = pd.DataFrame([{
                "model": run.model,
                "dataset_type": run.dataset_type,
                "class_name": run.class_name,
                "backbone": run.backbone,
                "ad_layers": run.ad_layers,
                "contamination": run.contamination,
                "runs": 1,
                "state": state,
                "error": error
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)

    def save(self):
        self.df.to_csv(self.csv_file, index=False)

    def run_exists(self, run: RunConfig):
        return not self.existing_row(run).empty


