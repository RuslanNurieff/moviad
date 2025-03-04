import argparse

import torch
from PIL.PngImagePlugin import is_cid
from sympy import false

import wandb
import pandas as pd
import os

from benchmark_config import DatasetRunConfig, BenchmarkConfig
from moviad.datasets.builder import DatasetType, DatasetConfig
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.entrypoints.cfa import CFAArguments, train_cfa
from moviad.entrypoints.padim import train_padim, PadimArgs
from moviad.entrypoints.patchcore import PatchCoreArgs, train_patchcore
from moviad.entrypoints.stfpm import STFPMArgs, train_stfpm

seed = 42
columns = ["Method", "Dataset type", "Class name", "Backbone", "AD layers",
               "Contamination", "Runs", "State", "Error"]

def update_dataframe(df, run, state="Running", error=""):
    existing_row = df[(df["Method"] == run.model) &
                      (df["Dataset type"] == run.dataset_type) &
                      (df["Class name"] == run.class_name) &
                      (df["Backbone"] == run.backbone) &
                      (df["AD layers"] == str(run.ad_layers)) &
                      (df["Contamination"] == run.contamination)]

    if not existing_row.empty:
        df.loc[existing_row.index, "Runs"] += 1
        df.loc[existing_row.index, "State"] = state
    else:
        new_row = pd.DataFrame([{
            "Method": run.model,
            "Dataset type": run.dataset_type,
            "Class name": run.class_name,
            "Backbone": run.backbone,
            "AD layers": run.ad_layers,
            "Contamination": run.contamination,
            "Runs": 1,
            "State": state,
            "Error": error
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def generate_full_checklist(df, benchmark_config: BenchmarkConfig):
    new_runs = []
    for benchmark_run in benchmark_config.get_benchmark_runs():
        for run in benchmark_run.get_runs():
            if run_exists(df, run): # Skip if run already exists
                continue

            run_data = {
                "Method": run.model,
                "Dataset type": run.dataset_type,
                "Class name": run.class_name,
                "Backbone": run.backbone,
                "AD layers": str(run.ad_layers),
                "Contamination": float(run.contamination),
                "Runs": 0,
                "State": "To run",
                "Error": ""
            }

            new_runs.append(run_data)

    # Create DataFrame with new runs
    if new_runs:
        new_df = pd.DataFrame(new_runs, columns=columns)
        # Concatenate with existing DataFrame
        df = pd.concat([df, new_df], ignore_index=True)

    return df


def save_dataframe(df, csv_file):
    df.to_csv(csv_file, index=False)


def run_exists(df, run):
    existing_row = df[(df["Method"] == run.model) &
                      (df["Dataset type"] == run.dataset_type) &
                      (df["Class name"] == run.class_name) &
                      (df["Backbone"] == run.backbone) &
                      (df["AD layers"] == str(run.ad_layers)) &
                      (df["Contamination"] == run.contamination) &
                      (df["State"] == "Completed")]
    return not existing_row.empty


def benchmark_cfa(args: CFAArguments, df, csv_file):
    logger = wandb.init(project="moviad_benchmark", group="cfa")
    logger.config.update({
        "ad_model": "cfa",
        "dataset": args.dataset_type,
        "category": args.category,
        "backbone": args.backbone,
        "ad_layers": args.ad_layers,
        "seed": args.seed,
        "contamination_ratio": args.contamination_ratio
    }, allow_val_change=True)
    logger.tags = ["cfa", args.dataset_type, args.backbone]
    if args.contamination_ratio > 0:
        logger.tags += tuple(["contaminated"])
    logger.name = f"cfa_{args.dataset_type}_{args.backbone}"
    train_cfa(args, logger)
    logger.finish()


def benchmark_padim(args: PadimArgs, df, csv_file):
    logger = wandb.init(project="moviad_benchmark", group="padim")
    logger.config.update({
        "ad_model": "padim",
        "dataset": args.dataset_type,
        "category": args.category,
        "backbone": args.backbone,
        "ad_layers": args.ad_layers,
        "seed": args.seed,
        "contamination_ratio": args.contamination_ratio
    }, allow_val_change=True)
    logger.tags = ["padim", args.dataset_type, args.backbone]
    if args.contamination_ratio > 0:
        logger.tags += tuple(["contaminated"])
    logger.name = f"padim_{args.dataset_type}_{args.backbone}"
    train_padim(args, logger)
    logger.finish()


def benchmark_patchcore(args: PatchCoreArgs, df, csv_file):
    group_name = "patchcore_quantized" if args.quantized else "patchcore"
    logger = wandb.init(project="moviad_benchmark", group="patchcore")
    logger.config.update({
        "ad_model": group_name,
        "dataset": DatasetType.MVTec,
        "category": args.category,
        "backbone": args.backbone,
        "ad_layers": args.ad_layers,
        "seed": args.seed,
        "contamination_ratio": args.contamination_ratio
    }, allow_val_change=True)
    logger.tags = [group_name, args.dataset_type, args.backbone]
    if args.contamination_ratio > 0:
        logger.tags += tuple(["contaminated"])
    logger.name = f"{group_name}_{args.dataset_type}_{args.backbone}"
    train_patchcore(args, logger)
    logger.finish()


def benchmark_stfpm(args: STFPMArgs, df, csv_file):
    logger = wandb.init(project="moviad_benchmark", group="stfpm")
    logger.config.update({
        "ad_model": "stfpm",
        "dataset": DatasetType.MVTec,
        "category": args.categories,
        "backbone": args.backbone,
        "ad_layers": args.ad_layers,
        "seed": args.seed,
        "contamination_ratio": args.contamination_ratio
    }, allow_val_change=True)
    logger.tags = ["stfpm", args.dataset_type, args.backbone]
    if args.contamination_ratio > 0:
        logger.tags += tuple(["contaminated"])
    logger.name = f"stfpm_{args.dataset_type}_{args.backbone}"
    train_stfpm(args, logger)
    logger.finish()


class BenchmarkArgs:
    def __init__(self, config_file, mode,checklist_path, device):
        self.config_file = config_file
        self.mode = mode
        self.checklist_path = checklist_path
        self.device = device

    @classmethod
    def from_parser(cls, args):
        return cls(
            config_file=args.config_file,
            mode=args.mode,
            checklist_path=args.checklist_path,
            device=args.device
        )


def is_cuda_device_available(device_name):
    if not torch.cuda.is_available():
        return False

    if device_name == "cpu":
        return True

    if device_name.startswith("cuda:"):
        try:
            device_index = int(device_name.split(":")[1])
            return device_index < torch.cuda.device_count()
        except (ValueError, IndexError):
            return False

    return False

def main(benchmark_args: BenchmarkArgs):
    # Parse the config file into DatasetConfig and BenchmarkConfig
    dataset_config = DatasetConfig(benchmark_args.config_file)
    benchmark_config = BenchmarkConfig(benchmark_args.config_file)
    csv_file = "benchmark_checklist.csv" if benchmark_args.checklist_path is None else benchmark_args.checklist_path

    if not is_cuda_device_available(benchmark_args.device):
        raise RuntimeError(f"CUDA device '{benchmark_args.device}' is not available. "
                           f"Available devices: CPU and CUDA devices 0 to {torch.cuda.device_count() - 1 if torch.cuda.is_available() else 'none'}")

    device = torch.device(benchmark_args.device)

    print("DEVICE: ", device)

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=columns)

    if benchmark_args.mode == "generate-checklist":
        df = generate_full_checklist(df, benchmark_config)
        save_dataframe(df, "benchmark_checklist.csv")
        return

    for benchmark_run in benchmark_config.get_benchmark_runs():
        for run in benchmark_run.get_runs():
            if run_exists(df, run):
                print(
                    f"Run already exists: {run.model}, {run.dataset_type}, {run.class_name}, {run.backbone}, {run.ad_layers}, {run.contamination}")
                continue
            print(f"Method: {run.model}")
            print(f"Dataset type: {run.dataset_type}")
            print(f"Class name: {run.class_name}")
            print(f"Backbone: {run.backbone}")
            print(f"AD layers: {run.ad_layers}")
            print(f"Contamination: {run.contamination}")
            print("-------------------------------------")
            state = "Running"
            error = ""
            df = update_dataframe(df, run, state=state, error=error)
            try:
                if run.model == 'cfa':
                    args = CFAArguments(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        category=run.class_name,
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed,
                        device=device
                    )

                    benchmark_cfa(args, df, csv_file)
                    state = "Completed"

                elif run.model == 'padim':
                    args = PadimArgs(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        category=run.class_name,
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed,
                        device=device
                    )
                    benchmark_padim(args, df, csv_file)
                    state = "Completed"
                elif run.model == 'patchcore':
                    args = PatchCoreArgs(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        category=run.class_name,
                        img_input_size=(256, 256),
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed,
                        device=device
                    )

                    benchmark_patchcore(args, df, csv_file)
                    state = "Completed"
                elif run.model == 'patchcore_quantized':
                    args = PatchCoreArgs(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        category=run.class_name,
                        img_input_size=(256, 256),
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed,
                        quantized=True,
                        device=device
                    )
                    benchmark_patchcore(args, df, csv_file)
                    state = "Completed"

                elif run.model == 'stfpm':
                    args = STFPMArgs(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        categories=[run.class_name],
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed,
                        device=device
                    )

                    benchmark_stfpm(args, df, csv_file)
                    state = "Completed"

            except DatasetTooSmallToContaminateException:
                print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                state = "Failed"
                error = "Dataset too small to contaminate"
                update_dataframe(df, run, state=state, error=error)
                save_dataframe(df, csv_file)
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                state = "Failed"
                error = str(e)
                update_dataframe(df, run, state=state, error=error)
                save_dataframe(df, csv_file)
                continue

            update_dataframe(df, run, state=state, error=error)
            save_dataframe(df, csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file",
                        type=str,
                        help="Path to the config file",
                        required=True)

    parser.add_argument("--mode",
                        choices=["generate-checklist", "run"],
                        help="Script execution mode: generate-checklist or run",
                        default="run")

    parser.add_argument("--checklist-path",  # Changed to optional argument with --
                        type=str,
                        default=None,
                        help="Path of the checklist file")

    parser.add_argument("--device",
                        type=str,
                        default=None,
                        help="Device to run the benchmark")

    args = parser.parse_args()
    benchmark_args = BenchmarkArgs.from_parser(args)
    main(benchmark_args)
