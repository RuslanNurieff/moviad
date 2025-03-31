import argparse
import traceback
from abc import abstractmethod

import torch
import wandb
import pandas as pd
import os

from benchmark_logger import CsvLogger
from benchmark_model_mappings import MODEL_MAPPINGS
from benchmark_config import DatasetRunConfig, BenchmarkConfig
from benchmark_args import BenchmarkArgs
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
        df.loc[existing_row.index, "Error"] = error
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
            if run_exists(df, run):  # Skip if run already exists
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


def benchmark(model_type, args, train_method):
    """Generic benchmark function that accepts any model type and training method."""
    # Handle special group name for patchcore
    group_name = model_type
    if model_type == "patchcore" and hasattr(args, "quantized") and args.quantized:
        group_name = "patchcore_quantized"

    logger = wandb.init(project="moviad_benchmark", group=model_type)

    # Get the right category attribute (some use category, others use categories)
    category = args.categories if hasattr(args, "categories") else args.category

    logger.config.update({
        "ad_model": group_name,
        "dataset": args.dataset_type,
        "category": category,
        "backbone": args.backbone,
        "ad_layers": args.ad_layers,
        "seed": args.seed,
        "contamination_ratio": args.contamination_ratio
    }, allow_val_change=True)

    logger.tags = [group_name, args.dataset_type, args.backbone]
    if args.contamination_ratio > 0:
        logger.tags += tuple(["contaminated"])

    logger.name = f"{group_name}_{args.dataset_type}_{args.backbone}"

    # Call the appropriate training method
    if model_type == "stfpm":
        train_method(args, logger, evaluate=True)
    else:
        train_method(args, logger)

    logger.finish()


def benchmark_cfa(args: CFAArguments, df, csv_file):
    benchmark("cfa", args, df, csv_file, train_cfa)

def benchmark_padim(args: PadimArgs, df, csv_file):
    benchmark("padim", args, df, csv_file, train_padim)

def benchmark_patchcore(args: PatchCoreArgs, df, csv_file):
    benchmark("patchcore", args, df, csv_file, train_patchcore)

def benchmark_stfpm(args: STFPMArgs, df, csv_file):
    benchmark("stfpm", args, df, csv_file, train_stfpm)


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
    dataset_config = DatasetConfig(benchmark_args.config_file, image_size=(224, 224))
    benchmark_config = BenchmarkConfig(benchmark_args.config_file)

    csv_file = "benchmark_checklist.csv" if benchmark_args.checklist_path is None else benchmark_args.checklist_path
    benchmark_logger = CsvLogger(csv_file)

    if not is_cuda_device_available(benchmark_args.device):
        raise RuntimeError(f"CUDA device '{benchmark_args.device}' is not available. "
                           f"Available devices: CPU and CUDA devices 0 to {torch.cuda.device_count() - 1 if torch.cuda.is_available() else 'none'}")

    device = torch.device(benchmark_args.device)
    print("DEVICE: ", device)

    if benchmark_args.mode == "generate-checklist":
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=columns)

        df = generate_full_checklist(df, benchmark_config)
        save_dataframe(df, benchmark_args.checklist_path)
        return

    for benchmark_run in benchmark_config.get_benchmark_runs():
        for run in benchmark_run.get_runs():
            if benchmark_logger.run_exists(run):
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
            benchmark_logger.update(run, state=state, error=error)

            try:
                # Get the mapping for the model
                model_key = run.model
                if model_key not in MODEL_MAPPINGS:
                    print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                    continue

                mapping = MODEL_MAPPINGS[model_key]

                # Create arguments using the constructor from the mapping
                args = mapping['arg_constructor'](dataset_config, run, seed, device)

                # Run the benchmark using the training method from the mapping
                benchmark(model_key, args, mapping['train_method'])

                state = "Completed"

            except DatasetTooSmallToContaminateException:
                print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                state = "Failed"
                error = "Dataset too small to contaminate"
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback_str = traceback.format_exc()
                print(f"Stack trace:\n{traceback_str}")
                state = "Failed"
                error = str(e)

            benchmark_logger.update(run, state=state, error=error)
            benchmark_logger.save()


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
                        required=True,
                        default=None,
                        help="Device to run the benchmark",
                        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["mps"], )

    args = parser.parse_args()
    benchmark_args = BenchmarkArgs.from_parser(args)
    main(benchmark_args)
