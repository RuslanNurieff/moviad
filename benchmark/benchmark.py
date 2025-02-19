import argparse
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


def update_dataframe(df, run):
    existing_row = df[(df["Method"] == run.model) &
                      (df["Dataset type"] == run.dataset_type) &
                      (df["Class name"] == run.class_name) &
                      (df["Backbone"] == run.backbone) &
                      (df["AD layers"] == str(run.ad_layers)) &
                      (df["Contamination"] == run.contamination)]

    if not existing_row.empty:
        df.loc[existing_row.index, "Runs"] += 1
    else:
        new_row = pd.DataFrame([{
            "Method": run.model,
            "Dataset type": run.dataset_type,
            "Class name": run.class_name,
            "Backbone": run.backbone,
            "AD layers": run.ad_layers,
            "Contamination": run.contamination,
            "Runs": 1
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def save_dataframe(df, csv_file):
    df.to_csv(csv_file, index=False)

def run_exists(df, run):
    existing_row = df[(df["Method"] == run.model) &
                      (df["Dataset type"] == run.dataset_type) &
                      (df["Class name"] == run.class_name) &
                      (df["Backbone"] == run.backbone) &
                      (df["AD layers"] == str(run.ad_layers)) &
                      (df["Contamination"] == run.contamination)]
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
    def __init__(self, config_file, mode, model, dataset, category, backbone, ad_layers, epochs, save_path,
                 visual_test_path, device, seed):
        self.config_file = config_file
        self.mode = mode
        self.model = model
        self.dataset = dataset
        self.category = category
        self.backbone = backbone
        self.ad_layers = ad_layers
        self.epochs = epochs
        self.save_path = save_path
        self.visual_test_path = visual_test_path
        self.device = device
        self.seed = seed

    @classmethod
    def from_parser(cls, args):
        return cls(
            config_file=args.config_file,
            mode=args.mode,
            model=args.model,
            dataset=args.dataset,
            category=args.category,
            backbone=args.backbone,
            ad_layers=args.ad_layers,
            epochs=args.epochs,
            save_path=args.save_path,
            visual_test_path=args.visual_test_path,
            device=args.device,
            seed=args.seed
        )


def main(benchmark_args: BenchmarkArgs):
    # Parse the config file into DatasetConfig and BenchmarkConfig
    dataset_config = DatasetConfig(benchmark_args.config_file)
    benchmark_config = BenchmarkConfig(benchmark_args.config_file)

    # Initialize the DataFrame
    columns = ["Method", "Dataset type", "Class name", "Backbone", "AD layers", "Contamination", "Runs"]
    csv_file = "benchmark_checklist.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=columns)

    for benchmark_run in benchmark_config.get_benchmark_runs():
        for run in benchmark_run.get_runs():
            if run_exists(df, run):
                print(f"Run already exists: {run.model}, {run.dataset_type}, {run.class_name}, {run.backbone}, {run.ad_layers}, {run.contamination}")
                continue
            print(f"Method: {run.model}")
            print(f"Dataset type: {run.dataset_type}")
            print(f"Class name: {run.class_name}")
            print(f"Backbone: {run.backbone}")
            print(f"AD layers: {run.ad_layers}")
            print(f"Contamination: {run.contamination}")
            print("-------------------------------------")
            try:
                if run.model == 'cfa':
                    args = CFAArguments(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        category=run.class_name,
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed
                    )
                    try:
                        benchmark_cfa(args, df, csv_file)
                        df = update_dataframe(df, run)
                        save_dataframe(df, csv_file)
                    except DatasetTooSmallToContaminateException:
                        print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                        continue
                elif run.model == 'padim':
                    args = PadimArgs(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        category=run.class_name,
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed,
                    )
                    try:
                        benchmark_padim(args, df, csv_file)
                        df = update_dataframe(df, run)
                        save_dataframe(df, csv_file)
                    except DatasetTooSmallToContaminateException:
                        print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                        continue
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
                    )
                    try:
                        benchmark_patchcore(args, df, csv_file)
                        df = update_dataframe(df, run)
                        save_dataframe(df, csv_file)
                    except DatasetTooSmallToContaminateException:
                        print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                        continue
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
                        quantized=True
                    )
                    try:
                        benchmark_patchcore(args, df, csv_file)
                        df = update_dataframe(df, run)
                        save_dataframe(df, csv_file)
                    except DatasetTooSmallToContaminateException:
                        print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                        continue
                elif run.model == 'stfpm':
                    args = STFPMArgs(
                        dataset_config=dataset_config,
                        dataset_type=run.dataset_type,
                        categories=[run.class_name],
                        backbone=run.backbone,
                        ad_layers=run.ad_layers,
                        contamination_ratio=run.contamination,
                        seed=seed
                    )
                    try:
                        benchmark_stfpm(args, df, csv_file)
                        df = update_dataframe(df, run)
                        save_dataframe(df, csv_file)
                    except DatasetTooSmallToContaminateException:
                        print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                        continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, help="Path to the config file (specify only this argument)")
    parser.add_argument("--csv-file", type=str, help="Path to the checklist file")
    parser.add_argument("--mode", choices=["train", "test"], help="Script execution mode: train or test")
    parser.add_argument("--model", choices=["cfa", "padim", "patchcore", "stfpm"],
                        help="Script execution mode: train or test")
    parser.add_argument("--dataset", choices=["mvtec", "realiad", "visa"], type=str, help="Dataset type")
    parser.add_argument("--category", type=str, help="Dataset category to test")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--ad_layers", type=str, nargs="+", help="List of ad layers")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--save_path", type=str, default=None, help="Path of the .pt file where to save the model")
    parser.add_argument("--visual_test_path", type=str, default=None,
                        help="Path of the directory where to save the visual paths")
    parser.add_argument("--device", type=str, help="Where to run the script")
    parser.add_argument("--seed", type=int, default=1, help="Execution seed")
    args = parser.parse_args()
    benchmark_args = BenchmarkArgs.from_parser(args)
    main(benchmark_args)