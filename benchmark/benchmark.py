import argparse
import os

import wandb

from benchmark_config import DatasetRunConfig, BenchmarkConfig
from moviad.datasets.builder import DatasetType, DatasetConfig
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.entrypoints.cfa import CFAArguments, train_cfa
from moviad.entrypoints.padim import train_padim, PadimArgs
from moviad.entrypoints.patchcore import PatchCoreArgs, train_patchcore
from moviad.entrypoints.stfpm import STFPMArgs, train_stfpm

seed = 42


def benchmark_cfa(args: CFAArguments):
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

def benchmark_padim(args: PadimArgs):
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

def benchmark_patchcore(args: PatchCoreArgs):
    logger = wandb.init(project="moviad_benchmark", group="patchcore")
    logger.config.update({
        "ad_model": "patchcore",
        "dataset": DatasetType.MVTec,
        "category": args.category,
        "backbone": args.backbone,
        "ad_layers": args.ad_layers,
        "seed": args.seed,
        "contamination_ratio": args.contamination_ratio
    }, allow_val_change=True)
    logger.tags = ["patchcore", args.dataset_type, args.backbone]
    if args.contamination_ratio > 0:
        logger.tags += tuple(["contaminated"])
    logger.name = f"patchcore_{args.dataset_type}_{args.backbone}"
    train_patchcore(args, logger)
    logger.finish()

def benchmark_stfpm(args: STFPMArgs):
    logger = wandb.init(project="moviad_benchmark", group="stfpm")
    logger.config.update({
        "ad_model": "stfpm",
        "dataset": DatasetType.MVTec,
        "category": args.category,
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


def main(config_file: str):
    # Parse the config file into DatasetConfig and BenchmarkConfig
    dataset_config = DatasetConfig(config_file)
    benchmark_config = BenchmarkConfig(config_file)

    # Iterate through all the BenchmarkRun instances in the BenchmarkConfig
    for benchmark_run in benchmark_config.get_benchmark_runs():
        for run in benchmark_run.get_runs():
            print(f"Method: {run.model}")
            print(f"Dataset type: {run.dataset_type}")
            print(f"Class name: {run.class_name}")
            print(f"Backbone: {run.backbone}")
            print(f"AD layers: {run.ad_layers}")
            print(f"Contamination: {run.contamination}")
            print("-------------------------------------")

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
                    benchmark_cfa(args)
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
                    seed=seed
                )
                try:
                    benchmark_padim(args)
                except DatasetTooSmallToContaminateException:
                    print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                    continue
            elif run.model == 'patchcore':
                args = PatchCoreArgs(
                    dataset_config=dataset_config,
                    dataset_type=run.dataset_type,
                    category=run.class_name,
                    backbone=run.backbone,
                    ad_layers=run.ad_layers,
                    contamination_ratio=run.contamination,
                    seed=seed
                )
                try:
                    benchmark_patchcore(args)
                except DatasetTooSmallToContaminateException:
                    print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                    continue

            elif run.model == 'stfpm':
                args = STFPMArgs(
                    dataset_config=dataset_config,
                    dataset_type=run.dataset_type,
                    category=run.class_name,
                    backbone=run.backbone,
                    ad_layers=run.ad_layers,
                    contamination_ratio=run.contamination,
                    seed=seed
                )
                try:
                    benchmark_stfpm(args)
                except DatasetTooSmallToContaminateException:
                    print(f"Dataset {run.dataset_type} is too small to contaminate. Skipping...")
                    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, help="Path to the config file")

    args = parser.parse_args()
    main(args.config_file)
