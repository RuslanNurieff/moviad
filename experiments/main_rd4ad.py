from moviad.models.stfpm.stfpm import STFPMTrainArgs
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models import RD4AD
from moviad.models.rd4ad.rd4ad import RD4ADTrainArgs
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.datasets.mvtec import MVTecDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
import torch
import wandb
import random
import numpy as np

def seet_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)

def train_rd4ad_replay():

    SEEDS = [1,3,1024]
    SIZES = [40, 300]

    for seed  in SEEDS:

        for size in SIZES:

            seet_seed(seed)

            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            model = RD4AD(backbone_name = "wide_resnet50_2").to(device)

            args = {
                "dataset_path" : "/home/u0052/disk/datasets/mvtec",
                "img_size" : (224, 224),
                "gt_mask_size" : (224, 224),
                "image_transform_list" : None
            }

            continual_dataset = ContinualDataset(
                DatasetArguments(**args),
                MVTecDataset
            )

            continual_model = Replay(model, memory_size=size, replay_ratio=0.5)

            training_args = RD4ADTrainArgs(epochs=30, batch_size=32, evaluation_epoch_interval=40)
            training_args.init_train(model)

            wandb.init(
                project="stfpm_adapters",
                name="rd4ad_replay",
                config={
                    "training_args": training_args.__dict__,
                    "model_name": "wide_resnet50_2",
                    "memory_size": size,
                    "replay_ratio": 0.5,
                    "seed": seed
                },
            )


            trainer = ContinualTrainer(
                continual_dataset,
                continual_model,
                device,
                metrics=[
                    RocAuc(MetricLvl.IMAGE),
                    RocAuc(MetricLvl.PIXEL),
                    AvgPrec(MetricLvl.IMAGE),
                    AvgPrec(MetricLvl.PIXEL),
                    F1(MetricLvl.IMAGE),
                    F1(MetricLvl.PIXEL),
                    ProAuc(MetricLvl.PIXEL),
                ],
                training_args=training_args,
                logger=wandb
            )

            trainer.train()

train_rd4ad_replay()
