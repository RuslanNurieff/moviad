from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models import PatchCore
from moviad.models.training_args import TrainingArgs
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.patchcore_cl import PatchCoreCL
from moviad.datasets.mvtec import MVTecDataset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
import torch
import wandb

def train_patchcore_cl():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    model = PatchCore(feature_extractor=feature_extractor)

    args = {
        "dataset_path" : "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        "image_transform_list" : None
    }

    continual_dataset = ContinualDataset(
        DatasetArguments(**args),
        MVTecDataset
    )

    continual_model = PatchCoreCL(model)

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
        training_args=TrainingArgs(
            batch_size = 32, 
            epochs = 1
        ),
        logger=None
    )

    trainer.train()

train_patchcore_cl()
