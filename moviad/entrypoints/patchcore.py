import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.common.args import Args
from moviad.common.common_utils import obsolete
from moviad.datasets.builder import DatasetFactory
from moviad.datasets.common import IadDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.entrypoints.common import load_datasets
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator

@dataclass
class PatchCoreArgs(Args):
    contamination_ratio : float = 0.0
    visual_test_path = None
    model_checkpoint_path = "./patch.pt"
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    category: str = None
    backbone: str = None
    ad_layers: list = None
    img_input_size: tuple = (224, 224)
    save_path: str = "./temp.pt"
    batch_size: int = 2
    device: torch.device = None
    quantized: bool = False
    batched: bool = False



def train_patchcore(args: PatchCoreArgs, logger=None) -> None:
    train_dataset, test_dataset = load_datasets(args.dataset_config, args.dataset_type, args.category)
    # initialize the feature extractor
    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=True)

    # define the model
    patchcore = PatchCore(args.device, input_size=args.img_input_size, feature_extractor=feature_extractor, apply_quantization=args.quantized)
    patchcore.to(args.device)
    patchcore.train()
    trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, args.device, logger)
    trainer.train()

    # save the model
    if args.save_path:
        torch.save(patchcore.state_dict(), args.save_path)

    # force garbage collector in case
    del patchcore
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

def test_patchcore(args: PatchCoreArgs, logger=None) -> None:
    dataset_factory = DatasetFactory(args.dataset_config)
    test_dataset = dataset_factory.build(args.dataset_type, Split.TEST,  args.category)
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)
    patchcore = PatchCore(args.device, input_size=args.img_input_size, feature_extractor=feature_extractor)
    patchcore.load_model(args.model_checkpoint_path)
    patchcore.to(args.device)
    patchcore.eval()

    evaluator = Evaluator(test_dataloader, args.device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(patchcore)

    print("Evaluation performances:")
    print(f"""
    img_roc: {img_roc}
    pxl_roc: {pxl_roc}
    f1_img: {f1_img}
    f1_pxl: {f1_pxl}
    img_pr: {img_pr}
    pxl_pr: {pxl_pr}
    pxl_pro: {pxl_pro}
    """)

    if logger is not None:
        logger.log({
            "img_roc": img_roc,
            "pxl_roc": pxl_roc,
            "f1_img": f1_img,
            "f1_pxl": f1_pxl,
            "img_pr": img_pr,
            "pxl_pr": pxl_pr,
            "pxl_pro": pxl_pro,
        })

    # chek for the visual test
    if args.visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(args.visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        for images, labels, masks, paths in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores = patchcore(images.to(args.device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                patchcore.save_anomaly_map(args.visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
                                           labels[i], masks[i])
