import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from moviad.common.common_utils import obsolete
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClass
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator

REAL_IAD_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\Real-IAD\\realiad_256'
AUDIO_JACK_DATASET_JSON = 'E:/VisualAnomalyDetection/datasets/Real-IAD/realiad_jsons/audiojack.json'
IMAGE_SIZE = (224, 224)


def train_patchcore(train_dataset: Dataset, test_dataset: Dataset, category: str, backbone: str, ad_layers: list,
                    save_path: str,
                    device: torch.device):
    # initialize the feature extractor
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    print(f"Training Pathcore for category: {category} \n")
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

    # define the model
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor)
    patchcore.to(device)
    patchcore.train()

    trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, device)
    trainer.train()

    # save the model
    if save_path:
        torch.save(patchcore.state_dict(), save_path)

    # force garbage collector in case
    del patchcore
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()


def test_patchcore(test_dataset: Dataset, category: str, backbone: str, ad_layers: list, model_checkpoint_path: str,
                   device: torch.device, visual_test_path: str = None):
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor)
    patchcore.load_model(model_checkpoint_path)
    patchcore.to(device)
    patchcore.eval()

    evaluator = Evaluator(test_dataloader, device)
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

    # chek for the visual test
    if visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        for images, labels, masks, paths in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores = patchcore(images.to(device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                patchcore.save_anomaly_map(visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
                                           labels[i], masks[i])