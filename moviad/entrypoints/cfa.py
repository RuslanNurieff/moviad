import random
import argparse
import pathlib

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.cfa.cfa import CFA
from moviad.trainers.trainer_cfa import TrainerCFA
from moviad.utilities.configurations import TaskType
from moviad.utilities.evaluator import Evaluator


def train_cfa(train_dataset: Dataset, test_dataset: Dataset, category: str, backbone: str, ad_layers: list,
                 epochs: int, save_path: str, device: torch.device):
    gamma_c = 1
    gamma_d = 1

    print(f"Training CFA for category: {category} \n")
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, drop_last=True)

    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device)

    cfa_model = CFA(feature_extractor, backbone, device)
    cfa_model.initialize_memory_bank(train_dataloader)
    cfa_model = cfa_model.to(device)

    trainer = TrainerCFA(cfa_model, backbone, feature_extractor, train_dataloader, test_dataloader, category, device)
    trainer.train(epochs)

    # save the model
    if save_path:
        torch.save(cfa_model.state_dict(), save_path)


def test_cfa(test_dataset: Dataset, category: str, backbone: str, ad_layers: list, model_checkpoint_path: str,
                device: torch.device, visual_test_path: str = None):
    gamma_c = 1
    gamma_d = 1

    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    cfa_model = CFA(feature_extractor, backbone, device)
    cfa_model.load_model(model_checkpoint_path)
    cfa_model.to(device)
    cfa_model.eval()

    evaluator = Evaluator(test_dataloader, device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(cfa_model)

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
            anomaly_maps, pred_scores = cfa_model(images.to(device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                cfa_model.save_anomaly_map(dirpath, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i], labels[i],
                                           masks[i])
