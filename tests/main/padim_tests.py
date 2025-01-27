import unittest
import unittest

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, InterpolationMode

from moviad.datasets.builder import DatasetConfig
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.entrypoints.padim import PadimArgs
from moviad.models.padim.padim import Padim
from moviad.trainers.trainer_padim import PadimTrainer
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator
from tests.datasets.realiaddataset_tests import IMAGE_SIZE



class PadimTrainTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig("./config.yaml")
        self.args = PadimArgs()
        self.args.model_checkpoint_path = '.'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ['features.4', 'features.7', 'features.10']
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.Resize(
                IMAGE_SIZE,
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),

        ])

    def test_padim_with_diagonalization(self):
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            'pill',
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            'pill',
            Split.TEST,
            img_size=IMAGE_SIZE,
            gt_mask_size=IMAGE_SIZE,
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()

        padim = Padim(
            self.args.backbone,
            self.args.category,
            device=self.args.device,
            diag_cov=True,
            layers_idxs=self.args.ad_layers,
        )
        padim.to(self.args.device)
        trainer = PadimTrainer(
            model=padim,
            device=self.args.device,
            save_path=self.args.model_checkpoint_save_path,
            data_path=None,
            class_name=self.args.category,
            apply_diagonalization=True
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, pin_memory=True, drop_last=True
        )

        trainer.train(train_dataloader, None)

        # evaluate the model
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )

        evaluator = Evaluator(test_dataloader=test_dataloader, device=self.args.device)

        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)


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

    def test_padim_without_diagonalization(self):
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            'pill',
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            'pill',
            Split.TEST,
            img_size=IMAGE_SIZE,
            gt_mask_size=IMAGE_SIZE,
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()

        padim = Padim(
            self.args.backbone,
            self.args.category,
            device=self.args.device,
            diag_cov=self.args.diagonal_convergence,
            layers_idxs=self.args.ad_layers,
        )
        padim.to(self.args.device)
        trainer = PadimTrainer(
            model=padim,
            device=self.args.device,
            save_path=self.args.model_checkpoint_save_path,
            data_path=None,
            class_name=self.args.category,
            apply_diagonalization=False
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, pin_memory=True, drop_last=True
        )

        trainer.train(train_dataloader, None)

        # evaluate the model
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True
        )

        evaluator = Evaluator(test_dataloader=test_dataloader, device=self.args.device)

        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)


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




if __name__ == '__main__':
    unittest.main()