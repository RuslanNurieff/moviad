import os.path
import unittest
import torch
from memory_profiler import profile
from torchvision.transforms import transforms

from moviad.datasets.builder import DatasetConfig
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.entrypoints.patchcore import PatchCoreArgs, train_patchcore
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.patchcore.product_quantizer import ProductQuantizer
from moviad.profiler.pytorch_profiler import torch_profile
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluator import Evaluator
from moviad.utilities.metrics import compute_product_quantization_efficiency

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])

CONFIG_PATH = 'config.json'


class PatchCoreTrainTests(unittest.TestCase):

    def setUp(self):
        self.args = PatchCoreArgs()
        self.config = DatasetConfig(CONFIG_PATH)
        self.args.contamination_ratio = 0.25
        self.args.batch_size = 1
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = (224, 224)
        self.args.train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            'pill',
            Split.TRAIN,
            img_size=self.args.img_input_size,
        )
        self.args.test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            'pill',
            Split.TEST,
            img_size=self.args.img_input_size,
        )
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.category
        self.contamination = 0
        self.args.backbone = "mobilenet_v2"
        self.args.ad_layers = ["features.4", "features.7", "features.10"]

    def test_patchcore_quantization_efficiency(self):
        unquantized_memory_bank = torch.rand([30000, 160], dtype=torch.float32)

        pq = ProductQuantizer()
        pq.fit(unquantized_memory_bank)

        quantized_memory_bank = pq.encode(unquantized_memory_bank)

        quantization_efficiency, distortion = compute_product_quantization_efficiency(
            unquantized_memory_bank.cpu().numpy(),
            quantized_memory_bank.cpu().numpy(),
            pq)

        self.assertGreater(quantization_efficiency, 0)
        self.assertGreater(distortion, 0)

    def test_patchcore_with_quantization(self):
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)
        train_dataloader = torch.utils.data.DataLoader(self.args.train_dataset, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=True)

        trainer = TrainerPatchCore(patchcore_model, train_dataloader, test_dataloader, self.args.device)
        trainer.train()

        patchcore_model.save_model("./")

    @torch_profile
    def test_patchcore_with_quantization_and_load(self):
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)

        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=True)

        patchcore_model.load("./patchcore_model.pt", "./product_quantizer.bin")

        evaluator = Evaluator(test_dataloader, self.args.device)
        img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(patchcore_model)

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

    @profile
    def test_patchcore_without_quantization(self):
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                   False, None)
        train_dataloader = torch.utils.data.DataLoader(self.args.train_dataset, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=False)
        trainer = TrainerPatchCore(patchcore_model, train_dataloader, test_dataloader, self.args.device)
        trainer.train()

        quantized_memory_bank = patchcore_model.memory_bank
        patchcore_model.save_model("./")

        model_memory_size = os.path.getsize("./patchcore_model.pt")
        print(f"Model memory size: {model_memory_size}")


if __name__ == '__main__':
    unittest.main()
