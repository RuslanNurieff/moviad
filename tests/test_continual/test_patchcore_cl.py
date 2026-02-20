def test_patchcore_fine_tuning():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models import PatchCore
    from moviad.models.training_args import TrainingArgs
    from moviad.scenarios.continual.continual_trainer import ContinualTrainer
    from moviad.scenarios.continual.continual_dataset import ContinualDataset
    from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch
    import wandb

    wandb.init(project="moviad_test", name="stfpm_continual_fine_tuning")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], frozen=True)    
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

    continual_model = FineTuning(model)

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
        logger=wandb
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))

def test_patchcore_multi_task():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models import PatchCore
    from moviad.models.training_args import TrainingArgs
    from moviad.trainers.trainer import Trainer
    from moviad.scenarios.continual.continual_trainer import ContinualTrainer
    from moviad.scenarios.continual.continual_dataset import ContinualDataset
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch
    import wandb

    wandb.init(project="moviad_test", name="stfpm_continual_fine_tuning")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], frozen=True)    
    model = PatchCore(feature_extractor=feature_extractor)

    args = {
        "dataset_path" : "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        "image_transform_list" : None
    }

    train_dataset, test_dataset = ContinualDataset(
        DatasetArguments(**args),
        MVTecDataset
    ).get_all_tasks_data()

    trainer = Trainer(
        training_args = TrainingArgs(
            batch_size = 32, 
            epochs = 1
        ),
        model = model, 
        train_dataset=train_dataset, 
        eval_dataset=test_dataset, 
        device=device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ]
    )

    trainer.train()

def test_patchcore_continual():
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
        logger=wandb
    )

    trainer.train()
