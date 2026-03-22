def test_model_create_train():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.dinomaly.dinomaly import Dinomaly, DinomalyTrainArgs
    from moviad.trainers.trainer import Trainer
    from moviad.datasets.mvtec import MVTecDataset
    from torch.utils.data import Subset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    args = DatasetArguments(
        dataset_path = "/mnt/disk1/manuel_barusco/datasets/mvtec",
        img_size = (224, 224),
        gt_mask_size = (224, 224),
        image_transform_list = None
    )

    train_dataset = MVTecDataset(args, category="bottle", split="train")
    #train_dataset = Subset(train_dataset, list(range(0, 10)))  # use a subset for faster testing

    test_dataset = MVTecDataset(args, category="bottle", split="test")

    model = Dinomaly("deit_tiny_patch16_224.fb_in1k")
    model.to(device)
    training_args = DinomalyTrainArgs(batch_size=32, epochs=5000 // len(torch.utils.data.DataLoader(train_dataset, batch_size=32)))
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=None,
        save_path=None,
        saving_criteria=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.decoder.parameters()]
    trainer.train()
    params_after = [p for p in model.decoder.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
