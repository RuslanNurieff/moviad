# At the top of the file, define a mapping from model names to training methods and argument constructors
from moviad.entrypoints.cfa import train_cfa, CFAArguments
from moviad.entrypoints.padim import train_padim, PadimArgs
from moviad.entrypoints.patchcore import train_patchcore, PatchCoreArgs
from moviad.entrypoints.stfpm import train_stfpm, STFPMArgs

MODEL_MAPPINGS = {
    'cfa': {
        'train_method': train_cfa,
        'arg_constructor': lambda dataset_config, run, seed, device: CFAArguments(
            dataset_config=dataset_config,
            dataset_type=run.dataset_type,
            category=run.class_name,
            backbone=run.backbone,
            ad_layers=run.ad_layers,
            contamination_ratio=run.contamination,
            seed=seed,
            device=device
        )
    },
    'padim': {
        'train_method': train_padim,
        'arg_constructor': lambda dataset_config, run, seed, device: PadimArgs(
            dataset_config=dataset_config,
            dataset_type=run.dataset_type,
            category=run.class_name,
            backbone=run.backbone,
            ad_layers=run.ad_layers,
            contamination_ratio=run.contamination,
            seed=seed,
            device=device
        )
    },
    'patchcore': {
        'train_method': train_patchcore,
        'arg_constructor': lambda dataset_config, run, seed, device: PatchCoreArgs(
            dataset_config=dataset_config,
            dataset_type=run.dataset_type,
            category=run.class_name,
            img_input_size=(224, 224),
            backbone=run.backbone,
            ad_layers=run.ad_layers,
            contamination_ratio=run.contamination,
            seed=seed,
            device=device
        )
    },
    'patchcore_quantized': {
        'train_method': train_patchcore,
        'arg_constructor': lambda dataset_config, run, seed, device: PatchCoreArgs(
            dataset_config=dataset_config,
            dataset_type=run.dataset_type,
            category=run.class_name,
            img_input_size=(224, 224),
            backbone=run.backbone,
            ad_layers=run.ad_layers,
            contamination_ratio=run.contamination,
            seed=seed,
            quantized=True,
            device=device
        )
    },
    'batchedcore': {
        'train_method': train_patchcore,
        'arg_constructor': lambda dataset_config, run, seed, device: PatchCoreArgs(
            dataset_config=dataset_config,
            dataset_type=run.dataset_type,
            category=run.class_name,
            img_input_size=(224, 224),
            backbone=run.backbone,
            ad_layers=run.ad_layers,
            contamination_ratio=run.contamination,
            seed=seed,
            batched=True,
            device=device
        )
    },
    'stfpm': {
        'train_method': train_stfpm,
        'arg_constructor': lambda dataset_config, run, seed, device: STFPMArgs(
            dataset_config=dataset_config,
            dataset_type=run.dataset_type,
            categories=[run.class_name],
            backbone=run.backbone,
            ad_layers=run.ad_layers,
            contamination_ratio=run.contamination,
            seed=seed,
            device=device
        )
    }
}