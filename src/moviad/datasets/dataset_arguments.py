from dataclasses import dataclass
from typing import Optional, Union
from moviad.utilities.configurations import Split

@dataclass
class DatasetArguments:
    dataset_path: str
    img_size: Optional[Union[tuple, list]] = None
    gt_mask_size: Optional[Union[tuple, list]] = None
    image_transform_list: Optional[list] = None
