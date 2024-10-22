#!/bin/bash

# backbones: 
# - wide_resnet50_2 ["layer1", "layer2","layer3"]
# - mobilenet_v2 [4,7,10] [7,10,13] [10,13,16]
# - phinet_1.2_0.5_6_downsampling [4,5,6] [5,6,7] [6,7,8]
# - mcunet-in3 [3,6,9] [6,9,12] [9,12,15]
# - micronet-m1 [1,2,3] [2,3,4] [3,4,5]

python main_stfpm.py --train --eval \
    --model_name mobilenet_v2 \
    --categories bottle \
    --ad_layers 3 4 5 \
    --boot_layer 2 \
    --results_dirpath debug_outputs/metrics \
    --checkpoint_dir debug_outputs/checkpoints \
    --seeds 0 --epochs 3 --input_size 224 224 \
    --device cuda:1
