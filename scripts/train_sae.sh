#!/bin/bash

# hidden_dim: 1024 512 256 128
# l1: 0.01 0.05 0.1 0.2 0.3 0.4 0.5

for hidden_dim in 1024 512 256 128
do
  for l1_weight in 0.001 0.005 # 0.01 0.05 0.1 0.2 0.3 0.4 0.5
  do
    python -m src.main --hidden_dim=${hidden_dim} \
                       --tie_weights=1 \
                       --l1_weight=${l1_weight} \
                       --min_activation=1e-3 \
                       --aux_ratio=0.3 \
                       --dead_steps=128 \
                       --batch_size=32 \
                       --lr=5e-4 \
                       --epochs=50 \
                       --verbose=1 \
                       --use_full=1 \
                       --do_val=0 \
                       --do_test=0 \
                       --do_inference=1 \
                       --early_stop=1 \
                       --model_save_dir="/cluster/project/sachan/pencui/ProjectsData/skillset/output/sae_models" \
                       --feature_save_dir="/cluster/project/sachan/pencui/ProjectsData/skillset/output/sae_features"
  done
done