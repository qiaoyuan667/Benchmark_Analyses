#!/bin/bash

python -m src.sae_extract --hidden_dim=1024 \
                          --tie_weights=1 \
                          --batch_size=32 \
                          --lr=5e-4 \
                          --l1_weight=1e-1 \
                          --epochs=100 \
                          --verbose=1 \
                          --do_val=1 \
                          --do_test=1 \
                          --model_save_path="models/sae.pt" \
                          --min_activation=1e-3