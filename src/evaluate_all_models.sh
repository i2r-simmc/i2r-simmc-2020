#!/bin/bash

# Evaluate Bi-Encoder and Poly-Encoder for Fashion/Furniture

python3 run.py --bart_model bart-base/ --output_dir data/simmc_fashion/ --train_dir data/simmc_fashion/ --use_pretrain --architecture bi --eval 
python3 run.py --bart_model bart-base/ --output_dir data/simmc_fashion/ --train_dir data/simmc_fashion/ --use_pretrain --architecture poly --poly_m 16 --eval


python3 run.py --bart_model bart-base/ --output_dir data/simmc_furniture/ --train_dir data/simmc_furniture --use_pretrain --architecture bi --eval
python3 run.py --bart_model bart-base/ --output_dir data/simmc_furniture/ --train_dir data/simmc_furniture --use_pretrain --architecture poly --poly_m 16 --eval