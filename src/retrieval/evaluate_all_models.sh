#!/bin/bash

# Evaluate Bi-Encoder and Poly-Encoder for Fashion/Furniture

DOMAIN="fashion"
#DOMAIN="furniture"
# Directory where data is stored
TRAIN_DIR="../../data/simmc_${DOMAIN}/"
# Directory to output results
OUTPUT_DIR="../../output/${DOMAIN}/"

TESTSET="devtest"
#TESTSET='test-std'

# Directory where model is stored
MODEL_DIR="../../model/${DOMAIN}/best_model_fusion/"

#ARCHITECTURE="bi"
#ARCHITECTURE="poly"
POLY_M=16

echo "Performing evaluation for ${DOMAIN} dataset"

python3 run.py --domain ${DOMAIN} --bart_model ${MODEL_DIR} --model_out ${MODEL_DIR} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --testset ${TESTSET} --use_pretrain --architecture bi --eval 
python3 run.py --domain ${DOMAIN} --bart_model ${MODEL_DIR} --model_out ${MODEL_DIR} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --testset ${TESTSET} --use_pretrain --architecture poly --poly_m ${POLY_M} --eval
