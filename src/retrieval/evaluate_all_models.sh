#!/bin/bash

# Evaluate Bi-Encoder and Poly-Encoder for Fashion/Furniture

#DOMAIN="fashion"
DOMAIN="furniture"
TRAIN_DIR="../../data/simmc_${DOMAIN}/"
OUTPUT_DIR="../../output/${DOMAIN}/"
MODEL_DIR="../../model/${DOMAIN}/best_model_fusion"

#ARCHITECTURE="bi"
#ARCHITECTURE="poly"
POLY_M=16

echo "Performing evaluation for ${DOMAIN} dataset"

python3 run.py --bart_model ${MODEL_DIR} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --use_pretrain --architecture bi --eval 
#python3 run.py --bart_model ${MODEL_DIR} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --use_pretrain --architecture poly --poly_m ${POLY_M} --eval

