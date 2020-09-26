#!/bin/bash

# Evaluate Bi-Encoder and Poly-Encoder for Fashion/Furniture

GPU=0
SEED=0

#DOMAIN="fashion"
DOMAIN="furniture"
TEST_SPLIT_NAME="devtest"
#TEST_SPLIT_NAME="test-std"

BART_MODEL="bart-base"
#BART_MODEL="bart-large"

ARCHITECTURE="bi"
#ARCHITECTURE="poly"
#ARCHITECTURE="both"
if [ ${ARCHITECTURE} == "bi" ]
then
    POLY_M=0
    MODEL_LABEL="bi-encoder"
elif [ ${ARCHITECTURE} == "poly" ]
then
    POLY_M=16
    MODEL_LABEL="poly-encoder"
fi

# Directory where data is stored
TRAIN_DIR="../../data/simmc_${DOMAIN}/"
# Directory to output results
OUTPUT_DIR="../../output/${DOMAIN}/${MODEL_LABEL}/${TEST_SPLIT_NAME}/"
# Directory where pretrained model is stored
MODEL_DIR="../../model/${DOMAIN}/${BART_MODEL}_${BART_MODEL}/best_model/"
# Directory to store trained model
if [ ${ARCHITECTURE} == "bi" ]
then
    MODEL_OUT="../../model/${DOMAIN}/bi-encoder/best_model/"
elif [ ${ARCHITECTURE} == "poly" ]
then 
    MODEL_OUT="../../model/${DOMAIN}/poly-encoder/best_model/"
fi

echo "Performing evaluation for ${DOMAIN} ${TEST_SPLIT_NAME} dataset with ${BART_MODEL} and ${MODEL_LABEL}"

python3 run.py --bart_model ${BART_MODEL} --model_in ${MODEL_DIR} --model_out ${MODEL_OUT} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --domain ${DOMAIN} \
 --testset ${TEST_SPLIT_NAME} --use_pretrain --architecture ${ARCHITECTURE} --poly_m ${POLY_M} --gpu ${GPU} --set_seed --seed ${SEED} --eval 
