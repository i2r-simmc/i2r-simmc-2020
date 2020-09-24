# Train Bi-Encoder and Poly-Encoder for Fashion/Furniture

# Select domain as appropriate
DOMAIN="fashion"
#DOMAIN="furniture"
TRAIN_DIR="../../data/simmc_${DOMAIN}/"
OUTPUT_DIR="../../output/${DOMAIN}/"
MODEL_DIR="../../model/${DOMAIN}/best_model_fusion/"

#ARCHITECTURE="bi"
#ARCHITECTURE="poly"
POLY_M=16

echo "Performing training for ${DOMAIN} dataset"

# Run training for bi-encoder and poly-encoder
#python3 run.py --bart_model ${MODEL_DIR} --model_out ${MODEL_DIR} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --domain ${DOMAIN} --use_pretrain --architecture bi 
python3 run.py --bart_model ${MODEL_DIR} --model_out ${MODEL_DIR} --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --domain ${DOMAIN} --use_pretrain --architecture poly --poly_m ${POLY_M}
