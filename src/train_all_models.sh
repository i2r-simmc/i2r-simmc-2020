# Train Bi-Encoder and Poly-Encoder for Fashion/Furniture


DOMAIN="fashion"
#DOMAIN="furniture"
TRAIN_DIR="./data/simmc_${DOMAIN}/"
OUTPUT_DIR="../output/${DOMAIN}/"

#ARCHITECTURE="bi"
#ARCHITECTURE="poly"
POLY_M=16

# Run training for bi-encoder and poly-encoder
python3 run.py --bart_model bart-base/ --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --domain ${DOMAIN} --use_pretrain --architecture bi 
python3 run.py --bart_model bart-base/ --output_dir ${OUTPUT_DIR} --train_dir ${TRAIN_DIR} --domain ${DOMAIN} --use_pretrain --architecture poly --poly_m ${POLY_M}
