# Train Bi-Encoder and Poly-Encoder for Fashion/Furniture


DOMAIN="fashion"
#DOMAIN="furniture"
TRAIN_DIR="./data/simmc_${DOMAIN}/"

ARCHITECTURE="bi"
#ARCHITECTURE="poly"
POLY_M=16

TESTSET="devtest"
#TESTSET="test-std"

# Run training for bi-encoder and poly-encoder
python3 run.py --bart_model bart-base/ --output_dir ${TRAIN_DIR} --train_dir ${TRAIN_DIR} --use_pretrain --architecture bi 
python3 run.py --bart_model bart-base/ --output_dir ${TRAIN_DIR} --train_dir ${TRAIN_DIR} --use_pretrain --architecture poly --poly_m ${POLY_M}