# Train Bi-Encoder and Poly-Encoder for Fashion/Furniture


DOMAIN = "fashion"
#DOMAIN = "furniture"
TRAIN_DIR = "./data/simmc_${DOMAIN}/"

TESTSET = 'devtest'
#TESTSET = 'test-std'


if [DOMAIN == "fashion"]
then
    python3 run.py --bart_model bart-base/ --output_dir ${TRAIN_DIR} --train_dir ${TRAIN_DIR} --use_pretrain --architecture bi 
    #python3 run.py --bart_model bart-base/ --output_dir data/simmc_fashion/ --train_dir data/simmc_fashion/ --domain fashion --use_pretrain --architecture bi 
    #python3 run.py --bart_model bart-base/ --output_dir data/simmc_fashion/ --train_dir data/simmc_fashion/ --domain fashion --use_pretrain --architecture poly --poly_m 16
else
    if
        python3 run.py --bart_model bart-base/ --output_dir data/simmc_furniture/ --train_dir data/simmc_furniture --domain furniture --use_pretrain --architecture bi 
        python3 run.py --bart_model bart-base/ --output_dir data/simmc_furniture/ --train_dir data/simmc_furniture --domain furniture --use_pretrain --architecture poly --poly_m 16
    fi
fi
