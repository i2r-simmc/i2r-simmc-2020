# Train Bi-Encoder and Poly-Encoder for Fashion/Furniture

python3 run.py --bart_model bart-base/ --output_dir data/simmc_fashion/ train_dir data/simmc_fashion/ --use_pretrain --architecture bi 
python3 run.py --bart_model bart-base/ --output_dir data/simmc_fashion/ train_dir data/simmc_fashion/ --use_pretrain --architecture poly --poly-m 16


python3 run.py --bart_model bart-base/ --output_dir data/simmc_furniture/ train_dir data/simmc_furniture --use_pretrain --architecture bi 
python3 run.py --bart_model bart-base/ --output_dir data/simmc_furniture/ train_dir data/simmc_furniture --use_pretrain --architecture poly --poly-m 16
