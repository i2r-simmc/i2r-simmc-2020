DOMAIN=$1
LOCAL_RANK=${2:-0}
LEARNING_RATE=${3:-1e-5}
BATCH_SIZE=${4:-2}
ROOT=../data/simmc_$DOMAIN
MODEL="facebook/bart-large"

python main.py \
    --action=generate \
    --config_file=../config/simmc_transformers_fusion_${DOMAIN}.yml \
    --test_data_src="$ROOT"/${DOMAIN}_dev_dials_predict.txt \
    --test_data_tgt_subtask1="$ROOT"/simmc_${DOMAIN}_api_dev.json \
    --test_data_tgt_subtask2="$ROOT"/${DOMAIN}_dev_dials_target.txt \
    --test_data_tgt_subtask3="$ROOT"/simmc_${DOMAIN}_resp_dev.json \
    --encoder_decoder_model_name_or_path=$MODEL \
    --learning_rate=$LEARNING_RATE \
    --local_rank=$LOCAL_RANK \
    --batch_size=$BATCH_SIZE \
    --load_model_index=$LOCAL_RANK