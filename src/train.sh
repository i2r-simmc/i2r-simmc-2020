DOMAIN=$1
MODEL=${2:-"facebook/bart-large"}
LOCAL_RANK=${3:-0}
LEARNING_RATE=${4:-1e-5}
BATCH_SIZE=${5:-3}
ROOT=../data/simmc_$DOMAIN

python main.py \
    --action=train \
    --config_file=../config/simmc_transformers_fusion_${DOMAIN}.yml \
    --train_data_src="$ROOT"/${DOMAIN}_train_dials_predict.txt \
    --train_data_tgt_subtask1="$ROOT"/simmc_${DOMAIN}_api_train.json \
    --train_data_tgt_subtask2="$ROOT"/${DOMAIN}_train_dials_target.txt \
    --train_data_tgt_subtask3="$ROOT"/simmc_${DOMAIN}_resp_train.json \
    --dev_data_src="$ROOT"/${DOMAIN}_dev_dials_predict.txt \
    --dev_data_tgt_subtask1="$ROOT"/simmc_${DOMAIN}_api_dev.json \
    --dev_data_tgt_subtask2="$ROOT"/${DOMAIN}_dev_dials_target.txt \
    --dev_data_tgt_subtask3="$ROOT"/simmc_${DOMAIN}_resp_dev.json \
    --model_metainfo_path="$ROOT"/${DOMAIN}_model_metainfo.json \
    --encoder_decoder_model_name_or_path=$MODEL \
    --learning_rate=$LEARNING_RATE \
    --local_rank=$LOCAL_RANK \
    --batch_size=$BATCH_SIZE \
    --load_model_index=$LOCAL_RANK