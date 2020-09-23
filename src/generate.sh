DOMAIN=$1
LOCAL_RANK=${2:-0}
LEARNING_RATE=${3:-1e-5}
BATCH_SIZE=${4:-2}
TEST_BATCH_SIZE=${5:-100}
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN
MODEL="facebook/bart-large"
TEST_SPLIT=devtest

python main.py \
    --action=generate \
    --config_file=../config/simmc_transformers_fusion_${DOMAIN}.yml \
    --test_data_src="$ROOT"/${DOMAIN}_${TEST_SPLIT}_dials_predict.txt \
    --encoder_decoder_model_name_or_path=$MODEL \
    --model_metainfo_path="$ROOT"/${DOMAIN}_model_metainfo.json \
    --test_output_pred="$OUTPUT_ROOT"/output.json \
    --domain=$DOMAIN \
    --learning_rate=$LEARNING_RATE \
    --local_rank=$LOCAL_RANK \
    --batch_size=$BATCH_SIZE \
    --test_batch_size=$TEST_BATCH_SIZE \
    --load_model_index=$LOCAL_RANK


python main.py \
    --action=postprocess \
    --config_file=../config/simmc_transformers_fusion_${DOMAIN}.yml \
    --encoder_decoder_model_name_or_path=$MODEL \
    --test_output_pred="$OUTPUT_ROOT"/output.json \
    --test_data_output_subtask1="$OUTPUT_ROOT"/output_subtask1.json \
    --test_data_output_subtask2="$OUTPUT_ROOT"/output_subtask2.txt \
    --test_data_output_subtask3="$OUTPUT_ROOT"/output_subtask3.txt \
    --test_data_original_file="$ROOT"/${DOMAIN}_${TEST_SPLIT}_dials.json \
    --domain=$DOMAIN \
    --learning_rate=$LEARNING_RATE \
    --local_rank=$LOCAL_RANK \
    --batch_size=$BATCH_SIZE \
    --load_model_index=$LOCAL_RANK