DOMAIN=$1
LOCAL_RANK=${2:-0}
LEARNING_RATE=${3:-1e-5}
BATCH_SIZE=${4:-2}
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN
MODEL="facebook/bart-large"
TEST_SPLIT=devtest

#python main.py \
#    --action=generate \
#    --config_file=../config/simmc_transformers_fusion_${DOMAIN}.yml \
#    --test_data_src="$ROOT"/${DOMAIN}_${TEST_SPLIT}_dials_predict.txt \
#    --test_data_tgt_subtask1="$ROOT"/simmc_${DOMAIN}_api_${TEST_SPLIT}.json \
#    --test_data_tgt_subtask2="$ROOT"/${DOMAIN}_${TEST_SPLIT}_dials_target.txt \
#    --test_data_tgt_subtask3="$ROOT"/simmc_${DOMAIN}_resp_${TEST_SPLIT}.json \
#    --encoder_decoder_model_name_or_path=$MODEL \
#    --test_output_pred="$OUTPUT_ROOT"/output.json \
#    --domain=$DOMAIN \
#    --learning_rate=$LEARNING_RATE \
#    --local_rank=$LOCAL_RANK \
#    --batch_size=$BATCH_SIZE \
#    --load_model_index=$LOCAL_RANK


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