DOMAIN=$1
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN
TEST_SPLIT=${2:-devtest}

# Evaluate for devtest dataset
python -m mm_action_prediction.tools.response_evaluation \
    --data_json_path="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials.json \
    --model_response_path="${OUTPUT_ROOT}"/output_subtask2.json \
    --single_round_evaluation
