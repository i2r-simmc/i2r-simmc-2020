DOMAIN=$1
INDEX=${2:-0}
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN

# Evaluate for devtest dataset
python -m mm_action_prediction.tools.response_evaluation \
    --data_json_path="${ROOT}"/${DOMAIN}_devtest_dials.json \
    --model_response_path="${OUTPUT_ROOT}"/output_subtask2.json_${INDEX} \
