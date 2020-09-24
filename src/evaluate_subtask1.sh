DOMAIN=$1
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN
TEST_SPLIT=${2:-devtest}

# Evaluate for devtest dataset
python -m mm_action_prediction.tools.action_evaluation \
    --action_json_path="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials_api_calls.json \
    --model_output_path="${OUTPUT_ROOT}"/output_subtask1.json \
    --report_output_path="${OUTPUT_ROOT}"/output_subtask1_report.json
