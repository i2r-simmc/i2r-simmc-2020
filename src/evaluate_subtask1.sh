DOMAIN=$1
INDEX=${2:-0}
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN
TEST_SPLIT=devtest

# Evaluate for devtest dataset
python -m mm_action_prediction.tools.action_evaluation \
    --input_path_target="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials_api_calls.json \
    --input_path_predicted="${OUTPUT_ROOT}"/output_subtask1.json_${INDEX} \
    --output_path_report="${OUTPUT_ROOT}"/output_subtask1_report.json_${INDEX}
