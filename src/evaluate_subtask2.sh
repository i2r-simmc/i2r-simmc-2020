DOMAIN=$1
ROOT=../data/simmc_$DOMAIN
MODEL_TYPE_COMBINED=bart-large_poly-encoder
TEST_SPLIT=${2:-devtest}
OUTPUT_ROOT=../output/$DOMAIN/$MODEL_TYPE_COMBINED/$TEST_SPLIT

# Evaluate for devtest or test-std dataset
if [ $TEST_SPLIT == "devtest" ]
then
  python -m mm_action_prediction.tools.response_evaluation \
      --data_json_path="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials.json \
      --model_response_path="${OUTPUT_ROOT}"/dstc9-simmc-${TEST_SPLIT}-${DOMAIN}-subtask-2-generation.json \
      --report_output_path="${OUTPUT_ROOT}"/dstc9-simmc-${TEST_SPLIT}-${DOMAIN}-subtask-2-generation-report.json
else
  python -m mm_action_prediction.tools.response_evaluation \
      --data_json_path="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials.json \
      --model_response_path="${OUTPUT_ROOT}"/dstc9-simmc-${TEST_SPLIT}-${DOMAIN}-subtask-2-generation.json \
      --report_output_path="${OUTPUT_ROOT}"/dstc9-simmc-${TEST_SPLIT}-${DOMAIN}-subtask-2-generation-report.json \
      --single_round_evaluation
fi