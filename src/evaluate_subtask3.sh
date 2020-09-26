DOMAIN=$1
ROOT=../data/simmc_$DOMAIN
MODEL_TYPE_COMBINED=bart-large_poly-encoder
TEST_SPLIT=${2:-devtest}
OUTPUT_ROOT=../output/$DOMAIN/$MODEL_TYPE_COMBINED/$TEST_SPLIT

# Evaluate for devtest or test-std dataset
python -m mm_dst.utils.evaluate_dst \
    --input_path_target="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials.json \
    --input_path_predicted="${OUTPUT_ROOT}"/dstc9-simmc-${TEST_SPLIT}-${DOMAIN}-subtask-3.json \
    --output_path_report="${OUTPUT_ROOT}"/dstc9-simmc-${TEST_SPLIT}-${DOMAIN}-subtask-3-report.json