DOMAIN=$1
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN
TEST_SPLIT=${2:-devtest}

# Evaluate for devtest dataset
python -m mm_dst.utils.evaluate_dst \
    --input_path_target="${ROOT}"/${DOMAIN}_${TEST_SPLIT}_dials.json \
    --input_path_predicted="${OUTPUT_ROOT}"/output_subtask3.json \
    --output_path_report="${OUTPUT_ROOT}"/output_subtask3_report.json
