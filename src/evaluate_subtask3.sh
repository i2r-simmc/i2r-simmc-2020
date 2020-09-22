DOMAIN=$1
INDEX=${2:-0}
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN

# Evaluate for devtest dataset
python -m mm_dst.utils.evaluate_dst \
    --input_path_target="${ROOT}"/${DOMAIN}_devtest_dials.json \
    --input_path_predicted="${OUTPUT_ROOT}"/output_subtask3.json_${INDEX} \
    --output_path_report="${OUTPUT_ROOT}"/output_subtask3_report.json_${INDEX}
