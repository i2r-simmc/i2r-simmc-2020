DOMAIN=$1
INDEX=${2:-0}
ROOT=../data/simmc_$DOMAIN
OUTPUT_ROOT=../output/$DOMAIN

# Evaluate for devtest dataset
python -m evaluate_bleu \
    --input_path_target="${ROOT}"/${DOMAIN}_devtest_dials.json \
    --input_path_predicted="${OUTPUT_ROOT}"/output_subtask2.txt_${INDEX} \
    --output_path_report="${OUTPUT_ROOT}"/output_subtask2_report.json_${INDEX}
