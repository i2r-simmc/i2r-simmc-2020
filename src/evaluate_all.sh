DOMAIN=$1
TEST_SPLIT=${2:-devtest}
MODEL_TYPE_COMBINED=bart-large_poly-encoder

bash evaluate_subtask1.sh ${DOMAIN} ${TEST_SPLIT}
bash evaluate_subtask2.sh ${DOMAIN} ${TEST_SPLIT}
bash evaluate_subtask3.sh ${DOMAIN} ${TEST_SPLIT}
echo ${MODEL_TYPE_COMBINED}
python generate_report.py --domain=${DOMAIN} --test_split_name=${TEST_SPLIT} --model_type_combined=${MODEL_TYPE_COMBINED}