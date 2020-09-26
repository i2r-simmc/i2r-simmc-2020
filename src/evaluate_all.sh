DOMAIN=$1
TEST_SPLIT=${2:-devtest}
MODEL=${3:-"facebook/bart-large"}
JOINT_MODEL_NAME="${MODEL/facebook\//}"
MODEL_TYPE_COMBINED=${JOINT_MODEL_NAME}_poly-encoder

bash evaluate_subtask1.sh ${DOMAIN} ${TEST_SPLIT} ${MODEL}
bash evaluate_subtask2.sh ${DOMAIN} ${TEST_SPLIT} ${MODEL}
bash evaluate_subtask3.sh ${DOMAIN} ${TEST_SPLIT} ${MODEL}
echo ${MODEL_TYPE_COMBINED}
python generate_report.py --domain=${DOMAIN} --test_split_name=${TEST_SPLIT} --model_type_combined=${MODEL_TYPE_COMBINED} --joint_model_name=${JOINT_MODEL_NAME}