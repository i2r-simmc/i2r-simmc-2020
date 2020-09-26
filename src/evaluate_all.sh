DOMAIN=$1
TEST_SPLIT=${2:-devtest}
MODEL=${3:-"facebook/bart-large"}
JOINT_MODEL_NAME="${MODEL/facebook\//}"
RETRIEVAL_MODEL_POLY=poly-encoder
RETRIEVAL_MODEL_BI=bi-encoder
MODEL_TYPE_COMBINED_PLY_ENCODER=${JOINT_MODEL_NAME}_${RETRIEVAL_MODEL_POLY}
MODEL_TYPE_COMBINED_BI_ENCODER=${JOINT_MODEL_NAME}_${RETRIEVAL_MODEL_BI}

bash evaluate_subtask1.sh ${DOMAIN} ${TEST_SPLIT} ${MODEL}
bash evaluate_subtask2.sh ${DOMAIN} ${TEST_SPLIT} ${MODEL}
bash evaluate_subtask3.sh ${DOMAIN} ${TEST_SPLIT} ${MODEL}
echo ${MODEL_TYPE_COMBINED}
python generate_report.py --domain=${DOMAIN} --test_split_name=${TEST_SPLIT} --model_type_combined=${MODEL_TYPE_COMBINED} --joint_model_name=${JOINT_MODEL_NAME}

# Copy to bi-encoder
mkdir -p ../output/${DOMAIN}/${MODEL_TYPE_COMBINED_BI_ENCODER}/${TEST_SPLIT}
cp -rf ../output/${DOMAIN}/${MODEL_TYPE_COMBINED_PLY_ENCODER}/${TEST_SPLIT}/* ../output/${DOMAIN}/${MODEL_TYPE_COMBINED_BI_ENCODER}/${TEST_SPLIT}