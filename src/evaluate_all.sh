DOMAIN=$1
TEST_SPLIT=${2:-devtest}

bash evaluate_subtask1.sh $1 $2
bash evaluate_subtask2.sh $1 $2
bash evaluate_subtask3.sh $1 $2
python generate_report.py --domain=$1