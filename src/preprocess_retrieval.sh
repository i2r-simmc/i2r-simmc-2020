# Preprocess data for both Fashion and Furniture domains 

TRAIN_DIR="../data/"

TESTSET="devtest"
#TESTSET="test-std"

# Fashion
python3 ./retrieval/parse.py --mode train --domain fashion --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode dev --domain fashion --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode ${TESTSET} --domain fashion --train_dir ${TRAIN_DIR}

mv ${TRAIN_DIR}simmc_fashion/${TESTSET}.txt ${TRAIN_DIR}simmc_fashion/test.txt
cp ${TRAIN_DIR}simmc_fashion/fashion_${TESTSET}_dials_retrieval_candidates.json ${TRAIN_DIR}simmc_fashion/candidates.json


#Furniture
python3 ./retrieval/parse.py --mode train --domain furniture --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode dev --domain furniture --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode ${TESTSET} --domain furniture --train_dir ${TRAIN_DIR}

mv ${TRAIN_DIR}simmc_furniture/${TESTSET}.txt ${TRAIN_DIR}simmc_furniture/test.txt
cp ${TRAIN_DIR}simmc_furniture/furniture_${TESTSET}_dials_retrieval_candidates.json ${TRAIN_DIR}simmc_furniture/candidates.json

