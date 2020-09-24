# Preprocess data for both Fashion and Furniture domains 

TRAIN_DIR="../data/"

TESTSET="devtest"
#TESTSET="test-std"

# Fashion
python3 ./retrieval/parse.py --mode train --domain fashion --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode dev --domain fashion --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode ${TESTSET} --domain fashion --train_dir ${TRAIN_DIR}

if [ TESTSET == "devtest" ]
then
    mv ../data/simmc_fashion/devtest.txt ../data/simmc_fashion/test.txt
    cp ../data/simmc_fashion/fashion_devtest_dials_retrieval_candidates.json data/simmc_fashion/candidates.json
elif [ TESTSET == "test-std" ]
then
    mv ../data/simmc_fashion/test-std.txt ../data/simmc_fashion/test.txt
    cp ../data/simmc_fashion/fashion_test-std_dials_retrieval_candidates.json data/simmc_fashion/candidates.json
fi


#Furniture
python3 ./retrieval/parse.py --mode train --domain furniture --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode dev --domain furniture --train_dir ${TRAIN_DIR}
python3 ./retrieval/parse.py --mode ${TESTSET} --domain furniture --train_dir ${TRAIN_DIR}

if [ TESTSET == "devtest" ]
then
    mv ../data/simmc_furniture/devtest.txt ../data/simmc_furniture/test.txt
    cp ../data/simmc_furniture/furniture_devtest_dials_retrieval_candidates.json ../data/simmc_furniture/candidates.json
elif [ TESTSET == "test-std" ]
then
    mv ../data/simmc_furniture/test-std.txt ../data/simmc_furniture/test.txt
    cp ../data/simmc_furniture/furniture_test-std_dials_retrieval_candidates.json ../data/simmc_furniture/candidates.json
fi
