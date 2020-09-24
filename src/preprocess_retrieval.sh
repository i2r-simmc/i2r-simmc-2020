# Preprocess data for both Fashion and Furniture domains 

TESTSET="devtest"
#TESTSET="test-std"

# Fashion
python3 ./retrieval/parse.py --mode train --domain fashion
python3 ./retrieval/parse.py --mode dev --domain fashion
python3 ./retrieval/parse.py --mode ${TESTSET} --domain fashion

if [ TESTSET == "devtest" ]
then
    mv ../data/simmc_fashion/devtest.txt data/simmc_fashion/test.txt
    cp ../data/simmc_fashion/fashion_devtest_dials_retrieval_candidates.json data/simmc_fashion/candidates.json
elif [ TESTSET == "test-std" ]
then
    mv ../data/simmc_fashion/test-std.txt data/simmc_fashion/test.txt
    cp ../data/simmc_fashion/fashion_test-std_dials_retrieval_candidates.json data/simmc_fashion/candidates.json
fi


#Furniture
python3 ./retrieval/parse.py --mode train --domain furniture
python3 ./retrieval/parse.py --mode dev --domain furniture
python3 ./retrieval/parse.py --mode ${TESTSET} --domain furniture

if [ TESTSET == "devtest" ]
then
    mv ../data/simmc_furniture/devtest.txt data/simmc_furniture/test.txt
    cp ../data/simmc_furniture/furniture_devtest_dials_retrieval_candidates.json data/simmc_furniture/candidates.json
elif [ TESTSET == "test-std" ]
then
    mv ../data/simmc_furniture/test-std.txt data/simmc_furniture/test.txt
    cp ../data/simmc_furniture/furniture_test-std_dials_retrieval_candidates.json data/simmc_furniture/candidates.json
fi
