# Preprocess data for both Fashion and Furniture domains 

# Fashion
python3 data/simmc_fashion/parse.py --mode train
python3 data/simmc_fashion/parse.py --mode dev
python3 data/simmc_fashion/parse.py --mode devtest

mv data/simmc_fashion/devtest.txt data/simmc_fashion/test.txt
cp data/simmc_fashion/fashion_devtest_dials_retrieval_candidates.json data/simmc_fashion/candidates.json


#Furniture
python3 data/simmc_furniture/parse.py --mode train
python3 data/simmc_furniture/parse.py --mode dev
python3 data/simmc_furniture/parse.py --mode devtest

mv data/simmc_furniture/devtest.txt data/simmc_furniture/test.txt
cp data/simmc_furniture/furniture_devtest_dials_retrieval_candidates.json data/simmc_furniture/candidates.json

