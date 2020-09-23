# Preprocess data for both Fashion and Furniture domains 

# Fashion
python3 parse.py --mode train --domain fashion
python3 parse.py --mode dev --domain fashion
python3 parse.py --mode devtest --domain fashion

mv data/simmc_fashion/devtest.txt data/simmc_fashion/test.txt
cp data/simmc_fashion/fashion_devtest_dials_retrieval_candidates.json data/simmc_fashion/candidates.json


#Furniture
python3 parse.py --mode train --domain furniture
python3 parse.py --mode dev --domain furniture
python3 parse.py --mode devtest --domain furniture

mv data/simmc_furniture/devtest.txt data/simmc_furniture/test.txt
cp data/simmc_furniture/furniture_devtest_dials_retrieval_candidates.json data/simmc_furniture/candidates.json

