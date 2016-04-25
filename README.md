# ml-brute

ML pipeline for parameter tuning of xgboost.

To use, follow these steps:

1.  add test.csv and train.csv to data/
2.  python data/split_data.py
3.  python playg/feat_improve.py 
4.  python playg/feat_dup.py
5.  python models/xgb_super.py
6.  python score_models.py

The submission should be in submissions/ folder
