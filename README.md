
# Solution to Kaggle's Quora  Question Pair
The competition can be found via the link: https://www.kaggle.com/c/quora-question-pairs
## Prerequisites
- Download pre-trained word vectors googlenews-vectors-negative300.bin which is available here
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
- Download pre-trained word vectors glove.840B.300d which is available here https://nlp.stanford.edu/projects/glove/
- Download the train and test data from https://www.kaggle.com/c/quora-question-pairs/data. Create a folder named "data" and put them in.
## Environment
- Python 3.5
- Xgboost 0.6
- lightGBM 0.2
- numpy 1.11.3
- pandas 0.19.3
- sklearn 0.18.1
- jupyter 1.0.1
## Pipeline
- This code is written in Python 3.5 and tested on a machine with Intel E5-1630 V4 processor and Nvidia GeForce GTX 1060. Keras is used with Tensorflow backend and GPU support
- Go to the [feature] and run Feature_1 Feature_2 Feature_3 Feature_4 and Generate the csv files in the feature [train] & the [test]
- Run LSTM and xgboost in [model]
- IF you got want to have a better performance,go to [feature] and run feature_engineering.py

## Performance
we got the Top4% in the competition ,The code show here may will have a better performance,SO,Try this!
