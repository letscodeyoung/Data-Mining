import pandas as pd
import pickle
from train import extract_features

# read the test data and extract features
test_data = pd.read_csv("test.csv", header=None)
test_data_features = extract_features(test_data)

# load the model
modelname = 'SVM_model.pkl'
with open(modelname, 'rb') as file:
    svm = pickle.load(file)

# predict the test data and save the results as Result.csv
result = svm.predict(test_data_features)
final_result = pd.Series(result, name=None)
final_result.to_csv("Result.csv", index=False, header=None)
