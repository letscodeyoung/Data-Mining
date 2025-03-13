# step1: extract meal data from InsulinData.csv and CGMData.csv.
# step2: extract no meal data from InsulinData.csv and CGMData.csv.
# step3: extract features for the model: ftt, difference, entropy, etc.
# step4: train model using SVM and using k-fold cross-validation on the training data to evaluate the system.

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

#######################
# Extract meal data #
#######################

# Read data from InsulinData.csv, combine 'Timestamp' as one column, and filter data.


def read_insulin(df):
    insulin_df = df[['Date', 'Time', 'BWZ Carb Input (grams)']].copy()
    insulin_df["Timestamp"] = pd.to_datetime(
        insulin_df['Date'] + ' ' + insulin_df['Time'])
    insulin_df.set_index("Timestamp", inplace=True)

    # filter data which the BWZ Carb Input (grams) is not None or not zero.
    insulin_df['BWZ Carb Input (grams)'] = insulin_df['BWZ Carb Input (grams)'].replace(
        0, np.NaN)
    insulin_df.dropna(inplace=True)
    insulin_df.sort_values("Timestamp", inplace=True)
    return insulin_df

# Read data from CGMData.csv and combine 'Timestamp' as one column.


def read_cgm(df):
    cgm_df = df[['Date', 'Time', 'Sensor Glucose (mg/dL)']].copy()
    cgm_df['Timestamp'] = pd.to_datetime(cgm_df['Date'] + ' ' + cgm_df['Time'])
    cgm_df.set_index("Timestamp", inplace=True)
    cgm_df.sort_values("Timestamp", inplace=True)
    return cgm_df


# load data
insulin_df_1 = pd.read_csv("InsulinData.csv", low_memory=False)
cgm_df_1 = pd.read_csv("CGMData.csv", low_memory=False)
insulin_df_2 = pd.read_csv("Insulin_patient2.csv", low_memory=False)
cgm_df_2 = pd.read_csv("CGM_patient2.csv", low_memory=False)


# Extract meal time from InsulinData.csv which can satisfy the three conditions
def meal_time(df):
    insulin_df = read_insulin(df)
    # define start time as the time - 0.5h
    insulin_df['start_meal_time'] = insulin_df.index.shift(-0.5, freq='H')
    # define end time as the time + 2h
    insulin_df['end_meal_time'] = insulin_df.index.shift(2, freq='H')
    # if there is a meal, we set the label 'meal' it to 1
    insulin_df['Meal'] = 1
    # i means every row, idx means every column
    for i, idx in enumerate(insulin_df.index):
        if i < len(insulin_df) - 1:
            # Now, the columns should be ['Timestamp', 'BWZ Carb Input (grams)', 'start_meal_time', 'end_meal_time', 'meal']
            # if the next meal time less than the previous end time, means there is a meal between the start time and end time.
            if insulin_df.iloc[i, 4] >= insulin_df.index[i + 1]:
                # we should drop the current meal time and only consider the next meal time.
                insulin_df.iloc[i, 5] = np.NaN
    insulin_df.dropna(inplace=True)

    insulin_df.set_index("start_meal_time", inplace=True)
    return insulin_df.index.tolist()

# Extract meal time and glucose level from CGMData.csv and format them in a P*30 matrix.


def extract_meal_data(df, start_meal_time):
    """Take a CGM DataFrame and start time list, return a DataFrame containing p * 30 meal data"""
    cgm_df = read_cgm(df)
    meal_df_column = np.array(range(0, 150, 5))  # set an empty table
    meal_df = pd.DataFrame(columns=meal_df_column)
    for i in range(1, len(start_meal_time)):
        # meal time should larger than the start_time and less than it + 2.5 hours
        meal = cgm_df.loc[
            (cgm_df.index >= start_meal_time[i]) & (cgm_df.index <= start_meal_time[i] + pd.Timedelta('2.5 hours'))]
        # read the glucose level
        glucose = meal['Sensor Glucose (mg/dL)'].values
        # since 150/5=30, so when add to 30 columns, we should change to the new row
        if len(glucose) != 30:
            continue
        else:
            meal_df.loc[len(meal_df)] = glucose
    meal_df = meal_df.dropna(axis=0)
    return meal_df


# extract meal data
meal_time_1 = meal_time(insulin_df_1)
meal_df_1 = extract_meal_data(cgm_df_1, meal_time_1)
meal_time_2 = meal_time(insulin_df_2)
meal_df_2 = extract_meal_data(cgm_df_2, meal_time_2)
meal_data = pd.concat([meal_df_1, meal_df_2], ignore_index=True)


# Extract no meal data
# No meal time means that there is no meal between the previous end meal time and the next start meal time, and longer than 2 hours.
def no_meal_time(df):
    """take a insulin dataframe, return a list of (nomeal start time, nomeal end time),
    which the nomeal time > 2 hours """
    insulin_df = read_insulin(df)
    insulin_df["start_meal_time"] = insulin_df.index.shift(-0.5, freq='H')
    insulin_df["end_meal_time"] = insulin_df.index.shift(2, freq='H')
    no_meal = []
    for i in range(len(insulin_df) - 1):
        if (insulin_df['start_meal_time'][i + 1] - insulin_df['end_meal_time'][i]).total_seconds() <= 2 * 3600:
            continue  # expel the time period less than 2 hours
        no_meal.append((insulin_df['end_meal_time'][i],
                       insulin_df['start_meal_time'][i + 1]))
    return no_meal

# Extract no meal time and glucose level according to the no_meal_time_data and create Q*24 matrix


def extract_no_meal_data(df, no_meal_time):
    """Take cgm DataFrame and nomeal time list, return a q * 24 no meal dataframe"""
    cgm_df = read_cgm(df)
    no_meal_df_column = np.array(range(0, 120, 5))  # set an empty table
    no_meal_df = pd.DataFrame(columns=no_meal_df_column)
    for (start_time, end_time) in no_meal_time:
        nomeal = cgm_df.loc[
            (cgm_df.index >= start_time)
            & (cgm_df.index <= start_time + pd.Timedelta("2 hours"))]
        glucose = nomeal['Sensor Glucose (mg/dL)'].values
        if len(glucose) != 24:
            continue
        no_meal_df.loc[len(no_meal_df)] = glucose
    no_meal_df = no_meal_df.dropna(axis=0)
    return no_meal_df


# extract no meal data
no_meal_time_1 = no_meal_time(insulin_df_1)
no_meal_df_1 = extract_no_meal_data(cgm_df_1, no_meal_time_1)
no_meal_time_2 = no_meal_time(insulin_df_2)
no_meal_df_2 = extract_no_meal_data(cgm_df_2, no_meal_time_2)
nomeal_data = pd.concat([no_meal_df_1, no_meal_df_2], ignore_index=True)


#####################
# Extract features #
#####################

# calculate the difference between the min glucose level to the max glucose level after a meal
def min_to_max(df):
    """ Take the data df, return two features of the time series,
    including the climb up time span and min_to_max_feature(the time and value change
    from gluce min to max after meal."""
    feature = pd.DataFrame()
    feature["max_value"] = df.iloc[:, 2:20].max(axis=1)
    feature["max_time / min"] = df.iloc[:, 2:20].idxmax(axis=1)
    feature["min_value"] = df.iloc[:, 2:20].min(axis=1)
    feature["min_time /min"] = df.iloc[:, 2:20].idxmin(axis=1)
    # feature["climb up time"] = feature["max_time / min"] - \
    #     feature["min_time /min"]
    feature["min_to_max_feature"] = (feature["max_value"] -
                                     feature["min_value"]) / feature["min_value"]

    return feature['min_to_max_feature']

# calculate the ftt


def cal_fft(df):
    # ftt has 4 features, 'pf1', 'f1', 'pf2', 'f2'
    fft_features = pd.DataFrame(columns=['pf1', 'f1', 'pf2', 'f2'])
    for i in range(len(df)):
        # Extract the time series column as a numpy array
        data = df.iloc[i].values
        # Apply FFT
        fft_data = np.fft.fft(data)
        # Set every 5 minutes as the frequency
        freq = np.fft.fftfreq(len(data), 5)
        # Calculate the power of ftt
        pftt = np.abs(fft_data) ** 2
        # fetch the value and location of the  second and the third peak
        pftt_list = [(pftt_value, i) for (i, pftt_value) in enumerate(pftt)]
        pftt_list.sort(reverse=True)
        pf1 = pftt_list[0][0]
        pf2 = pftt_list[1][0]
        f1_location = pftt_list[0][1]
        f2_location = pftt_list[1][1]
        f1 = freq[f1_location]
        f2 = freq[f2_location]
        # f_feature = np.array([pf1, f1, pf2, f2])
        fft_features.loc[len(fft_features)] = np.array([pf1, f1, pf2, f2])
    return fft_features

# Calculate the difference between two peaks


def cal_diff(df):
    diff_feature = pd.DataFrame(columns=['diff1', 'diff2'])
    diff1_df = df.diff(axis=1)
    diff_feature['diff1'] = diff1_df.max(axis=1)
    diff_2_df = diff1_df.diff(axis=1)
    diff_feature['diff2'] = diff_2_df.max(axis=1)
    return diff_feature

# Calculate the entropy


def cal_entropy(df):
    entropy_feature = pd.DataFrame(columns=["entropy"])
    for i in range(len(df)):
        # Extract the time series column as a numpy array
        data = df.iloc[i].values
        len_param = len(data)
        entropy = 0
        value, ctr = np.unique(data, return_counts=True)
        ratio = ctr / len_param
        ratio_nonzero = np.count_nonzero(ratio)
        if ratio_nonzero <= 1:
            entropy = 0
        for u in ratio:
            entropy -= u * np.log2(u)
        entropy_feature.loc[len(entropy_feature)] = entropy
    return entropy_feature

# Summary about extract features


def extract_features(df):
    min_to_max_feature = min_to_max(df)
    fft_features = cal_fft(df)
    diff_features = cal_diff(df)
    entropy = cal_entropy(df)
    features = pd.concat(
        [min_to_max_feature, fft_features, diff_features, entropy], axis=1)
    return features


# extract feature matrix
meal_part = extract_features(meal_data)
meal_class = np.ones(len(meal_part))
meal_part["class"] = meal_class
meal_part["class"] = meal_part["class"].astype(int)

nomeal_part = extract_features(nomeal_data)
nomeal_class = np.zeros(len(nomeal_part))
nomeal_part["class"] = nomeal_class
nomeal_part["class"] = nomeal_part["class"].astype(int)


#######################
# Train Model #
#######################

# Train and test svm with k=5 fold cross validation, save the model in pickle format file
def model_training(df):
    # Assume X is your feature matrix and y is your target vector
    X = df.iloc[:, :-1]
    y = df["class"]

    # Initialize SVM
    svm_clf = SVC()

    # Initialize performance metrics
    svm_f1_scores = []
    svm_precision_scores = []
    svm_accuracy_scores = []

    # Define 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Iterate over the 5 folds
    for train_idx, test_idx in kfold.split(X, y):

        # Split the data into training and testing sets for the current fold
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # Fit SVM on the training data for the current fold
        svm_clf.fit(X_train, y_train)

        # Make predictions on the testing data for the current fold
        svm_preds = svm_clf.predict(X_test)

        # Calculate F1 score, precision, and accuracy for SVM classifier on the current fold
        svm_f1_scores.append(f1_score(y_test, svm_preds))
        svm_precision_scores.append(precision_score(y_test, svm_preds))
        svm_accuracy_scores.append(accuracy_score(y_test, svm_preds))

    # Print the mean F1 score, precision, and accuracy for SVM classifier
    print("SVM Classifier:")
    print("F1 Score:", svm_f1_scores)
    print("Precision:", np.mean(svm_precision_scores))
    print("Accuracy:", np.mean(svm_accuracy_scores))

    # save the trained model
    svm_filename = 'SVM_MODEL.pkl'
    with open(svm_filename, 'wb') as file:
        pickle.dump(svm_clf, file)


data_to_analysis = pd.concat(
    [meal_part, nomeal_part], axis=0, ignore_index=True)
# print(data_to_analysis.info())
# train models
model_training(data_to_analysis)
