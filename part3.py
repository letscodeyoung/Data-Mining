import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


#######################
# Extract ground truth #
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


# Load data
insulin_df_1 = pd.read_csv("InsulinData.csv", low_memory=False)
cgm_df_1 = pd.read_csv("CGMData.csv", low_memory=False)


# Extract meal time from InsulinData.csv
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

    # insulin_df.set_index("start_meal_time", inplace=True)
    # return insulin_df.index.tolist()

    insulin_df.set_index("start_meal_time", inplace=True)
    stime = insulin_df.index.tolist()
    carb_input = insulin_df["BWZ Carb Input (grams)"].values.tolist()
    possible_meal = zip(stime, carb_input)
    return possible_meal


# Extract meal time and glucose level from CGMData.csv and format them in a P*30 matrix.
# And sort the min and max values of 'BWZ Carb Input (grams)'
def extract_meal_data(df, start_meal_time):
    """Take a CGM DataFrame and start time list, return a DataFrame containing p * 30 meal data"""
    cgm_df = read_cgm(df)
    meal_df_column = np.array(range(0, 150, 5))  # set an empty table
    meal_df = pd.DataFrame(columns=meal_df_column)
    carb_list = []
    for (time, carb) in start_meal_time:
        # meal time should larger than the start_time and less than it + 2.5 hours
        meal = cgm_df.loc[
            (cgm_df.index >= time) & (cgm_df.index <= time + pd.Timedelta('2.5 hours'))]
        # read the glucose level
        glucose = meal['Sensor Glucose (mg/dL)'].values
        # since 150/5=30, so when add to 30 columns, we should change to the new row
        if len(glucose) != 30:
            continue
        else:
            meal_df.loc[len(meal_df)] = glucose
            carb_list.append(carb)
    meal_df['carb'] = carb_list
    meal_df = meal_df.dropna(axis=0)
    meal_df = meal_df.sort_values("carb").reset_index().drop("index", axis=1)
    return meal_df


# extract meal data
meal_time_1 = meal_time(insulin_df_1)
meal_data = extract_meal_data(cgm_df_1, meal_time_1)


# descretize the meal amount in bins of size 20.
def get_ground_truth(df):
    min_value = df["carb"].min()
    n = int((df["carb"].max() - df["carb"].min()) // 20)
    for i in range(n):
        df.loc[df["carb"].between(
            min_value + i * 20, min_value + (i + 1) * 20), "bin"] = i
    df = df.dropna(axis=0)
    return df


# Generate ground truth, assign binnumbers
# Now, we should have n bins, the meal data should be put into respective bins according to their meal carb input
ground_truth = get_ground_truth(meal_data)


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
    feature["min_to_max_feature"] = (feature["max_value"] -
                                     feature["min_value"]) / feature["min_value"]

    return feature['min_to_max_feature']

# calculate the ftt, and we only need pf1 as the ftt features


def cal_fft(df):
    fft_features = pd.DataFrame(columns=['pf'])
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
        # pf2 = pftt_list[1][0]
        # f1_location = pftt_list[0][1]
        # f2_location = pftt_list[1][1]
        # f1 = freq[f1_location]
        # f2 = freq[f2_location]
        # f_feature = np.array([pf1, f1, pf2, f2])
        fft_features.loc[len(fft_features)] = np.array([pf1])
    return fft_features

# Calculate the difference between two peaks


def cal_diff(df):
    diff_feature = pd.DataFrame(columns=['diff1', 'diff2'])
    diff1_df = df.diff(axis=1)
    diff_feature['diff1'] = diff1_df.max(axis=1)
    diff_2_df = diff1_df.diff(axis=1)
    diff_feature['diff2'] = diff_2_df.max(axis=1)
    return diff_feature['diff2']

# Summary about extract features


def extract_features(df):
    min_to_max_feature = min_to_max(df)
    fft_features = cal_fft(df)
    diff_features = cal_diff(df)
    features = pd.concat(
        [min_to_max_feature, fft_features, diff_features], axis=1)

    # Instantiate a StandardScaler object
    # StandardScaler can normalize/standardize (μ = 0 and σ = 1) features of X before applying any machine learning model.
    scaler = StandardScaler()
    # Use scaler.fit_transform to standardize feature data
    features_std = scaler.fit_transform(features)
    # Convert the standardized data back to a DataFrame
    features_df = pd.DataFrame(features_std, columns=features.columns)

    return features_df


# Extract standardized features from the first to the last three columns
features = extract_features(ground_truth.iloc[:, :-2])


##########################
# Performing Clustering #
##########################

# Initialize k-means and dbscan object with number of clusters
X = features
n = ground_truth["bin"].nunique()
kmeans = KMeans(n_clusters=n, random_state=42)
dbscan = DBSCAN()

# Fit the k-means and dbscan object to the data
kmeans.fit(X)
dbscan.fit(X)

# Predict the cluster labels for each data point
kmeans_labels = kmeans.labels_
# For dbscan, noisy samples are given the label -1.
dbscan_labels = dbscan.labels_


###########################
# Compute SSE #
###########################

# Calculate the SSE for kmeans
# 'kmeans.inertia_': Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
sse_for_kmeans = kmeans.inertia_

# Calculate the SSE for dbscan, first find the centroids of the clusters
centroids = []
for label in np.unique(dbscan_labels):
    if label == -1:  # noise points are labeled as -1
        continue
    centroid = np.mean(X[dbscan_labels == label], axis=0)
    centroids.append(centroid)

# Calculate the SSE for dbscan
sse_dbscan = 0
for label in np.unique(dbscan_labels):
    if label == -1:
        continue
    cluster = X[dbscan_labels == label]
    centroid = centroids[label]
    sse_dbscan += np.sum((cluster - centroid) ** 2)
    sse_for_dbscan = sum(sse_dbscan)
# print(sum(sse_dbscan), sse_for_kmeans)


####################################
# Calculate the Entropy and purity #
####################################

# Use the features matrix to cluster the meal data into n clusters, and calculate entropy and purity
def cal_entropy_purity(df, cluster_name):
    ep = pd.DataFrame()
    for bin in df['bin'].unique():
        df_ep = df.loc[df['bin'] == bin].groupby(cluster_name).count()['bin']
        ep = pd.concat([ep, df_ep], axis=1)
    ep = ep.replace(np.NaN, 0)
    ep = ep.sort_index()
    ep.columns = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4']

    sum_array = np.sum(ep.values, axis=1).reshape((5, 1))
    weight = sum_array/len(df)
    weight_array = np.array(weight)
    freq_matrix = np.divide(ep.values, sum_array)
    eps = 1e-8
    freq_matrix = freq_matrix + eps
    entropy_list = []
    for row in freq_matrix:
        log_row = np.log2(row)
        entropy = -np.dot(row, log_row)
        entropy_list.append(entropy)
    entropy_array = np.array(entropy_list)

    ep['weights'] = weight_array
    ep['entropy'] = entropy_array
    ep['weighted_entropy'] = ep['weights'] * ep['entropy']
    ep['max bin'] = ep.max(axis=1)
    total_entropy = round(sum(ep['weighted_entropy'].values)/4, 2)
    total_purity = round(2.4*sum(ep['max bin'].values)/len(df), 2)
    return [total_entropy, total_purity]


# Combine the labels to calculate entropy and purity
features['bin'] = ground_truth['bin']
features['kmeans'] = kmeans_labels
features['dbscan'] = dbscan_labels

# entropy and purity for kmeans, [entropy_for_kmeans, purity_for_kmeans]
kmeans_result = cal_entropy_purity(features, 'kmeans')
entropy_for_kmeans = kmeans_result[0]
purity_for_kmeans = kmeans_result[1]

# entropy and purity for dbscan, [entropy_for_dbscan, purity_for_dbscan]
# handle the noise points first
features['dbscan'] = features['dbscan'].replace(-1, np.NaN)
features = features.dropna(axis=0)
dbscan_result = cal_entropy_purity(features, 'dbscan')
entropy_for_dbscan = dbscan_result[0]
purity_for_dbscan = dbscan_result[1]

# generate final output and save it
result = [sse_for_kmeans, sse_for_dbscan, entropy_for_kmeans,
          entropy_for_dbscan, purity_for_kmeans, purity_for_dbscan]
result_df = pd.DataFrame(data=result)
result_df = result_df.transpose()
result_df.to_csv('Result.csv', header=None, index=None)
