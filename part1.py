# Step1: Determine the time stamp when Auto mode starts. The data starts with manual mode until you get a message “AUTO MODE ACTIVE PLGM OFF” in the column “Q” of the InsulinData.csv.
# Step2: Figure out the timestamp in CGMData.csv where Auto Mode starts, which can be done simply by searching for the time stamp nearest to (and later than) the Auto mode start time stamp obtained from InsulinData.csv.
# Step3: Tackle the “missing data problem”, Popular strategies include deletion of the entire day of data, or interpolation.
# Step4: Parsed and divided CGM data into segments of a day. One day is considered to start at 12 am and end at 11:59 pm, which has 288 samples in each segment.
# Step5: Each segment is then divided into two sub-segments: the daytime sub-segment (06:00-23:59) and the overnight sub-segment (00:00-05:59).
# Step6: Count the number of samples that belong to the ranges specified in the metrics.
# Step7: Calculate the percentage for the metrics.
# Step8: Save results as Result.csv

import pandas as pd

# Load the files and generate timestamp
data1 = pd.read_csv("InsulinData.csv")
Insulin_df = data1[['Date', 'Time', 'Alarm']]
Insulin_df["Timestamp"] = pd.to_datetime(
    Insulin_df['Date'] + ' ' + Insulin_df['Time'])

data2 = pd.read_csv("CGMData.csv")
CGM_df = data2[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
CGM_df['Timestamp'] = pd.to_datetime(CGM_df['Date'] + ' ' + CGM_df['Time'])
CGM_df = CGM_df.dropna()  # dropna.


# find the time when manual mode changing to auto mode, and create time interval for wholeday, overnight and daytime.
mode_change = "AUTO MODE ACTIVE PLGM OFF"
changing_to_auto_time = Insulin_df.loc[Insulin_df['Alarm']
                                       == mode_change]['Timestamp'].min()

# before changing time is manual mode
wholeday_manual_df = CGM_df.loc[CGM_df['Timestamp'] <= changing_to_auto_time]
# after changing time is auto mode
wholeday_auto_df = CGM_df.loc[CGM_df['Timestamp'] > changing_to_auto_time]

# Count number of distinct elements in "Date". Can ignore NaN values.
manual_dates = wholeday_manual_df["Date"].nunique()
auto_dates = wholeday_auto_df["Date"].nunique()

# overnight
overnight_manual_df = wholeday_manual_df.loc[wholeday_manual_df['Timestamp'].dt.hour.between(
    0, 6)]
overnight_auto_df = wholeday_auto_df.loc[wholeday_auto_df['Timestamp'].dt.hour.between(
    0, 6)]
# daytime
daytime_manual_df = wholeday_manual_df.loc[wholeday_manual_df['Timestamp'].dt.hour.between(
    7, 24)]
daytime_auto_df = wholeday_auto_df.loc[wholeday_auto_df['Timestamp'].dt.hour.between(
    7, 24)]


# calculate the sum of time
def calculation(df):
    if len(df) == 0:
        return 0
    else:
        return (df.groupby('Date')['Date'].count()).sum()

# get 6 metrics


def metrics(df):
    hyperglycemia = calculation(df[df['Sensor Glucose (mg/dL)'] > 180])
    hyperglycemia_critical = calculation(
        df[df['Sensor Glucose (mg/dL)'] > 250])
    range = calculation(df[(df['Sensor Glucose (mg/dL)'] >= 70)
                        & (df['Sensor Glucose (mg/dL)'] <= 180)])
    range_secondary = calculation(
        df[(df['Sensor Glucose (mg/dL)'] >= 70) & (df['Sensor Glucose (mg/dL)'] <= 150)])
    hypoglycemia_level1 = calculation(df[df['Sensor Glucose (mg/dL)'] < 70])
    hypoglycemia_level2 = calculation(df[df['Sensor Glucose (mg/dL)'] < 54])
    return [hyperglycemia, hyperglycemia_critical, range, range_secondary, hypoglycemia_level1, hypoglycemia_level2]


# calculate the percentage time for metrics
manual_metrics_sum = (metrics(overnight_manual_df) +
                      metrics(daytime_manual_df) + metrics(wholeday_manual_df))
auto_metrics_sum = (metrics(overnight_auto_df) +
                    metrics(daytime_auto_df) + metrics(wholeday_auto_df))
manual_metrics = [round(i * 100 / manual_dates / 288, 2)
                  for i in manual_metrics_sum]
auto_metrics = [round(i * 100 / auto_dates / 288, 2) for i in auto_metrics_sum]

# save the results
result = pd.DataFrame(data=[manual_metrics, auto_metrics],
                      index=["Manual Mode", "Auto Mode"],
                      columns=["Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)",
                               "Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
                               "Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
                               "Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
                               "Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
                               "Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
                               "Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)",
                               "Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
                               "Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
                               "Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
                               "Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
                               "Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)",
                               "Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)",
                               "Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)",
                               "Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)",
                               "Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)",
                               "Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)",
                               "Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"])

result.to_csv("Result.csv", index=False, header=None)
