import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("/Volumes/Untitled/HAR/ppg+dalia/data/PPG_FieldStudy/S1/S1_E4/ACC_with_activity.csv")
denoisedData = pd.read_csv("/Volumes/Untitled/HAR/ppg+dalia/data/PPG_FieldStudy/S1/S1_E4/ACC_with_activity_denoised.csv")
denoisedData2 = pd.read_csv("/Volumes/Untitled/HAR/ppg+dalia/data/PPG_FieldStudy/S3/S3_E4/ACC_with_activity_denoised.csv")

X = denoisedData2[["denoised_acc_x","denoised_acc_y", "denoised_acc_z"]]
y = denoisedData2["activity"]

denoisedData2Features = denoisedData2[["denoised_acc_x", "denoised_acc_y", "denoised_acc_z"]].to_numpy()
denoisedData2Activity = denoisedData2[["activity"]].to_numpy()

denoisedData[['denoised_acc_x',"denoised_acc_y","denoised_acc_z"]].plot(title='Denoised Acc', xlabel='Index', ylabel='Values')
denoisedData2[['denoised_acc_x',"denoised_acc_y","denoised_acc_z"]].plot(title='Denoised Acc 2', xlabel='Index', ylabel='Values')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=0)

# Normalize
X_train = StandardScaler().fit_transform(X_train)
X_val = StandardScaler().fit_transform(X_val)
denoisedData2Features = StandardScaler().fit_transform(denoisedData2Features)

# # Used to find the k-value with highest accuracy
# k = np.arange(1,100)
# accuracy = np.array([])
# percision = np.array([])
# maxAccK = 0
# maxPerk = 0
# maxAcc = 0
# maxPer = 0
# for i in k:
#     knn = KNeighborsClassifier(i)
#     knn.fit(X_train, y_train)
#     acc = (accuracy_score(y_val, knn.predict(X_val)))
#     per = precision_score(y_val, knn.predict(X_val))
#     accuracy = np.append(accuracy, acc)
#     percision = np.append(percision, per)
#     if acc > maxAcc:
#         maxAccK = i
#         maxAcc = acc
#     if per > maxPer:
#         maxPerK = i
#         maxPer = per

# plt.plot(k, accuracy)
# plt.title("Model Accuracy Per K-Value")
# plt.xlabel("K-Value")
# plt.ylabel("Accuracy")
# plt.show()

# plt.plot(k, percision)
# plt.title("Model Percision Per K-Value")
# plt.xlabel("K-Value")
# plt.ylabel("Percision")
# plt.show()


# Using general KNN model
knn = KNeighborsClassifier(63)
knn.fit(X_train, y_train)

# y_pred = knn.predict(X_val)
y_pred = knn.predict(denoisedData2Features)
cmat = confusion_matrix(denoisedData2Activity, y_pred) #validate model predictions with actual y values
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cmat, display_labels = [0, 1])
cm_display.plot()
plt.show()

print("Accuracy Rate: " + str(format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat)))))

# import os
# import matplotlib.dates as mdates
# import funtions 

# patient = "001"
# directory = f"/Volumes/Untitled/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/{patient}"
# food = pd.read_csv(f"/Volumes/Untitled/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/{patient}/Food_Log_{patient}.csv", encoding='latin-1')

# dexcom = pd.read_csv(f"/Volumes/Untitled/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2/{patient}/Dexcom_{patient}.csv", encoding='latin-1')
# ds_glucose = funtions.glucoseResampler(dexcom)

# food['time_begin'] = food.iloc[:,2]
# food['time_begin'] = pd.to_datetime(food['time_begin'])


# acc = pd.read_csv(os.path.join(directory, f"ACC_{patient}.csv"), encoding='latin-1')

# acc["datetime"] = pd.to_datetime(acc["datetime"])
# acc = acc.resample("5min", on="datetime").mean().reset_index()
# acc = pd.merge(ds_glucose, acc, on='datetime')
# acc.set_index('datetime', inplace=True)
# acc = acc.dropna()
# acc_no_glucose = acc.drop(columns=['Glucose Value (mg/dL)'])
# acc_array = acc_no_glucose.to_numpy()

# predictions = knn.predict(acc_array)
# acc['predictions'] = predictions
# acc = acc.reset_index()
# acc["datetime"] = pd.to_datetime(acc["datetime"])

# start_dates = []
# end_dates = []

# count = 0
# start_index = 0

# for i, row in acc.iterrows():
#     if row['predictions'] == 1:
#         if count == 0:
#             start_index = i
#         count += 1
#     else:
#         if count >= 3:
#             start_dates.append(acc.iloc[start_index]['datetime'])
#             end_dates.append(acc.iloc[i - 1]['datetime'])
#         count = 0
#         start_index = 0

# # Check if the last sequence reached the end of the DataFrame
# if count >= 3:
#     start_dates.append(acc.iloc[start_index]['datetime'])
#     end_dates.append(acc.iloc[i - 1]['datetime'])

# # Create a new DataFrame with the start and end dates
# periods = pd.DataFrame({'all_start': start_dates, 'all_end': end_dates})

# # array to store periods of activity
# above = []

# # Find the start and end points of periods
# starts = periods["all_start"]
# ends = periods["all_end"]

# # Last row ends period if it doesn"t end within the dataset
# if len(starts) > len(ends) and len(ends) != 0:
#     ends = pd.concat([ends, pd.Series(acc["datetime"].iloc[-1])])

# for start_time, end_time in zip(starts, ends):
#   food_start = start_time - pd.Timedelta(hours=1)
#   food_end = end_time + pd.Timedelta(hours=1)
#   if start_time <= pd.Timestamp(start_time.date()) + pd.Timedelta(hours=6):
#     pass
#   elif start_time >= pd.Timestamp(start_time.date()) + pd.Timedelta(hours=21):
#     pass
#   elif not (food[(food['time_begin'] >= food_start) & (food['time_begin'] <= food_end)]).empty:
#     print("FOOOOOOOOOOOOOOOOOOOD")
#   elif len(above) != 0 and start_time <= pd.Timestamp(above[-1][-1] + pd.Timedelta(hours=1)):
#     pass
#   else:
#     above.append((start_time, end_time))

# if len(above) != 0:
#   above = np.array(above)
#   start_times = above[:, 0]
#   end_times = above[:, 1]
# else:
#   start_times = np.array([])
#   end_times = np.array([])

# start_times = pd.to_datetime(start_times)
# end_times = pd.to_datetime(end_times)

# activity = pd.DataFrame(columns = acc.columns)
# for i, (start, end) in enumerate(zip(start_times, end_times)):
#   # Filter DataFrame based on current period
#   act = acc[acc['datetime'].between(start, (end + pd.Timedelta(hours=1)))]
#   act["a"] = (end - start)
#   act["a"] = act["a"].dt.total_seconds() / 60
#   activity = pd.concat([activity, act])
