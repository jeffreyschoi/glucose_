import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

dataframes = {}
for i in range(15):
    name = f"data{i+1}"
    denoisedData = pd.read_csv(f"/Volumes/Untitled/HAR/ppg+dalia/data/PPG_FieldStudy/S{i+1}/S{i+1}_E4/ACC_with_activity_denoised.csv")
    denoisedData = denoisedData[["denoised_acc_x","denoised_acc_y","denoised_acc_z","activity"]]
    dataframes[name] = denoisedData


for trainer in dataframes:
    X = dataframes[trainer][["denoised_acc_x","denoised_acc_y", "denoised_acc_z"]]
    y = dataframes[trainer]["activity"]
    
    X = StandardScaler().fit_transform(X)
    
    KFoldAccuracy = np.array([])
    KFoldPercision = np.array([])
    
    for tester in dataframes:
        if tester != trainer:
            testingFeatures = dataframes[tester][["denoised_acc_x", "denoised_acc_y", "denoised_acc_z"]].to_numpy()
            testingActivity = dataframes[tester]["activity"].to_numpy()
            
            testingFeatures = StandardScaler().fit_transform(testingFeatures)
            
            knn = KNeighborsClassifier(1)
            knn.fit(X, y)
        
            # y_pred = knn.predict(X_val)
            y_pred = knn.predict(testingFeatures)
            cmat = confusion_matrix(testingActivity, y_pred) #validate model predictions with actual y values
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cmat, display_labels = [0, 1])
            cm_display.plot()
            plt.show()
            
            KFoldAccuracy = np.append(KFoldAccuracy, accuracy_score(testingActivity, y_pred))
            KFoldPercision = np.append(KFoldPercision, precision_score(testingActivity, y_pred))
            print(f"{tester} has been tested on {trainer}")

    plt.plot(np.arange(1, 15), KFoldAccuracy)
    plt.title(f"Accuracy of Individuals Tested on S{trainer}")
    plt.xlabel("Tested Individual")
    plt.ylabel("Accuracy")
    plt.show()
    
    plt.plot(np.arange(1, 15), KFoldPercision)
    plt.title(f"Percision of Individuals Tested on S{trainer}")
    plt.xlabel("Tested Individual")
    plt.ylabel("Percision")
    plt.show()

