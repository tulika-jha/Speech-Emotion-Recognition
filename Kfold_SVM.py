# STRATIFIES K-FOLD CROSS VALIDATION { 10-fold }
import datetime
# Import Required Modules.
import json
from statistics import mean, stdev

import imblearn
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


#open extracted features
with open("features_experiment.json", "r") as f:
    data_dict = json.load(f)

data_with_labels = data_dict["0"]
data = []
labels = []


#populating data matrix
for i in range(len(data_with_labels)):
    data.append(data_with_labels[i][0])
    labels.append(data_with_labels[i][1])


data = np.asarray(data)
#data = np.load("X_diff.npy")



#oversampling minority class

oversample = imblearn.over_sampling.SMOTE()
data, labels = oversample.fit_resample(data, labels)


# Feature Scaling for input features.
"""
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(data)
"""
sc = StandardScaler()
x_scaled = sc.fit_transform(data)

y = np.asarray(labels)


# Create  classifier object.
C_best = 100#100
gamma_best = 0.025#0.0029  # 0.0028
svclassifier = SVC(kernel='rbf', gamma=gamma_best, C=C_best)

# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []
times = []
y_actual_list = []
y_pred_list = []

for train_index, test_index in skf.split(x_scaled, labels):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    a = datetime.datetime.now()
    svclassifier.fit(x_train_fold, y_train_fold)
    b = datetime.datetime.now()
    lst_accu_stratified.append(svclassifier.score(x_test_fold, y_test_fold))

    y_pred = svclassifier.predict(x_test_fold)

    times.append((b-a))
    #add actual and pred labels to lists
    y_actual_list.append(y_test_fold.copy())
    y_pred_list.append(y_pred.copy())


s = np.zeros((8,8))

for i in range(len(y_actual_list)):
    print("Fold ",i,"..................")
    print("accuracy = ", lst_accu_stratified[i])
    #print(classification_report(y_actual_list[i], y_pred_list[i]))
    #print(confusion_matrix(y_actual_list[i], y_pred_list[i]))
    s += confusion_matrix(y_actual_list[i], y_pred_list[i])

#print(s)
# Print the output.
print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified) * 100, '%')
print('\nMinimum Accuracy:',
      min(lst_accu_stratified) * 100, '%')
print('\nOverall Accuracy:',
      mean(lst_accu_stratified) * 100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))
print("average time taken by a fold = ", np.asarray(times).mean())