# STRATIFIES K-FOLD CROSS VALIDATION { 10-fold }

# Import Required Modules.
import json
from statistics import mean, stdev

import imblearn
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#open extracted features
from sklearn.preprocessing import StandardScaler

with open("features_652.json", "r") as f:
    data_dict = json.load(f)

data_with_labels = data_dict["0"]
data = []
labels = []

#populating data matrix
for i in range(len(data_with_labels)):
    data.append(data_with_labels[i][0])
    labels.append(data_with_labels[i][1])


data = np.load("X_diff.npy")
data = np.asarray(data)


#oversampling minority class
oversample = imblearn.over_sampling.SMOTE()
data, labels = oversample.fit_resample(data, labels)


# Feature Scaling for input features.
sc = StandardScaler()
x_scaled = sc.fit_transform(data)

y = np.asarray(labels)


# Create  classifier object.
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=1)
lst_accu_stratified = []
y_actual_list = []
y_pred_list = []

for train_index, test_index in skf.split(x_scaled, labels):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    rf.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(rf.score(x_test_fold, y_test_fold))

    y_pred = rf.predict(x_test_fold)

    y_actual_list.append(y_test_fold.copy())
    y_pred_list.append(y_pred.copy())

#Evaluating the model...
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
