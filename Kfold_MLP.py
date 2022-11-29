# STRATIFIES K-FOLD CROSS VALIDATION { 10-fold }

# Import Required Modules.
import datetime
import json
from statistics import mean, stdev

import imblearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# open extracted features
with open("features_652.json", "r") as f:
    data_dict = json.load(f)

data_with_labels = data_dict[ "0" ]
data = [ ]
labels = [ ]

# populating data matrix
for i in range(len(data_with_labels)):
    data.append(data_with_labels[ i ][ 0 ])
    labels.append(data_with_labels[ i ][ 1 ])

data = np.asarray(data)

data = np.load("X_diff.npy")

# oversampling minority class
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

# one hot encode class labels
emotions = [ "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised" ]
y_encoded = [ ]
index_emotions = {}

for i in range(len(emotions)):
    index_emotions[ emotions[ i ] ] = i

for i in range(len(y)):
    temp_array = np.zeros(len(emotions))
    temp_array[ index_emotions[ y[ i ] ] ] = 1
    y_encoded.append(temp_array.copy())

y_encoded = np.asarray(y_encoded)
y = y_encoded

# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
lst_accu_stratified = [ ]
times = [ ]
class_reports = [ ]
conf_matrics = [ ]
y_actual_list = [ ]
y_pred_list = [ ]

for train_index, test_index in skf.split(x_scaled, labels):
    x_train_fold, x_test_fold = x_scaled[ train_index ], x_scaled[ test_index ]
    y_train_fold, y_test_fold = y[ train_index ], y[ test_index ]
    model = Sequential()
    model.add(Dense(640, activation='relu', input_dim=len(x_scaled[ 0 ])))
    # model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(8, activation='softmax'))
    a = datetime.datetime.now()
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[ 'accuracy' ])
    model.fit(x_train_fold, y_train_fold, epochs=20)
    b = datetime.datetime.now()

    _, accuracy = model.evaluate(x_test_fold, y_test_fold)

    y_pred = model.predict_classes(x_test_fold)

    lst_accu_stratified.append(accuracy)
    times.append((b - a))
    print("ACCURACY = " + str(accuracy))
    # Convert encoded vectors back to emotion labels
    y_actual = [ ]
    for i in y_test_fold:
        y_actual.append(emotions[ int(np.argmax(i)) ])
    y_predicted = [ ]
    for i in y_pred:
        y_predicted.append(emotions[ i ])

    y_actual_list.append(y_actual.copy())
    y_pred_list.append(y_predicted.copy())



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