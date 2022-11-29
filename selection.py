import json
import numpy as np
from sklearn.feature_selection import mutual_info_classif, GenericUnivariateSelect

with open("features_652.json","r") as f:
    data_dict = json.load(f)


data_with_labels = data_dict["0"]

data = []
labels = []

#populating data matrix
for i in range(len(data_with_labels)):
    data.append(data_with_labels[i][0])
    labels.append(data_with_labels[i][1])


data = np.asarray(data)
labels = np.asarray(labels)


MFCC_sel = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=20)
X_trans_MFCC = MFCC_sel.fit_transform(data[:, 0:360], labels)
print("shape of MFCC features -> ",np.asarray(data[:, 0:360]).shape)
print(X_trans_MFCC.shape," <- MFCC")

LFCC_sel = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=20)
X_trans_LFCC = LFCC_sel.fit_transform(data[:, 382:616], labels)
print("shape of LFCC features -> ",np.asarray(data[:, 382:616]).shape)
print(X_trans_LFCC.shape," <- LFCC")

X_sc = data[:, 360:364].copy()
X_for = data[:, 364:382].copy()
X_in = data[:, 616:634].copy()
X_pit = data[:, 634:652].copy()

X_total = np.concatenate((X_trans_MFCC, X_sc, X_for, X_trans_LFCC, X_in, X_pit), axis=1)
print("shape before feature selection -> ",np.asarray(data).shape)
print("concatenated........")
print(X_total.shape)

np.save("X_diff.npy", X_total)