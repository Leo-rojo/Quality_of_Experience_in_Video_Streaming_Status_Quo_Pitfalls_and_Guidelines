import numpy as np
import pickle
import os
import json
import pandas as pd
from itu_p1203 import P1203Standalone
from sklearn.preprocessing import LabelBinarizer
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
os.chdir('C:/Users/leona/Desktop/QoE_comsnet/QoE_combination/QoE_selector')
all_features=np.load('allfeat_allscores/all_feat_hdtv.npy',allow_pickle=True)
feat_vmaf=np.load('features_and_scores_qoes_hdtv/feat_vmaf.npy')
feat_va=np.load('features_and_scores_qoes_hdtv/feat_va.npy')
#merge feat_vmaf and feat_va
feat_merge=[]
for i in range(len(feat_vmaf)):
    feat_merge.append(np.append(feat_vmaf[i],feat_va[i]))
#remove last column from all_fetures
all_features=all_features[:,:-1]
#remove column 3,16,29,42,55,68,81
all_features=np.delete(all_features, [3,16,29,42,55,68,81], axis=1)
#best_models=np.load('bestmodels_hdtv.npy',allow_pickle=True)
best_models_va_vmaf=np.load('bestmodels_va_vmaf_hdtv.npy',allow_pickle=True)
print(best_models_va_vmaf)
#label the best models as one hot encoding
# Initialize the LabelBinarizer
lb = LabelBinarizer()
# Fit the LabelBinarizer
lb.fit(best_models_va_vmaf)
# Transform the labels
one_hot_encoded = lb.transform(best_models_va_vmaf)

for rs in range(1,100):
    #build rf classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # Split the data into a training and test set.
    X_train, X_test, y_train, y_test = train_test_split(all_features, one_hot_encoded, random_state=rs)
    #normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    # Instantiate the classifier: clf
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    #normalize test features
    X_test = scaler.transform(X_test)
    # Predict the labels of the test set: y_pred
    y_pred = clf.predict(X_test)
    # Compute and print the confusion matrix and classification report
    from sklearn.metrics import classification_report, confusion_matrix
    y_test_lab = lb.inverse_transform(y_test)
    y_pred_lab = lb.inverse_transform(y_pred)
    #print(confusion_matrix(y_test_lab, y_pred_lab))
    #print(classification_report(y_test_lab, y_pred_lab))
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    #calculate accuracy
    print(accuracy_score(y_test_lab, y_pred_lab))
    #put in df y_test_lab, y_pred_lab, y_pred_prob
    df=pd.DataFrame({'y_test_lab':y_test_lab,'y_pred_lab':y_pred_lab,'y_pred_prob_0':y_pred_prob[:,0],'y_pred_prob_1':y_pred_prob[:,1],})
    ########################
    #build svr classifier
    # from sklearn.svm import SVC
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score
    # # Split the data into a training and test set.
    # X_train, X_test, y_train, y_test = train_test_split(all_features, one_hot_encoded, random_state=42)
    # #normalize features
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # # Fit only to the training data
    # scaler.fit(X_train)
    # # Now apply the transformations to the data:
    # X_train = scaler.transform(X_train)
    # # Instantiate the classifier: clf
    # clf = SVC(kernel='rbf', C=1, gamma=1)
    # # Fit the classifier to the training data
    # clf.fit(X_train, y_train)
    # #normalize test features
    # X_test = scaler.transform(X_test)
    # # Predict the labels of the test set: y_pred
    # y_pred = clf.predict(X_test)
    # # Compute and print the confusion matrix and classification report
    # from sklearn.metrics import classification_report, confusion_matrix
    # y_test_lab = lb.inverse_transform(y_test)
    # y_pred_lab = lb.inverse_transform(y_pred)
    # print(confusion_matrix(y_test_lab, y_pred_lab))
    # print(classification_report(y_test_lab, y_pred_lab))
    # # Compute predicted probabilities: y_pred_prob
    # #y_pred_prob = clf.predict_proba(X_test)
    # #calculate accuracy
    # print(accuracy_score(y_test_lab, y_pred_lab))
#
