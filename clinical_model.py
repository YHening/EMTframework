#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 20:22:00 2020

@author: Sope
"""
from sklearn import tree, preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier, ElasticNet, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from uuid import uuid4, UUID
import pickle
import pandas as pd
from itertools import cycle
from sklearn.metrics import confusion_matrix
import src.detec_helper as dh
from tensorflow.keras.utils import to_categorical
import src.utils_detectron as ud
from sklearn.metrics import roc_auc_score
input_data = "./training_features.csv"
all_data = []

best_acc = 0
best_model = None
best_models = []
VALID_PART = 0.2
TEST_PART = 0.2
scores = dict()
def features_data(data_fr_loc,folder_index):
    train_idx = []
    valid_idx = []
    test_idx = []
    train_and_valid_idx = []
    # get the categories by which to split
    cats = [1,2,3,4]
    # folder_index=2
    folder_index = int(folder_index)
    #df_test = pd.read_csv('/public/yanghening/Bonetumor/BonetumorNet-main/clinical_test.csv')
    for cat in cats:
        # get all matching data_fr_loc entries
        # print(data_fr_loc[ENTITY_KEY])
        data_fr_loc_loc = data_fr_loc.loc[
            data_fr_loc['Entity'] == cat]  # data_fr_loc[ENTITY_KEY]列名索引，找到列名叫Tumor.Entitaet的列
        # data_fr_loc_loc_test = df_test.loc[
        #     df_test['Entity'] == cat]


        loclen = len(data_fr_loc_loc)

        # now split acc to the indices
        validlen = round(loclen * VALID_PART)
        testlen = round(loclen * TEST_PART)
        trainlen = loclen - validlen - testlen

        # get the matching indices and extend the idx list
        #
        folder_validlen = (validlen + trainlen) / 5
        data_fr_loc_loc_valid = data_fr_loc_loc.iloc[
                                int(folder_validlen * (folder_index - 1)):int(folder_validlen * folder_index)]
        valid_idx.extend(list(data_fr_loc_loc_valid.index))

        data_fr_loc_loc_train_and_valid = data_fr_loc_loc.iloc[:trainlen + validlen]
        train_and_valid_idx.extend(list(data_fr_loc_loc_train_and_valid.index))

        zhongjian = list(data_fr_loc_loc_train_and_valid.index)

        del (zhongjian[int(folder_validlen * (folder_index - 1)):int(folder_validlen * folder_index)])
        train_idx.extend(zhongjian)

        data_fr_loc_loc_test = data_fr_loc_loc.iloc[trainlen + validlen::]
        test_idx.extend(list(data_fr_loc_loc_test.index))
        #test_idx.extend(list(data_fr_loc_loc_test.index))
    # lists for data to be given to logistic regression

    file_name = data_fr_loc
    train1_set = []
    train1_labels = []
    training_labels = file_name['labels']
    all_locations_int = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    all_locations_1hot = to_categorical(all_locations_int)  # make one-hot vector for each location
    for i in range(len(file_name)):
        age = file_name.loc[i, 'age']
        sex = file_name.loc[i, 'gender']
        #red_cell = file_name.loc[i, 'red_cell']
        white_cell = file_name.loc[i, 'white_cell']
        swelling = file_name.loc[i, 'swelling']
        pain = file_name.loc[i, 'pain']

        package = file_name.loc[i, 'package']

        location_number = file_name.loc[i, 'location']
        this_location = all_locations_1hot[location_number]  # pulls one-hot vector for input location
        location =this_location
        addition = [age, sex,white_cell,swelling,pain,package]
        for e in location:
            addition.append(e)
        train1_set.append(addition)
        train1_labels.append(training_labels[i])

    val_labels = training_labels[valid_idx]

    test_labels = training_labels[test_idx]

    train_labels = training_labels[train_idx]
    val_set = [train1_set[i]for i in valid_idx]
    test_set = [train1_set[i]for i in test_idx]
    train_set= [train1_set[i]for i in train_idx]
    train_set = preprocessing.scale(train_set)
    val_set = preprocessing.scale(val_set)
    test_set = preprocessing.scale(test_set)

    return train_set, train_labels, val_set, val_labels, test_set, test_labels


def characterize_data(data):
    unique, counts = np.unique(data.classes, return_counts=True)
    index_to_count = dict(zip(unique, counts))
    characterization = {int(c): index_to_count[data.class_indices[c]] for c in data.class_indices}
    return characterization


def features_run(label_form, classifier, split_id=None, model="n/a"):
    # create split id and run id
    run_id = uuid4()
    if split_id is None:
        split_id = run_id

        # create initial data
    df = pd.read_csv('/public/yanghening/Bonetumor/BonetumorNet-main/clinical_two.csv')
    for folder_index in ['1', '2', '3', '4', '5']:
        train_set, train_labels, val_set, val_labels, test_set, test_labels= features_data(df,folder_index)
        c = classifier
        clf = c()  # , max_depth=depth) #max_iter=1000)#, kernel="linear", probability=True)
        #clf.fit(train_set, train_labels)
        clf = RFECV(clf, cv=4, step=1)  # n_features_to_select=j,
        clf.fit(train_set, train_labels)
        y_pro = clf.predict_proba(test_set)

        probabilities = clf.predict_proba(val_set).tolist()
        probabilities = [i[1] for i in probabilities]
        test_probabilities = clf.predict_proba(test_set).tolist()
        scores[int(folder_index)] = test_probabilities
        a = 1
        score = test_probabilities
        b = []
        for i in range(len(score)):
            b.append(np.array(score[i]))
        c = np.array(b)
        targets = test_labels

        yy = np.array(targets)
        d = c[:, 1]
        dd = d.ravel()
        fpr, tpr, threshold = roc_curve(yy, dd)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        print(roc_auc)
        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        
    result = np.average([scores[1], scores[2], scores[3], scores[4], scores[5]], axis=0).tolist()
    scores = result
    b = []
    for i in range(len(scores)):
        b.append(np.array(scores[i]))
    c = np.array(b)
    predict_label = []
    for j in range(0, len(result)):
        x = np.where(c[j, :] == np.max(c[j, :]))
        x1 = list(x)
        x2 = list(map(int, x1))
        for xx in x2:
            x3 = xx

        predict_label.append(x3)

    targets = test_labels
    conf = confusion_matrix(targets, predict_label)
    dh.print_confinfo(conf)
    yy = np.array(targets)
    d = c[:, 1]
    dd = d.ravel()
    fpr, tpr, threshold = roc_curve(yy, dd)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    print(roc_auc)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':

    features_run("outcome_pos", LogisticRegression, UUID("84a64c17-fe3e-440c-aaaf-e1bd5b02576f"), "logistic regression")
