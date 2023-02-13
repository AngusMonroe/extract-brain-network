"""
Code to run the experiments based on Grid Search

Under INPUT PARAMETERS, select datasets/parcellations/models to run on

Results will be generated as .csv files in the same filepath
"""

from scipy.io import loadmat

import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


"""
INPUT PARAMETERS -
Define the datasets, parcellations, and models for this run of Grid Search
"""

dataset = ["ppmi", "taowu", "neurocon"]  # ["abide", "adni", "ppmi", "taowu", "neurocon"]
parcellation = ["schaefer100"]  # ["AAL116", "harvard48", "schaefer100", "kmeans100", "ward100"]
models = ["LR", "NB", "SVC", "kNN", "RF"]


""" assign ROI counts to specified parcellation schemes """
num_roi = []
for x in range(0, len(parcellation)):
    if parcellation[x] == "AAL116":
        num_roi.append(116)
    elif parcellation[x] == "harvard48":
        num_roi.append(48)
    elif parcellation[x] == "schaefer100":
        num_roi.append(100)
    elif parcellation[x] == "kmeans100":
        num_roi.append(100)
    elif parcellation[x] == "ward100":
        num_roi.append(100)


def load_data(dataset, parcellation, num_rois):
    """
    Return [X, y] by first loading data then creating feature matrices by
    loading edge weights into X, and loading class labels into y
    """
    group_paths = glob.glob("data/" + dataset + "_" + parcellation + "/*")
    subject_paths = []
    splits = {}
    for path in group_paths:
        if path[-6:] == '.index':
            for split in ['train', 'val', 'test']:
                if split in path:
                    f = open(path, 'r')
                    splits[split] = [[int(s) for s in l.split(',')] for l in f.readlines()]
                    # splits[split] = np.loadtxt(path, delimiter=',', dtype=int)
            continue
        subject_paths += glob.glob(path + '/*')
    y = []
    X = np.zeros(num_rois * num_rois)

    for x in range(0, len(subject_paths)):
        # mat = loadmat(subject_paths[x])
        # adjacency = mat["data"]
        adjacency = np.loadtxt(subject_paths[x], delimiter=' ')

        X = np.vstack([X, adjacency.flatten()])

        # generate class label list y based on subject ID
        if "control" in subject_paths[x]:
            y.append(1)
        elif "patient" in subject_paths[x]:
            y.append(2)
        elif "mci" or "prodromal" in subject_paths[x]:
            y.append(3)
        elif "emci" or "swedd" in subject_paths[x]:
            y.append(4)
        elif "SMC" in subject_paths[x]:
            y.append(5)
        elif "LMCI" in subject_paths[x]:
            y.append(6)

    X = np.delete(X, 0, axis=0)  # delete empty first row of zeros from X

    split_iters = []
    if splits:
        for fold in range(len(splits['train'])):
            split_iters.append((splits['train'][fold], splits['val'][fold]))

    return [X, y], splits, split_iters


def cross_validation(X, y, dataset, parcellation, modelname, splits, split_iters):
    """
    Runs cross-validation and saves the Grid Search results
    into a .csv in the same filepath
    """
    if modelname == "LR":
        parameters = {"penalty": ("l2", 'none')}
        model = LogisticRegression(max_iter=1000000)
    elif modelname == "kNN":
        parameters = {
            "n_neighbors": (3, 4, 5, 6),
            "weights": ("uniform", "distance"),
            "p": (1, 2),
        }
        model = KNeighborsClassifier()
    elif modelname == "SVC":
        parameters = {
            "kernel": ("rbf", "linear", "poly", "sigmoid"),
            "C": [0.1, 1, 10],
            "gamma": ("auto", "scale"),
        }
        model = SVC()
    elif modelname == "RF":
        parameters = {
            "n_estimators": (50, 100, 150, 200),
            "criterion": ("gini", "entropy"),
            "max_depth": (2, 3, 4, 5),
        }
        model = RandomForestClassifier()
    elif modelname == "NB":
        parameters = {}
        model = GaussianNB()

    clf = GridSearchCV(model, parameters, cv=split_iters, refit=False) if split_iters else GridSearchCV(model, parameters)
    clf.fit(X, y)
    best_params = clf.best_params_

    df = pd.DataFrame.from_dict(clf.cv_results_)
    filepath = Path('result/' + dataset + "_" + parcellation + "_" + modelname + ".csv")
    df.to_csv(filepath)

    print("------------------------------")
    print("dataset:", dataset)
    print('subject number:', len(y))
    print("parcellation:", parcellation)
    print("model:", modelname)
    print("DONE")
    if splits:
        accs = []
        for i, test_idx in enumerate(splits['test']):
            train_idx = splits['train'][i]
            train_X = X[train_idx]
            train_y = np.array(y)[train_idx]
            fold_clf = model.set_params(**best_params)
            fold_clf.fit(train_X, train_y)
            predicted = fold_clf.predict(X[test_idx])
            test_acc = sum(predicted == np.array(y)[test_idx]) / len(np.array(y)[test_idx])
            accs.append(test_acc)
        acc_mean = np.mean(np.array(accs))
        acc_std = np.std(np.array(accs))
        print('test mean acc: {}'.format(acc_mean))
        print('test acc std: {}'.format(acc_std))
    print("------------------------------")


""" runs the loop and call methods to run experiments """
for i in range(0, len(dataset)):
    for j in range(0, len(parcellation)):
        for k in range(0, len(models)):
            data, splits, split_iters = load_data(dataset[i], parcellation[j], num_roi[j])
            cross_validation(data[0], data[1], dataset[i], parcellation[j], models[k], splits, split_iters)
