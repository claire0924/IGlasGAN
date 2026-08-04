# -*- coding: utf-8 -*-
"""
Fits a shallow class-balanced CART classifier to re-express a glass-composition clustering as a readable threshold rule, then quantifies the interpretability cost via SSE comparison.
"""

from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv(r'generated_6000_standard_cluster.csv')
df_comp = df.iloc[:, 1:-6]

X = df_comp

y = df['label']


def objective(params):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        clf = DecisionTreeClassifier(
            criterion=params['criterion'],
            class_weight='balanced',
            max_leaf_nodes=2,
            random_state=42)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_val)
        acc = accuracy_score(y_val, pred)
        accuracies.append(acc)
        f1_per_class = f1_score(y_val, pred, average='macro')
        f1_scores.append(f1_per_class)

    return {'loss': 1 - np.mean(f1_scores), 'status': STATUS_OK}


space = {
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print("Best parameters:", best)

best_params = {
    'criterion': ['gini', 'entropy'][best['criterion']]
}

final_clf = DecisionTreeClassifier(**best_params, class_weight='balanced', max_leaf_nodes=2, random_state=42)
final_clf.fit(X, y)

y_pred = final_clf.predict(X)
print("Final Accuracy:", accuracy_score(y, y_pred))
print("Final F1:", f1_score(y, y_pred, average='macro'))

pred_labels = final_clf.predict(X)

joblib.dump(final_clf, 'cluster_Tree.pkl')

centroids = {}
for label in np.unique(pred_labels):
    centroids[label] = df_comp[pred_labels == label].mean(axis=0).values

sse = 0
sse_per_class = defaultdict(float)

for idx, row in df_comp.iterrows():
    label = pred_labels[idx]
    centroid = centroids[label]
    distance = np.sum((row.values - centroid) ** 2)
    sse += distance
    sse_per_class[label] += distance

print("\n---------------------------- Cost of the interpretable model")
print("\nSSE Results:")
print(f"Total SSE: {sse:.2f}")
for label, val in sse_per_class.items():
    print(f"Class {label} SSE: {val:.2f}")

labels_r = y

centroids_r = {}
for label in np.unique(labels_r):
    centroids_r[label] = df_comp[labels_r == label].mean(axis=0).values

sse_r = 0
sse_per_class_r = defaultdict(float)

for idx, row in df_comp.iterrows():
    label = labels_r[idx]
    centroid = centroids_r[label]
    distance = np.sum((row.values - centroid) ** 2)
    sse_r += distance
    sse_per_class_r[label] += distance

print("\n------------------------- Cost of the optimal clustering")
print("\nSSE_r Results:")
print(f"Total SSE_r: {sse_r:.2f}")
for label, val in sse_per_class_r.items():
    print(f"Class {label} SSE_r: {val:.2f}")
