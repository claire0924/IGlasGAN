# -*- coding: utf-8 -*-
"""
Builds an IMM (Iterative Mistake Minimization) threshold tree that reproduces a given glass clustering with minimal misclassification, and reports the SSE cost of the interpretable partition.
"""

from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from graphviz import Digraph

df = pd.read_csv(r'generated_6000_standard_cluster.csv')

X = df.iloc[:, 1:-6].values
labels = df['label'].values
feature_names = df.iloc[:, 1:-6].columns

centers = np.array([X[labels == i].mean(axis=0) for i in np.unique(labels)])


class TreeNode:
    def __init__(self, data_indices, center_indices, depth=0):
        self.data_indices = data_indices
        self.center_indices = center_indices
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.depth = depth

    def is_leaf(self):
        return len(self.center_indices) == 1

    def build_tree(self, X, labels, centers):
        if self.is_leaf():
            return

        best_feature = None
        best_threshold = None
        min_errors = float('inf')

        valid_mask = np.isin(labels[self.data_indices], self.center_indices)
        valid_data_indices = self.data_indices[valid_mask]

        if len(valid_data_indices) == 0:
            return

        for feature in range(X.shape[1]):
            X_valid = X[valid_data_indices]
            sorted_indices = np.argsort(X_valid[:, feature])
            X_sorted = X_valid[sorted_indices]
            labels_sorted = labels[valid_data_indices[sorted_indices]]

            thresholds = (X_sorted[1:, feature] + X_sorted[:-1, feature]) / 2

            for i in range(len(thresholds)):
                theta = thresholds[i]
                S_left = [c for c in self.center_indices if centers[c, feature] <= theta]
                S_right = [c for c in self.center_indices if centers[c, feature] > theta]

                if not S_left or not S_right:
                    continue

                left_labels = labels_sorted[:i + 1]
                right_labels = labels_sorted[i + 1:]
                errors = (np.sum(~np.isin(left_labels, S_left)) + (np.sum(~np.isin(right_labels, S_right))))

                if errors < min_errors:
                    min_errors = errors
                    best_feature = feature
                    best_threshold = theta

        if best_feature is not None:
            self.feature = best_feature
            self.threshold = best_threshold

            left_mask = X[self.data_indices, best_feature] <= best_threshold
            left_data_indices = self.data_indices[left_mask]
            S_left = [c for c in self.center_indices if centers[c, best_feature] <= best_threshold]

            right_data_indices = self.data_indices[~left_mask]
            S_right = [c for c in self.center_indices if centers[c, best_feature] > best_threshold]

            self.left = TreeNode(left_data_indices, S_left, self.depth + 1)
            self.left.build_tree(X, labels, centers)
            self.right = TreeNode(right_data_indices, S_right, self.depth + 1)
            self.right.build_tree(X, labels, centers)

    def predict(self, samples):
        if self.is_leaf():
            return np.full(samples.shape[0], self.center_indices[0])

        if samples.ndim == 1:
            if samples[self.feature] <= self.threshold:
                return self.left.predict(samples)
            else:
                return self.right.predict(samples)
        else:
            left_mask = samples[:, self.feature] <= self.threshold
            pred_labels = np.empty(samples.shape[0], dtype=int)
            pred_labels[left_mask] = self.left.predict(samples[left_mask])
            pred_labels[~left_mask] = self.right.predict(samples[~left_mask])
            return pred_labels

def visualize_tree(node, feature_names, graph=None):
    if graph is None:
        graph = Digraph()

    if node.is_leaf():
        label = f"Cluster {node.center_indices[0]}"
    else:
        label = f"Split on {feature_names}\n<= {node.threshold:.2f}"

    graph.node(name=str(id(node)), label=label)

    if node.left:
        graph.edge(str(id(node)), str(id(node.left)), label='Yes')
        visualize_tree(node.left, feature_names, graph)
    if node.right:
        graph.edge(str(id(node)), str(id(node.right)), label='No')
        visualize_tree(node.right, feature_names, graph)

    return graph


tree = TreeNode(data_indices=np.arange(X.shape[0]), center_indices=list(range(len(centers))))
tree.build_tree(X, labels, centers)

joblib.dump(tree, 'cluster_IMM.pkl')

graph = visualize_tree(tree, feature_names)
graph.attr(dpi=str(300))


pred_labels = tree.predict(X)

centroids = {}
for label in np.unique(pred_labels):
    centroids[label] = df.iloc[:, 1:-6][pred_labels == label].mean(axis=0).values

sse = 0
sse_per_class = defaultdict(float)

for idx, row in df.iloc[:, 1:-6].iterrows():
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

labels_r = labels

centroids_r = {}
for label in np.unique(labels_r):
    centroids_r[label] = df.iloc[:, 1:-6][labels_r == label].mean(axis=0).values

sse_r = 0
sse_per_class_r = defaultdict(float)

for idx, row in df.iloc[:, 1:-6].iterrows():
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
