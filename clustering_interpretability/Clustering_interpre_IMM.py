# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:05:12 2026

@author: user
"""
import numpy as np
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
from collections import defaultdict

df=pd.read_csv('generated_6000_stansard_cluster.csv')

# 提取特征和标签
X = df.iloc[:,1:-6].values
labels = df['label'].values
feature_names = df.iloc[:,1:-6].columns  # 获取特征名称

# 计算每个簇的中心
centers = np.array([X[labels == i].mean(axis=0) for i in np.unique(labels)])

class TreeNode:
    def __init__(self, data_indices, center_indices, depth=0):
        self.data_indices = data_indices  # 数据点的索引
        self.center_indices = center_indices  # 当前节点的中心索引列表
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

        # 获取当前有效数据点（标签属于当前中心集合）
        valid_mask = np.isin(labels[self.data_indices], self.center_indices)
        valid_data_indices = self.data_indices[valid_mask]

        if len(valid_data_indices) == 0:
            return

        for feature in range(X.shape[1]):
            # 按当前特征排序有效数据
            X_valid = X[valid_data_indices]
            sorted_indices = np.argsort(X_valid[:, feature])
            X_sorted = X_valid[sorted_indices]
            labels_sorted = labels[valid_data_indices[sorted_indices]]

            # 生成候选阈值（相邻数据点的中点）
            thresholds = (X_sorted[1:, feature] + X_sorted[:-1, feature]) / 2

            for i in range(len(thresholds)):
                theta = thresholds[i]
                # 分割当前中心集合
                S_left = [c for c in self.center_indices if centers[c, feature] <= theta]
                S_right = [c for c in self.center_indices if centers[c, feature] > theta]

                if not S_left or not S_right:
                    continue  # 无效分割

                # 计算错误数
                left_labels = labels_sorted[:i+1]
                right_labels = labels_sorted[i+1:]
                errors = (np.sum(~np.isin(left_labels, S_left)) + (np.sum(~np.isin(right_labels, S_right))))

                if errors < min_errors:
                    min_errors = errors
                    best_feature = feature
                    best_threshold = theta

        if best_feature is not None:
            self.feature = best_feature
            self.threshold = best_threshold

            # 分割数据点
            left_mask = X[self.data_indices, best_feature] <= best_threshold
            left_data_indices = self.data_indices[left_mask]
            S_left = [c for c in self.center_indices if centers[c, best_feature] <= best_threshold]

            right_data_indices = self.data_indices[~left_mask]
            S_right = [c for c in self.center_indices if centers[c, best_feature] > best_threshold]

            # 递归构建子树
            self.left = TreeNode(left_data_indices, S_left, self.depth + 1)
            self.left.build_tree(X, labels, centers)
            self.right = TreeNode(right_data_indices, S_right, self.depth + 1)
            self.right.build_tree(X, labels, centers)
        
            
    def predict(self, samples):
        if self.is_leaf():
        # 如果是叶节点，返回当前节点的类别
            return np.full(samples.shape[0], self.center_indices[0])  # 返回所有样本的类别

    # 根据当前节点的特征和阈值划分样本
        if samples.ndim == 1:
        # 单个样本
            if samples[self.feature] <= self.threshold:
                return self.left.predict(samples)
            else:
                return self.right.predict(samples)
        else:
        # 多个样本
            left_mask = samples[:, self.feature] <= self.threshold
            pred_labels = np.empty(samples.shape[0], dtype=int)
            pred_labels[left_mask] = self.left.predict(samples[left_mask])
            pred_labels[~left_mask] = self.right.predict(samples[~left_mask])
            return pred_labels
        
        
feature_names=['Li$_{2}$O','B$_{2}$O$_{3}$','Na$_{2}$O','MgO','Al$_{2}$O$_{3}$','SiO$_{2}$','P$_{2}$O$_{5}$','K$_{2}$O','CaO','TiO$_{2}$','Fe$_{2}$O$_{3}$','CuO','ZnO','As$_{2}$O$_{3}$','SrO','Y$_{2}$O$_{3}$','ZrO$_{2}$','In$_{2}$O$_{3}$','SnO$_{2}$','Sb$_{2}$O$_{3}$','BaO','La$_{2}$O$_{3}$','CeO$_{2}$','PbO','Bi$_{2}$O$_{3}$']

def format_feature_name(feature_name):
    replacements = {
        "$_{2}$": "₂",
        "$_{3}$": "₃",
        "$_{5}$": "₅",
    }
    for latex, unicode_sub in replacements.items():
        feature_name = feature_name.replace(latex, unicode_sub)
    return feature_name

def visualize_tree(node, feature_names, graph=None):
    if graph is None:
        graph = Digraph()
        #graph.attr('node', style='filled', color='lightblue', fontname='Times New Roman')  # 美化节点样式
        #graph.attr('edge', fontname='Times New Roman')  # 美化边样式

    # 格式化节点标签
    if node.is_leaf():
        label = f"Cluster {node.center_indices[0]}"
    else:
        feature_name = format_feature_name(feature_names[node.feature])
        label = f"Split on {feature_name}\n<= {node.threshold:.2f}"

    # 添加节点
    graph.node(name=str(id(node)), label=label)

    # 递归处理子节点
    if node.left:
        graph.edge(str(id(node)), str(id(node.left)), label='Yes')
        visualize_tree(node.left, feature_names, graph)
    if node.right:
        graph.edge(str(id(node)), str(id(node.right)), label='No')
        visualize_tree(node.right, feature_names, graph)

    return graph


# 构建树
tree = TreeNode(data_indices=np.arange(X.shape[0]), center_indices=list(range(len(centers))))
tree.build_tree(X, labels, centers)

#保存树
joblib.dump(tree,'C://Users\\user\\Desktop\\GAN玻璃反向设计\\oxide\\性能惩罚项（修正）\\model_save\\cluster_IMM.pkl')

# 可视化树
graph = visualize_tree(tree, feature_names)
graph.attr(dpi=str(300))
graph.render('C://Users//user//Desktop//imm_tree.png', format='png', view=True)

# 测试 predict 函数
    #计算代价
    # ======================
    # 计算解释模型的可解释性成本，质心和SSE
    # ======================
    # 获取预测标签
pred_labels = tree.predict(X)
    
    # 计算每个类别的质心
centroids = {}
for label in np.unique(pred_labels):
        centroids[label] = df.iloc[:,1:-6][pred_labels == label].mean(axis=0).values
    
    # 计算每个样本到对应质心的距离平方
sse = 0
sse_per_class = defaultdict(float)
    
for idx, row in df.iloc[:,1:-6].iterrows():
        label = pred_labels[idx]
        centroid = centroids[label]
        distance = np.sum((row.values - centroid) ** 2)
        sse += distance
        sse_per_class[label] += distance
    
print("\n----------------------------解释模型代价")
print("\nSSE Results:")
print(f"Total SSE: {sse:.2f}")
for label, val in sse_per_class.items():
        print(f"Class {label} SSE: {val:.2f}")
        
    # ======================
    # 计算最优化聚类结果的可解释性成本，质心和SSE
    # ======================
    # 获取预测标签
labels_r = labels
    
    # 计算每个类别的质心
centroids_r = {}
for label in np.unique(labels_r):
        centroids_r[label] = df.iloc[:,1:-6][labels_r == label].mean(axis=0).values
    
    # 计算每个样本到对应质心的距离平方
sse_r = 0
sse_per_class_r = defaultdict(float)
    
for idx, row in df.iloc[:,1:-6].iterrows():
        label = labels_r[idx]
        centroid = centroids_r[label]
        distance = np.sum((row.values - centroid) ** 2)
        sse_r += distance
        sse_per_class_r[label] += distance
    
print("\n-------------------------最优化聚类代价")
print("\nSSE_r Results:")
print(f"Total SSE_r: {sse_r:.2f}")
for label, val in sse_per_class_r.items():
        print(f"Class {label} SSE_r: {val:.2f}")

    
