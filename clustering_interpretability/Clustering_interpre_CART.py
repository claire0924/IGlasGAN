# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 13:45:19 2025

@author: user
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score,f1_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from collections import defaultdict
import joblib
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('generated_6000_stansard_cluster.csv')
df_comp = df.iloc[:,1:-6]

X=df_comp

y = df['label']


def objective(params):
    # 五折交叉验证
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = DecisionTreeClassifier(
            #max_depth=params['max_depth'],
            #min_samples_split=params['min_samples_split'],
            #min_samples_leaf=params['min_samples_leaf'],
            criterion=params['criterion'],
            class_weight='balanced', 
            max_leaf_nodes=4,
            random_state=42 )
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_val)
        acc = accuracy_score(y_val, pred)
        accuracies.append(acc)
        # 计算加权F1（小类权重更高）
        f1_per_class = f1_score(y_val, pred, average='macro')
        #weighted_f1 = 0.25*f1_per_class[0] + 0.25*f1_per_class[1] + 0.25*f1_per_class[2] + 0.25*f1_per_class[3]
        f1_scores.append(f1_per_class)
    
    # 返回需要最小化的损失（1 - 平均准确率）
    #return {'loss': 1 - np.mean(accuracies), 'status': STATUS_OK}
    return {'loss': 1 - np.mean(f1_scores), 'status': STATUS_OK}

# 定义参数空间
space = {
    #'max_depth': hp.choice('max_depth', range(3, 13)),         
    #'min_samples_split': hp.choice('min_samples_split',range(5, 50)),  
    #'min_samples_leaf': hp.choice('min_samples_leaf',range(2, 20)),    
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}


#X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2)

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)


print("Best parameters:", best)


best_params = {
    #'max_depth': best['max_depth'] + 3,  
    #'min_samples_split': best['min_samples_split']+5,
    #'min_samples_leaf': best['min_samples_leaf']+2,
    'criterion': ['gini', 'entropy'][best['criterion']]
}

final_clf = DecisionTreeClassifier(**best_params, class_weight='balanced',max_leaf_nodes=4,random_state=42 )
final_clf.fit(X, y)  # 使用全部数据训练

# 评估整体准确性
y_pred = final_clf.predict(X)
print("Final Accuracy:", accuracy_score(y, y_pred))
print("Final F1:", f1_score(y, y_pred ,average='macro'))
joblib.dump(final_clf,'cluster_tree.pkl')
# ======================
# 计算解释模型的可解释性成本，质心和SSE
# ======================
# 获取预测标签
pred_labels = final_clf.predict(X)

# 计算每个类别的质心
centroids = {}
for label in np.unique(pred_labels):
    centroids[label] = df_comp[pred_labels == label].mean(axis=0).values

# 计算每个样本到对应质心的距离平方
sse = 0
sse_per_class = defaultdict(float)

for idx, row in df_comp.iterrows():
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
labels_r = y

# 计算每个类别的质心
centroids_r = {}
for label in np.unique(labels_r):
    centroids_r[label] = df_comp[labels_r == label].mean(axis=0).values

# 计算每个样本到对应质心的距离平方
sse_r = 0
sse_per_class_r = defaultdict(float)

for idx, row in df_comp.iterrows():
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

#%%-----------------------------------------树可视化
from sklearn.tree import export_graphviz
import graphviz
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('generated_6000_stansard_cluster.csv')


plt.rcParams['font.family']= 'Times New Roman'
plt.rcParams['font.weight']= 'bold'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['axes.linewidth']=2
plt.rcParams['xtick.bottom']=True
plt.rcParams['xtick.color']='black'
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.left']=True
plt.rcParams['ytick.color']='black'
plt.rcParams['ytick.direction']='in'
plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['figure.dpi']=300

# 提取特征和标签
X = df.iloc[:,1:-6]
y = df['label']

#加载模型
clf = joblib.load('cluster_clf_best_extreme.pkl')


plt.figure(figsize=(15,5))
# 绘制决策树
plot_tree(clf,
          filled=True,            # 颜色填充
          rounded=True,           # 圆角节点
          feature_names=X.columns.tolist(),  # 特征名称
          class_names=['C0','C1','C2','C3'],  # 类别名称
          proportion=True,        # 显示比例代替样本数
          max_depth=10,            # 显示前3层（避免过大）
          fontsize=10,
          label="all",
          impurity=True,
          precision=3,
          ax=None)             # 字体大小


plt.show()
# 生成可视化图形
#graph = graphviz.Source(dot_data)
#graph.render("decision_tree", view=True)


