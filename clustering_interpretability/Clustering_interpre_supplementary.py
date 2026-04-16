import numpy as np
import pandas as pd
import joblib
from sklearn.tree import _tree


df = pd.read_csv('generated_6000_stansard_cluster.csv')
df_comp = df.iloc[:, 1:-6]
X = df_comp
y = df['label']
feature_names = X.columns.tolist()

model_path = 'cluster_tree.pkl'
final_clf = joblib.load(model_path)
tree = final_clf.tree_


def parse_tree_structure():
    leaf_ids = final_clf.apply(X)
    node_to_samples = {}

    def get_leaf_descendants(node_id):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return [node_id]
        else:
            left_leaves = get_leaf_descendants(tree.children_left[node_id])
            right_leaves = get_leaf_descendants(tree.children_right[node_id])
            return left_leaves + right_leaves

    for node_id in range(tree.node_count):
        node_leaves = get_leaf_descendants(node_id)
        node_samples = np.where(np.isin(leaf_ids, node_leaves))[0]
        node_to_samples[node_id] = node_samples

    split_steps = []
    root_node = 0
    root_left_node = tree.children_left[root_node]
    root_right_node = tree.children_right[root_node]
    split_steps.append({
        "step_name": "根节点划分（区分C3与0/1/2）",
        "current_node_id": root_node,
        "left_target_clusters": [0, 1, 2],
        "right_target_clusters": [3],
        "samples_idx": node_to_samples[root_node]
    })

    inner_node_1 = root_left_node
    split_steps.append({
        "step_name": "内部节点划分（区分2与0/1）",
        "current_node_id": inner_node_1,
        "left_target_clusters": [2],
        "right_target_clusters": [0, 1],
        "samples_idx": node_to_samples[inner_node_1]
    })

    inner_node_2 = tree.children_right[inner_node_1]
    split_steps.append({
        "step_name": "最终内部节点划分（区分0与1）",
        "current_node_id": inner_node_2,
        "left_target_clusters": [1],
        "right_target_clusters": [0],
        "samples_idx": node_to_samples[inner_node_2]
    })
    return split_steps

split_steps = parse_tree_structure()


def calculate_split_wrong_count(rule, samples_idx, left_target_clusters, right_target_clusters):
    step_X = X.iloc[samples_idx]
    step_y = y.iloc[samples_idx].values

    if "≤" in rule:
        feat_name, threshold_str = rule.split(" ≤ ")
        threshold = float(threshold_str)
        left_mask = step_X[feat_name] <= threshold
    elif ">" in rule:
        feat_name, threshold_str = rule.split(" > ")
        threshold = float(threshold_str)
        left_mask = step_X[feat_name] > threshold
    else:
        return np.inf, np.inf, np.inf, ""

    right_mask = ~left_mask
    left_wrong = np.sum(~np.isin(step_y[left_mask], left_target_clusters))
    right_wrong = np.sum(~np.isin(step_y[right_mask], right_target_clusters))
    total_wrong = left_wrong + right_wrong
    return total_wrong, left_wrong, right_wrong, feat_name


def get_top5_rules(step):
    samples_idx = step["samples_idx"]
    left_target = step["left_target_clusters"]
    right_target = step["right_target_clusters"]


    all_rules = []
    for feat in feature_names:
        vals = X.iloc[samples_idx][feat].values
        uniques = np.unique(vals)
        if len(uniques) < 2:
            continue
        thresholds = (uniques[:-1] + uniques[1:]) / 2
        for t in thresholds:
            for op in ["≤", ">"]:
                rule = f"{feat} {op} {t:.4f}"
                tw, lw, rw, fn = calculate_split_wrong_count(rule, samples_idx, left_target, right_target)
                all_rules.append({
                    "rule": rule, "feat_name": fn,
                    "total_wrong": tw, "left_wrong": lw, "right_wrong": rw
                })


    best_per_feature = {}
    for r in all_rules:
        ft = r["feat_name"]
        if ft not in best_per_feature or r["total_wrong"] < best_per_feature[ft]["total_wrong"]:
            best_per_feature[ft] = r


    sorted_rules = sorted(best_per_feature.values(), key=lambda x: x["total_wrong"])
    return sorted_rules[:5]


for step in split_steps:
    print("=" * 100)
    print(f"划分步骤：{step['step_name']}")
    print(f"样本总数：{len(step['samples_idx'])}")
    print(f"左目标簇：{step['left_target_clusters']} | 右目标簇：{step['right_target_clusters']}")


    top5_rules = get_top5_rules(step)

    print("\n全局最优前5条规则（特征不重复｜每个特征自动选错分最小的规则）")
    for i, info in enumerate(top5_rules, 1):
        print(f"\n{i}、规则：{info['rule']}")
        print(f"   总错分数：{info['total_wrong']}")
        print(f"   左节点错分（非目标簇）：{info['left_wrong']}")
        print(f"   右节点错分（非目标簇）：{info['right_wrong']}")
