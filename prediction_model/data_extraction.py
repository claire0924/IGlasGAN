# -*- coding: utf-8 -*-
"""
Extracts element-composition and target property records from the raw data and writes csv files for the prediction model.

"""

import pandas as pd
import numpy as np
from tqdm import tqdm

df_comp = pd.read_csv('sciglass_element.csv')
df_prop = pd.read_csv('sciglass_oxides_prop_org.csv')
df_comp = df_comp.set_index('GlasNo')
df_prop = df_prop.set_index('GlasNo')

prop = 'ANY_TEC'

df = pd.merge(df_comp, df_prop[prop], how='left', left_index=True, right_index=True)
df = df.dropna()

df = df[df['O'] >= 30]
df = df[df['H'] == 0]
df = df[df['C'] == 0]

df_T = df.T
df_drop = []
for i in tqdm(range(len(df_T.columns))):
    if round(np.sum(df.iloc[i, :-1]), 2) != 100:
        df_drop.append(df_T.columns[i])

df = df.drop(df_drop, axis=0)

data = df[prop]
mean = np.mean(data)
std = np.std(data)
threshold_low = mean - 3 * std
threshold_high = mean + 3 * std
df = df[df[prop] > threshold_low]
df = df[df[prop] < threshold_high]

df = df.drop(df.columns[df.max() == 0], axis=1)

nonzero_mask = df != 0
nonzero_counts = nonzero_mask.sum()
features_to_remove = nonzero_counts[nonzero_counts < 100].index

nonzero_indices = set()
for feature in features_to_remove:
    nonzero_indices.update(df.index[df[feature] != 0])
df = df.drop(columns=features_to_remove, axis=1)
df = df.drop(nonzero_indices, axis=0)

duplicates = df[df.duplicated(subset=df.columns.difference([prop]), keep=False)]
index_dict = {}
for index, row in duplicates.iterrows():
    feature_values = tuple(row[df.columns.difference([prop])])
    if feature_values not in index_dict:
        index_dict[feature_values] = [row.name]
    else:
        index_dict[feature_values].append(row.name)
for value in index_dict.values():
    new_value = [x for x in value]
    median_value = np.median([df.loc[new_value, prop]])
    df.loc[new_value, prop] = median_value
df = df.drop_duplicates()

df.to_csv('property_CTE.csv')
print(len(df.index))
print(len(df.columns))
