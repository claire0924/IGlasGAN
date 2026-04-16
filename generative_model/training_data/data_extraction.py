import pandas as pd
import numpy as np
from tqdm import tqdm

df_comp= pd.read_csv('sciglass_element.csv')
df_prop= pd.read_csv('sciglass_oxides_prop_org.csv')
df_comp=df_comp.set_index('GlasNo')         
df_prop=df_prop.set_index('GlasNo')

prop1='ANY_TEC'  #膨胀系数
prop4='DENSITY'  #密度
prop3='MOD_UNG'  #杨氏模量
prop2='SPT'      #应变点温度

# 合并表格
df = pd.merge(df_comp, df_prop[[prop1, prop2, prop3, prop4]], 
              left_index=True, right_index=True, how='left')

df = df.dropna()

# 数据筛选
df = df[df['O']>=30]
df = df[(df['F']==0) & (df['N']==0) & (df['S']==0) & (df['C']==0) & (df['H']==0) &
        (df['Cl']==0) & (df['Br']==0) & (df['I']==0)]

# 去除组分之和不为100的项
to_drop = []
for idx in tqdm(df.index):
    if round(np.sum(df.loc[idx, :-4]), 2) != 100:
        to_drop.append(idx)
df = df.drop(to_drop, axis=0)

# 3σ异常值剔除（同时筛选，更科学）
def filter_3sigma(df, col):
    mean = df[col].mean()
    std = df[col].std()
    return df[(df[col] > mean-3*std) & (df[col] < mean+3*std)]

for p in [prop1, prop2, prop3, prop4]:
    df = filter_3sigma(df, p)

# 剔除全0列
df = df.drop(df.columns[df.max() == 0], axis=1)

# 剔除非零样本 <10 的特征及对应样本
nonzero_mask = df != 0
nonzero_counts = nonzero_mask.sum()
features_to_remove = nonzero_counts[nonzero_counts < 10].index

to_remove_samples = set()
for f in features_to_remove:
    to_remove_samples.update(df.index[df[f] != 0])

df = df.drop(columns=features_to_remove)
df = df.drop(to_remove_samples, axis=0)

# 去重（简化版，效果一样）
feat_cols = df.columns.difference([prop1,prop2,prop3,prop4])
df = df.groupby(feat_cols, as_index=False).median()
df = df.drop_duplicates()

# 氧化物转换
column=['Li$_{2}$O','B$_{2}$O$_{3}$','Na$_{2}$O','MgO','Al$_{2}$O$_{3}$','SiO$_{2}$',
        'P$_{2}$O$_{5}$','K$_{2}$O','CaO','TiO$_{2}$','Fe$_{2}$O$_{3}$','CuO','ZnO',
        'As$_{2}$O$_{3}$','SrO','Y$_{2}$O$_{3}$','ZrO$_{2}$','In$_{2}$O$_{3}$','SnO$_{2}$',
        'Sb$_{2}$O$_{3}$','BaO','La$_{2}$O$_{3}$','CeO$_{2}$','PbO','Bi$_{2}$O$_{3}$']

df_oxide = pd.DataFrame(columns=column, index=df.index)

df_oxide['Li$_{2}$O'] = df['Li']/2
df_oxide['B$_{2}$O$_{3}$'] = df['B']/2
df_oxide['Na$_{2}$O'] = df['Na']/2
df_oxide['MgO'] = df['Mg']
df_oxide['Al$_{2}$O$_{3}$'] = df['Al']/2
df_oxide['SiO$_{2}$'] = df['Si']
df_oxide['P$_{2}$O$_{5}$'] = df['P']/2
df_oxide['K$_{2}$O'] = df['K']/2
df_oxide['CaO'] = df['Ca']
df_oxide['TiO$_{2}$'] = df['Ti']
df_oxide['Fe$_{2}$O$_{3}$'] = df['Fe']/2
df_oxide['CuO'] = df['Cu']
df_oxide['ZnO'] = df['Zn']
df_oxide['As$_{2}$O$_{3}$'] = df['As']/2
df_oxide['SrO'] = df['Sr']
df_oxide['Y$_{2}$O$_{3}$'] = df['Y']/2
df_oxide['ZrO$_{2}$'] = df['Zr']
df_oxide['In$_{2}$O$_{3}$'] = df['In']/2
df_oxide['SnO$_{2}$'] = df['Sn']
df_oxide['Sb$_{2}$O$_{3}$'] = df['Sb']/2
df_oxide['BaO'] = df['Ba']
df_oxide['La$_{2}$O$_{3}$'] = df['La']/2
df_oxide['CeO$_{2}$'] = df['Ce']
df_oxide['PbO'] = df['Pb']
df_oxide['Bi$_{2}$O$_{3}$'] = df['Bi']/2

# 归一化到100%
df_oxide = df_oxide.div(df_oxide.sum(axis=1), axis=0) * 100

# 加回性能
df_oxide[prop1] = df[prop1].values
df_oxide[prop2] = df[prop2].values
df_oxide[prop3] = df[prop3].values
df_oxide[prop4] = df[prop4].values

# 电中性校验
df_ele = pd.DataFrame(index=df_oxide.index)
df_ele['Li'] = df_oxide['Li$_{2}$O'] * 2
df_ele['B'] = df_oxide['B$_{2}$O$_{3}$'] * 2
df_ele['O'] = (df_oxide['Li$_{2}$O'] 
+ df_oxide['B$_{2}$O$_{3}$']*3 
+ df_oxide['Na$_{2}$O'] 
+ df_oxide['MgO'] 
+ df_oxide['Al$_{2}$O$_{3}$']*3 
+ df_oxide['SiO$_{2}$']*2 
+ df_oxide['P$_{2}$O$_{5}$']*5 
+ df_oxide['K$_{2}$O'] 
+ df_oxide['CaO'] 
+ df_oxide['TiO$_{2}$']*2 
+ df_oxide['Fe$_{2}$O$_{3}$']*3 
+ df_oxide['CuO'] 
+ df_oxide['ZnO'] 
+ df_oxide['As$_{2}$O$_{3}$']*3 
+ df_oxide['SrO'] 
+ df_oxide['Y$_{2}$O$_{3}$']*3 
+ df_oxide['ZrO$_{2}$']*2 
+ df_oxide['In$_{2}$O$_{3}$']*3 
+ df_oxide['SnO$_{2}$']*2 
+ df_oxide['Sb$_{2}$O$_{3}$']*3 
+ df_oxide['BaO'] 
+ df_oxide['La$_{2}$O$_{3}$']*3 
+ df_oxide['CeO$_{2}$']*2 
+ df_oxide['PbO'] 
+ df_oxide['Bi$_{2}$O$_{3}$']*3)

df_ele['Na'] = df_oxide['Na$_{2}$O']*2
df_ele['Mg'] = df_oxide['MgO']
df_ele['Al'] = df_oxide['Al$_{2}$O$_{3}$']*2
df_ele['Si'] = df_oxide['SiO$_{2}$']
df_ele['P'] = df_oxide['P$_{2}$O$_{5}$']*2
df_ele['K'] = df_oxide['K$_{2}$O']*2
df_ele['Ca'] = df_oxide['CaO']
df_ele['Ti'] = df_oxide['TiO$_{2}$']
df_ele['Fe'] = df_oxide['Fe$_{2}$O$_{3}$']*2
df_ele['Cu'] = df_oxide['CuO']
df_ele['Zn'] = df_oxide['ZnO']
df_ele['As'] = df_oxide['As$_{2}$O$_{3}$']*2
df_ele['Sr'] = df_oxide['SrO']
df_ele['Y'] = df_oxide['Y$_{2}$O$_{3}$']*2
df_ele['Zr'] = df_oxide['ZrO$_{2}$']
df_ele['In'] = df_oxide['In$_{2}$O$_{3}$']*2
df_ele['Sn'] = df_oxide['SnO$_{2}$']
df_ele['Sb'] = df_oxide['Sb$_{2}$O$_{3}$']*2
df_ele['Ba'] = df_oxide['BaO']
df_ele['La'] = df_oxide['La$_{2}$O$_{3}$']*2
df_ele['Ce'] = df_oxide['CeO$_{2}$']
df_ele['Pb'] = df_oxide['PbO']
df_ele['Bi'] = df_oxide['Bi$_{2}$O$_{3}$']*2

# 计算差异
df_ele = df_ele.div(df_ele.sum(axis=1), axis=0)*100
for col in df_ele.columns:
    df_ele[col] = (df_ele[col] - df.loc[df_ele.index, col]).abs()

df_ele['sum_abs_diff'] = df_ele.sum(axis=1)
df_oxide['sum_abs_diff'] = df_ele['sum_abs_diff']

# 过滤电中性不满足的样本
df_oxide = df_oxide[df_oxide['sum_abs_diff'] <= 0.1]


df_oxide = df_oxide.drop('sum_abs_diff', axis=1)


df_oxide = df_oxide.reset_index(drop=True)
df_oxide.to_csv('property_4.csv', index=False)
