
#---------------------------------------单性能数据提取
import pandas as pd
import numpy as np
from tqdm import tqdm

df_comp= pd.read_csv('sciglass_element.csv')
df_prop= pd.read_csv('sciglass_oxides_prop_org.csv')
df_comp=df_comp.set_index('GlasNo')         
df_prop=df_prop.set_index('GlasNo')

#prop='MIKROTV'   #硬度
prop='ANY_TEC'  #膨胀系数
#prop='DENSITY'  #密度
#prop='TG'       #转变温度
#prop='MOD_UNG'  #杨氏模量
#prop='SPT'      #应变点温度
#prop='Tsoft'    #软化温度
#prop='ANPT'     #退火温度
#prop='MOD_SDV'  #剪切模量
#prop='pois'     #泊松比
#prop='ND300'    #折射率
#prop='Tm'       #阿贝数
#prop='Vmax'

#prop_range=[0.1,15]   #硬度
#prop_range=[0.1,300]  #膨胀系数
#prop_range=[1,10]     #密度
#prop_range=[25,1250]  #转变温度
#prop_range=[1,170]    #杨氏模量
#prop_range=[25,1250]  #应变点温度

#将第一个表和第二个表合并
df=pd.merge(df_comp,df_prop[prop],how='left',left_index=True, right_index=True)
df=df.dropna()     #删除含有缺失值的行


#数据筛选
df=df[df['O']>=30]
df=df[df['H']==0]
#df=df[df['F']==0]
#df=df[df['N']>0]
#df=df[df['S']==0]
df=df[df['C']==0]
#df=df[df['Cl']==0]
#df=df[df['Br']==0]
#df=df[df['I']==0]

#去除组分之和不为100的不完整项
df_T=df.T
df_drop=[]
for i in tqdm(range(len(df_T.columns))):
    if round(np.sum(df.iloc[i,:-1]),2)!=100:
        df_drop.append(df_T.columns[i])

df=df.drop(df_drop,axis=0)    

#异常值剔除
data=df[prop]
mean=np.mean(data)
std=np.std(data)
threshold_low=mean-3*std
threshold_high=mean+3*std
df=df[df[prop]>threshold_low]
df=df[df[prop]<threshold_high]

#剔除始终为0的特征
df=df.drop(df.columns[df.max()==0], axis=1)
#剔除非零元素小于100个的特征及对应的非0样本
#将不为0的特征值标记为True，为0的特征值标记为False
nonzero_mask = df != 0
# 计算每个特征中不为0的样本数量
nonzero_counts = nonzero_mask.sum()
# 找到样本数量少于100个的特征
features_to_remove = nonzero_counts[nonzero_counts < 100].index

nonzero_indices = set()
for feature in features_to_remove:
    nonzero_indices.update(df.index[df[feature] != 0])
# 删除这些特征及对应的样本
df = df.drop(columns=features_to_remove,axis=1)
# 删除特征值不为0的样本数量少于100个的样本
df = df.drop(nonzero_indices,axis=0)

#去除重复值
duplicates = df[df.duplicated(subset=df.columns.difference([prop]),keep=False)]
index_dict = {}
for index, row in duplicates.iterrows():
    feature_values = tuple(row[df.columns.difference([prop])])
    if feature_values not in index_dict:
        index_dict[feature_values] = [row.name]
    else:
        index_dict[feature_values].append(row.name)
for value in index_dict.values():
    new_value=[x for x in value]
    median_value=np.median([df.loc[new_value,prop]])
    df.loc[new_value,prop]=median_value
df=df.drop_duplicates()


df.to_csv('property_CTE.csv')
print(len(df.index))
print(len(df.columns))
