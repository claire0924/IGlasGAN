
import pandas as pd
from params import Parameters
import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv('property_4.csv')
df = df.set_index('GlasNo')

# 数据归一化
#计算数据的最小值和最大值
max = [54,84,27.66,46.2,32.208,100,18.48,23.1,43.536,6,2.772,10.272,43.716,1.428,16.44,5.952,10.128,2.4,0.684,2.4,24,13.128,0.36,12.696,55.188,150,1068,129.6,3.96]
min = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.04,302.4,36.64,1.672]
index = ['Li$_{2}$O','B$_{2}$O$_{3}$','Na$_{2}$O','MgO','Al$_{2}$O$_{3}$','SiO$_{2}$','P$_{2}$O$_{5}$','K$_{2}$O','CaO','TiO$_{2}$','Fe$_{2}$O$_{3}$','CuO','ZnO','As$_{2}$O$_{3}$','SrO','Y$_{2}$O$_{3}$','ZrO$_{2}$','In$_{2}$O$_{3}$','SnO$_{2}$','Sb$_{2}$O$_{3}$','BaO','La$_{2}$O$_{3}$','CeO$_{2}$','PbO','Bi$_{2}$O$_{3}$','ANY_TEC','SPT','MOD_UNG','DENSITY']
data_min = pd.Series(min, index=index)
data_max = pd.Series(max, index=index)

generator=torch.load('generator_oxide_property(4)_extend.pth')

# 生成新数据
noise = torch.rand((10000, 29))
# 生成范围在[-0.2,1.2)范围内的随机数矩阵
#noise = -0.2 + 1.4 * noise
#noise = torch.tensor(np.random.rand(8000, params.z_dim),dtype=torch.float32)

noise = noise.to(device)
fake= generator(noise)


# 将数据反归一化并保存
def inverse_standardize(data, data_min, data_max):
    return data * (data_max-data_min)+ data_min

fake = fake.cpu()
fake_data = fake.detach().numpy()
column=['Li$_{2}$O','B$_{2}$O$_{3}$','Na$_{2}$O','MgO','Al$_{2}$O$_{3}$','SiO$_{2}$','P$_{2}$O$_{5}$','K$_{2}$O','CaO','TiO$_{2}$','Fe$_{2}$O$_{3}$','CuO','ZnO','As$_{2}$O$_{3}$','SrO','Y$_{2}$O$_{3}$','ZrO$_{2}$','In$_{2}$O$_{3}$','SnO$_{2}$','Sb$_{2}$O$_{3}$','BaO','La$_{2}$O$_{3}$','CeO$_{2}$','PbO','Bi$_{2}$O$_{3}$','ANY_TEC','SPT','MOD_UNG','DENSITY']
fake_data_pd = pd.DataFrame(fake_data,columns=column)
de_standardized_data = inverse_standardize(fake_data_pd, data_min, data_max)
de_standardized_data['sum']=round(np.sum(de_standardized_data.iloc[:,0:-4],axis=1),2)
#print(de_standardized_data.columns)
de_standardized_data=de_standardized_data[(de_standardized_data.iloc[:,-5]>=33) & (de_standardized_data.iloc[:,-5]<=39)]
de_standardized_data=de_standardized_data[de_standardized_data.iloc[:,-4]>=720]
de_standardized_data=de_standardized_data[de_standardized_data.iloc[:,-3]>=80]
de_standardized_data=de_standardized_data[de_standardized_data.iloc[:,-2]<=2.6]
de_standardized_data=de_standardized_data[(de_standardized_data.iloc[:,-1]>=99.9) & (de_standardized_data.iloc[:,-1]<=100.1)]
print(len(de_standardized_data.iloc[:,-1]))
de_standardized_data.to_csv("generated_samples.csv")
