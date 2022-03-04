import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df1 = pd.DataFrame(pd.read_csv('charfinal.csv'))
df = df1.drop(columns='id')
df = df.drop(columns='label')
pca = PCA(n_components=3)
pca.fit(df)
X_pca = pca.transform(df)
# print(X_pca)
X_pca_df = pd.DataFrame(X_pca)
# print(df['label'])
x_pic = pd.concat([X_pca_df,df1['label']], axis=1)
print(x_pic)
ax = Axes3D(fig=plt.figure())
# x_pic.to_csv('pca_allfea_n=2.csv',index=None)

clrs = ['red','orange','yellow','green','black','blue','purple','gray','pink','brown']
for i in range(0, 10):
    x = x_pic[x_pic['label'] == i][0]
    y = x_pic[x_pic['label'] == i][1]
    # plt.scatter(x, y, c=clrs[i], label=i)
    z = x_pic[x_pic['label']==i][2]
    ax.scatter(x, y, z, c=clrs[i], label=i)
plt.show()


# data = pd.DataFrame(pd.read_csv('C:/Users/huany/Desktop/train.csv'))
# df = pd.DataFrame(pd.read_csv('charfinal.csv'))
# df = df.sort_values(by='label') #sorted according to the label # 排序
#
# plt.scatter(df['label'],df['sf'])
#
# plt.show()
