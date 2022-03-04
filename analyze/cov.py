import matplotlib.pyplot as plt
import pywt
import pandas as pd
import numpy as np
import csv
import seaborn as sns

def test(df):
    dfData = df.corr()
    plt.subplots(figsize=(9, 9), nrows=2) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Red")
    plt.savefig('./BluesStateRelation.png')
    plt.show()

def autoNorm(dataSet):
    minVals = dataSet.min(0) # 取每一列的最小值
    maxVals = dataSet.max(0) # 取每一列的最大值
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

raw = pd.read_csv('train_features.csv')


print('Header:')
print(raw.columns.values)
raw_data = raw.values
raw_feature = raw_data[0:, 1:(raw_data.shape[1] - 1)]

raw_df = raw.iloc[0:, 1:(raw_data.shape[1] - 1)]
dfData = raw_df.corr()

plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="YlGnBu")
plt.savefig('./correlation.png')
plt.show()

raw_feature_norm, _, _ = autoNorm(raw_feature)
cov_matrix = np.cov(raw_feature_norm, rowvar=0)

plt.subplots(figsize=(9, 9)) # 设置画面大小
headers = list(raw.columns.values[1:raw_data.shape[1] - 1])
sns.heatmap(pd.DataFrame(cov_matrix, columns=headers, index=headers),  annot=True, square=True, cmap="YlGnBu")
plt.savefig('./covariance.png')
plt.show()

with open('cov_matrix.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # file_header = ['id'] + list(range(6001))[1:] + ['label']
    #writer.writerow(file_header) #写入表头
    for row in cov_matrix:
        writer.writerow(row)

# print(raw_feature.shape)