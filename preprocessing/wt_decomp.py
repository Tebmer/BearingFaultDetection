import matplotlib.pyplot as plt
import pywt
import pandas as pd
import csv
import numpy as np

#计算sigmoid熵的函数
def sigmoid(x):
    s = 1 / (1 + np.exp(-x).sum())
    return s

raw = pd.read_csv('E:\迅雷下载\轴承数据\\train.csv')
raw_data = raw.values
raw_feature = raw_data[0:,1:(raw_data.shape[1] - 1)]

entropy_total = []
for m in range(0, len(raw_feature)):
#for m in range(0,2):
    data = raw_feature[m]
    print('data' + str(data) + str(data.shape[0]))

    w = pywt.Wavelet('bior3.9')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))
    threshold = 0.1  # Threshold for filtering，可调整

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'bior3.9', level=maxlev)  # 将信号进行小波分解,或者用“bior3.9”

    entropy = []
    for i in range(0, len(coeffs)):
    # for i in range(0, 2):
        p = []
        sum = coeffs[i].sum()
        print("sum:" + str(sum))

        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

        print ('coeffs:')
        print (coeffs[i])

        #获取i层的每一个系数概率p
        for j in range(0, len(coeffs[i])):
            print("j:" + str(j) + '  i:' + str(i))
            print ('coeff[i][j]' +  str(coeffs[i][j]))
            p.append(coeffs[i][j]/sum)
            print("p %f" % p[j])

        print("sigmoid熵：")
        # 计算sigmoid熵
        s = 1 / (1 + (np.exp([-c for c in p])).sum())
        print(s)
        entropy.append(s)

    print(entropy)
    #获取完整的数据格式: id+features+label
    data_id = [m+1]
    data_label = [int(raw_data[m][6001])]
    data_total = data_id + entropy + data_label
    print('data_total: ' + str(data_total))
    entropy_total.append(data_total)
    # plt.figure()
    index = list(range(0, len(coeffs)))

    #画图，将图存到对应的0—9文件里。
    plt.title(str(raw_data[m][6001]))
    plt.plot(index, entropy)
    # plt.show()

    plt.savefig('./decomp1/' + str(int(raw_data[m][6001])) + '/id_' + str(m) +' label' + str(raw_data[m][6001]) + '.png')
    #plt.savefig()
    plt.clf()

# #将数据写入到csv文件里。
# with open('wt.csv', 'w+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     file_header = ['id'] + list(index) + ['label']
#     print(file_header)
#     writer.writerow(file_header) #写入表头
#     for row in entropy_total:
#         writer.writerow(row)
