import matplotlib.pyplot as plt
import pywt
import pandas as pd
import csv
# Get data:
# ecg = pywt.data.ecg()  # 生成心电信号
# index = []
# data = []
# for i in range(len(ecg)-1):
#     X = float(i)
#     Y = float(ecg[i])
#     index.append(X)
#     data.append(Y)
# print(data)

raw = pd.read_csv('E:\迅雷下载\轴承数据\\train.csv')
raw_data = raw.values
raw_feature = raw_data[0:,1:6001] #不包括第6001个数据，其实是1~6000
data_filter = []
index = list(range(6000))
print(range(len(raw_feature)))
# for j in range(len(raw_feature)):
for j in range(10):
    data_id = [j+1]
    print(data_id)
    data = raw_feature[j]
    print(data)
    # Create wavelet object and define parameters
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))
    threshold = 0.1  # Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解,或者用“bior3.9”

    #plt.figure()
    for i in range(0, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    datarec_list = datarec.tolist()
    print(datarec_list)
    print('label: %d' % raw_data[j][6001])
    datarec_total = data_id + (datarec_list + [raw_data[j][6001]])
    print(datarec_total)
    data_filter.append(datarec_total)
print("length of data : %d" % len(data_filter))
print(data_filter[0])
with open('ultimate.csv', 'w+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    file_header = ['id'] + list(range(6001))[1:] + ['label']
    writer.writerow(file_header) #写入表头
    for row in data_filter:
        writer.writerow(row)

data = raw_feature[3]
datarec = data_filter[3][1:6001]
mintime = 0
maxtime = mintime + len(data) + 1

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime-1])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")

plt.tight_layout()
plt.show()
