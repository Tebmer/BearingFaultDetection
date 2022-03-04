from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train(file_name, m):
    ave_acc = []
    clf = RandomForestClassifier(n_estimators=(250))
    for i in range(0, 5):
        raw = pd.read_csv(file_name)
        raw_data = raw.values
        raw_feature = raw_data[0:, 1:(raw_data.shape[1] - 1)]
        label = raw['label'].values

        x_train, x_val, y_train, y_val = train_test_split(raw_feature, label, test_size=0.3)
        clf.fit(x_train, y_train)
        print(confusion_matrix(y_val, clf.predict(x_val)))
        print(f1_score(y_val, clf.predict(x_val), average='macro'))
        ave_acc.append(f1_score(y_val, clf.predict(x_val), average='macro'))
    print('average accuracy of ' + file_name + str(np.mean(ave_acc)))
    return np.mean(ave_acc)
    # print('average accuracy of ' + file_name + str(np.mean(ave_acc)))

acc_time_freq = []
acc_wpt = []
acc_time_freq_wpt = []
acc_time = []
acc_wt = []
times = 1
file_name_1 = './train_features.csv'
file_name_2 = './wpt.csv'
file_name_3 = './时频小波PCA_50_train.csv'
file_name_4 = './charfinal.csv'
#file_name_5 = './wt.csv'

for m in range(0, times):

    # acc_time_freq.append(train(file_name_1, m))

    # acc_wpt.append(train(file_name_2, m))

    acc_time_freq_wpt.append(train(file_name_3, m))

    # acc_time.append(train(file_name_4, m))

    #acc_wt.append(train(file_name_5, m))

# index = [element*50 for element in list(range(1, times))]
# # plt.plot(index, acc_time_freq, label='time+freq')
# # plt.plot(index, acc_wpt, label='wpt')
# plt.plot(index, acc_time_freq_wpt, label='time+freq+wpt')
# # plt.plot(index, acc_time, label='time')
# #plt.plot(index, acc_wt, label='wt')
# plt.legend(loc=0)
# plt.xlabel('the number of estimators')
# plt.ylabel('accuracy')
# plt.title('Classifier: Random Forest')
# plt.savefig('comparison.png')
# plt.show()
# plt.close()