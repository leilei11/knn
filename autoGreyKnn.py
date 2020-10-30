from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import defKnn


# 创建一个dataframe，其中data_dir为传入的dataframe，y_index为需要分离的标签
# 返回创建的新dataframe，index为'Date'，第一列为'y'
def create_dataframe(data_dir, y_index):
    data = pd.read_csv(data_dir, encoding='utf8')
    data['Time'] = pd.to_datetime(data['Time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')
    # data['Time'] = pd.to_datetime(data['Time'])
    data = data.sort_values("Time")

    new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'y'])
    new_data['Date'] = data['Time'].values
    new_data['y'] = data[y_index].values
    new_data.set_index('Date', drop=True, inplace=True)
    return new_data


# 其中dataset为时间序列数据， look_back为回溯的长度
# 返回时间窗口，其中data_x作为输入特征量，data_y为输出量
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


# 接受需要补充的数据data_test，返回二维list，包含nan的begin和end索引
def get_lost_index(data_test):
    list_nan = np.where(np.isnan(data_test))[0]
    begin_end_list = []
    list_all = []
    flag = True
    for i in range(len(list_nan) - 1):
        if flag:
            begin_end_list.append(list_nan[i])
        if list_nan[i] == list_nan[i + 1] - 1:
            if i == len(list_nan) - 2:
                begin_end_list.append(list_nan[i + 1])
                list_all.append(begin_end_list)
                break
            flag = False
            continue
        else:
            flag = True
            begin_end_list.append(list_nan[i])
            list_all.append(list.copy(begin_end_list))
            begin_end_list.clear()
    return list_all


complete_file = r"D:\pycharmPro\markov\2019.9.9-9.19(completed).csv"
data_complete = create_dataframe(complete_file, 'I')

lost_file = r"D:\pycharmPro\markov\2019.9.9-9.19(lost).csv"
data_lost = create_dataframe(lost_file, 'I')

# 数据分割，得到数据缺失部分
data_cut = data_lost.iloc[5500:6400]
values = data_cut.values.flatten()
lost_index = get_lost_index(data_cut)
print(lost_index)

# 分割训练数据
data_train = data_complete.iloc[:5500]
data_test = data_complete.iloc[5500:6400]

data_fill = data_cut.copy()
data_fill_all = data_cut.copy()
for i in range(len(lost_index)):
    print("===================No.%d missing data====================" % (i + 1))
    grc_cof_max = 0
    k_best = 0
    look_back_best = 0
    for look_back in range(5, 15):
        train_x, train_y = create_dataset(data_train.values, look_back)
        for n_neighbors in range(3, 10):
            print("------look_back = %d, k= %d-------" % (look_back, n_neighbors))
            sum = 0
            # 利用自己的knn
            knn2 = defKnn.KNNClassifier(n_neighbors)
            knn2.fit(train_x, train_y)

            for j in range(lost_index[i][0], lost_index[i][1] + 1):
                sum += 1
                look_back_x = np.array(data_fill.values[j - look_back:j])
                look_back_x = look_back_x.reshape(1, -1)
                val2 = knn2.predict(look_back_x)
                data_fill.loc[j:j + 1, 'y'] = val2

            temp = knn2.get_grc_cof() / n_neighbors / sum

            if not np.isnan(temp) and temp > grc_cof_max:
                grc_cof_max = temp
                k_best = n_neighbors
                look_back_best = look_back
            print("Now, No.%d: best look_back = %d, best k = %d\n" % (i + 1, look_back_best, k_best))
    print("No.%d: k_best = %d, look_back_best =%d\n" % (i + 1, k_best, look_back_best))

    train_x, train_y = create_dataset(data_train.values, look_back_best)
    # 利用自己的knn以及k_best以及look_back_best
    knn2 = defKnn.KNNClassifier(k_best)
    knn2.fit(train_x, train_y)
    for j in range(lost_index[i][0], lost_index[i][1] + 1):
        look_back_x = np.array(data_fill_all.values[j - look_back_best:j])
        look_back_x = look_back_x.reshape(1, -1)
        val = knn2.predict(look_back_x)
        data_fill_all.loc[j:j + 1, 'y'] = val
print("=================================end===============================")

plt.figure(0, figsize=(12, 6))
plt.plot(data_test.index, data_test['y'], label='real')
plt.plot(data_fill_all.index, data_fill_all['y'], label='auto-fill')
plt.legend()

plt.figure(1, figsize=(12, 6))
plt.plot(data_cut.index, data_cut['y'], label='missing')
plt.legend()

plt.figure(2, figsize=(12, 6))
plt.plot(data_test.index, data_test['y'], label='real')
plt.legend()
plt.show()

