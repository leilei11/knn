import sys
sys.path.append(r'D:\pycharmPro\markov')
import markovTraining
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # 取消keras加载后端出现的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1默认设置，为显示所有信息，2只显示error和warining信息，3只显示error
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, MaxPooling1D, Conv1D
import numpy as np
import pandas as pd
import defKnn
import math
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


# 利用列表来计算均方误差
def mean_absolute_error_list(y_true, y_pred):
    errors = []
    for i, value in enumerate(y_true):
        if value is None or value == 0.0:
            continue
        else:
            errors.append(np.abs(value - y_pred[i]) / value)
    return np.mean(errors)*100, errors*100


# 创建一个dataframe，其中data_dir为传入的dataframe路径，y_index为需要分离的标签
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
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


# 生成随机缺失数据，其中dataset为原数据，missing_ratio为缺失率，size为缺失段数量
# 返回缺失的data_cut以及真实的缺失率
def create_miss_data(dataset, missing_ratio, size=10):
    datacut = dataset.copy()
    sum = 0
    # missing_index = []
    np.random.seed(0)
    begin = int(np.random.randint(10, high=40, size=1))
    for _ in range(size):
        length = math.ceil(len(dataset) * missing_ratio / size * (1+np.random.uniform(-0.5, 0.5, 1)))
        sum += length
        end = math.ceil(len(dataset) / size * (1+np.random.uniform(-0.3, 0.5, 1))) + begin
        index = int(np.random.randint(begin, high=end, size=1))
        # print(index)
        datacut.iloc[index:index+length] = np.nan
        # missing_index.append(int(np.random.randint(begin, high=end, size=1)))
        begin = index + int(np.random.randint(length, high=length*1.5, size=1))
    return datacut, sum/len(dataset)


# 接受需要补充的数据data_test，返回二维list，包含nan的begin和end索引
def get_lost_index(data_test):
    list_nan = np.where(np.isnan(data_test))[0]
    begin_end_list = []
    list_all = []
    flag = True
    for i in range(len(list_nan)-1):
        if flag:
            begin_end_list.append(list_nan[i])
        if list_nan[i] == list_nan[i+1]-1:
            if i == len(list_nan)-2:
                begin_end_list.append(list_nan[i+1])
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


# 0.灰色自适应K-最近邻填补
def grey_auto_knn(data_cut, data_all, data_train, missing_ratio):
    n_neighbors = 3
    look_back = 5
    train_x, train_y = create_dataset(data_train.values, look_back)
    lost_index = get_lost_index(data_cut)
    # 利用自己的knn
    knn2 = defKnn.KNNClassifier(n_neighbors)
    knn2.fit(train_x, train_y)
    rmse = 0
    mre = 0
    sum = 0
    data_grey_auto_knn_fill = data_cut.copy()
    for i in range(len(lost_index)):
        for j in range(lost_index[i][0], lost_index[i][1] + 1):
            sum += 1
            look_back_x = np.array(data_grey_auto_knn_fill.values[j - look_back:j])
            look_back_x = look_back_x.reshape(1, -1)
            val = knn2.predict(look_back_x)
            data_grey_auto_knn_fill.loc[j:j + 1, 'y'] = val
            mre += abs((val - data_all.iloc[j]['y']) / data_all.iloc[j]['y'])
            rmse += pow(val - data_all.iloc[j]['y'], 2)

    mre = float(mre/sum)
    rmse = float(math.sqrt(rmse/sum))
    data_grey_auto_knn_fill.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data2\grey_auto_knn(ratio='+str(missing_ratio)+').csv')
    return mre, rmse


# 1.均值填充, 传入缺失的数据， 以及完整的数据，返回mre和rmse
def mean_fill(data_cut, data_all,  missing_ratio):
    lost_index = get_lost_index(data_cut)
    print(lost_index)
    mre = 0
    rmse = 0
    sum = 0
    data_mean_fill = data_cut.copy()
    for i in range(len(lost_index)):
        step = (data_cut.iloc[lost_index[i][1]+1]['y'] - data_cut.iloc[lost_index[i][0]-1]['y'])/(lost_index[i][1]-lost_index[i][0]+2)+data_cut.iloc[lost_index[i][0]-1]['y']
        for j in range(lost_index[i][0], lost_index[i][1] + 1):
            sum += 1
            val = data_cut.iloc[lost_index[i][0]-1]['y'] + step*(j-lost_index[i][0]+1)
            data_mean_fill.loc[j:j + 1, 'y'] = val
            mre += abs((val - data_all.iloc[j]['y']) / data_all.iloc[j]['y'])
            rmse += pow(val - data_all.iloc[j]['y'], 2)

    mre = float(mre/sum)
    rmse = float(math.sqrt(rmse/sum))
    data_mean_fill.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data2\mean(ratio=' + str(missing_ratio) + ').csv')
    return mre, rmse


# 2.马尔科夫填充
def markov_fill(data_cut, data_all, missing_ratio):
    lost_index = get_lost_index(data_cut)
    # print(lost_index)
    values = data_cut.values.flatten()
    mre = 0
    rmse = 0
    sum = 0
    data_markov_fill = data_cut.copy()
    for i in range(len(lost_index)):
        sum += lost_index[i][1] - lost_index[i][0] + 1
        predict_list = markovTraining.get_more_predict(data_train, values[lost_index[i][0] - len(markovTraining.status_values) + 1:lost_index[i][0]],
                                        lost_index[i][1] - lost_index[i][0] + 1)
        data_markov_fill.loc[lost_index[i][0]:lost_index[i][1] + 1, 'y'] = predict_list
        sub_arr = abs(np.array(predict_list) - np.array(data_all.iloc[lost_index[i][0]:lost_index[i][1]+1]['y']))
        mre += np.sum(sub_arr/np.array(predict_list))
        rmse += np.sum(sub_arr**2)

    mre = float(mre / sum)
    rmse = float(math.sqrt(rmse / sum))
    data_markov_fill.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data2\markov(ratio=' + str(missing_ratio) + ').csv')
    return mre, rmse


# 3.朴素knn
def knn_fill(data_cut, data_all, data_train, missing_ratio):
    look_back = 14
    lost_index = get_lost_index(data_cut)
    train_x, train_y = create_dataset(data_train.values, look_back)
    rmse = 0
    mre = 0
    sum = 0
    data_knn_fill = data_cut.copy()
    knn = KNeighborsRegressor(n_neighbors=8)
    knn.fit(train_x, train_y)
    for i in range(len(lost_index)):
        for j in range(lost_index[i][0], lost_index[i][1] + 1):
            sum += 1
            look_back_x = np.array(data_knn_fill.values[j - look_back:j])
            look_back_x = look_back_x.reshape(1, -1)
            val = knn.predict(look_back_x)
            data_knn_fill.loc[j:j + 1, 'y'] = val
            mre += abs((val - data_all.iloc[j]['y']) / data_all.iloc[j]['y'])
            rmse += pow(val - data_all.iloc[j]['y'], 2)
    mre = float(mre / sum)
    rmse = float(math.sqrt(rmse / sum))
    data_knn_fill.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data2\knn(ratio=' + str(missing_ratio) + ').csv')
    return mre, rmse


# 4.arima补充
def arima_fill(data_cut, data_all, data_train, missing_ratio):
    def test_stationarity(x):
        result = adfuller(x)
        print('ADF : %f' % result[0])
        print('p-value: %f' % result[1])
        print(result[4])
        for key, value in result[4].items():
            if result[0] > value:
                print("The data is non stationery")
            else:
                print("The data is stationery")
            break
    # data_diff = data_train.diff(1)
    # data_diff = data_diff.dropna()
    # print(data_train)
    rmse = 0
    mre = 0
    sum = 0
    data_train2 = None
    data_arima_fill = data_cut.copy()
    lost_index = get_lost_index(data_cut)
    for i in range(len(lost_index)):
        for j in range(lost_index[i][0], lost_index[i][1] + 1):
            sum += 1
            if lost_index[i][0]<len(data_train):
                data_train_2 = data_train.iloc[lost_index[i][0]:].append(data_arima_fill[:lost_index[i][0]])
            else:
                data_train_2 = data_arima_fill[lost_index[i][0]-len(data_train):lost_index[i][0]]
            arima_model = sm.tsa.ARIMA(data_train_2, order=(2, 1, 2), freq='min').fit(disp=-1)  # order=(p,d,q)
            predict_list = arima_model.forecast(steps=lost_index[i][1]-lost_index[i][0]+1)[0]
            data_arima_fill.loc[lost_index[i][0]:lost_index[i][1] + 1, 'y'] = predict_list
            sub_arr = abs(np.array(predict_list) - np.array(data_all.iloc[lost_index[i][0]:lost_index[i][1] + 1]['y']))
            mre += np.sum(sub_arr / np.array(predict_list))
            rmse += np.sum(sub_arr ** 2)
    mre = float(mre / sum)
    rmse = float(math.sqrt(rmse / sum))
    data_arima_fill.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data2\arima(ratio=' + str(missing_ratio) + ').csv')
    return mre, rmse

    # r, rac, Q = sm.tsa.acf(data_diff, qstat=True)
    # prac = pacf(data_diff, method='ywmle')
    # table_data = np.c_[range(1, len(r)), r[1:], rac, prac[1:len(rac) + 1], Q]
    # table = pd.DataFrame(table_data, columns=['lag', "AC", "Q", "PAC", "Prob(>Q)"])
    #
    # print(table)


# 获得不同缺失率下的mre, rmse
# missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
def get_all_error(missing_ratio, data_train, data_test):
    mre = []
    rmse = []
    for ratio in missing_ratio:
        data_cut, r = create_miss_data(data_test, ratio, 20)
        mean_mre, mean_rmse = mean_fill(data_cut, data_test, ratio)
        markov_mre, markov_rmse = markov_fill(data_cut, data_test, ratio)
        knn_mre, knn_rmse = knn_fill(data_cut, data_test, data_train, ratio)
        arima_mre, arima_rmse = arima_fill(data_cut, data_test, data_train, ratio)
        grey_auto_knn_mre, grey_auto_knn_rmse = grey_auto_knn(data_cut, data_test, data_train, ratio)
        mre.append([mean_mre, markov_mre, knn_mre, arima_mre, grey_auto_knn_mre])
        rmse.append([mean_rmse, markov_rmse, knn_rmse, arima_rmse, grey_auto_knn_rmse])

    return mre, rmse


# 根据不同的填补算法得到的结果来利用LSTM预测结果，当flag=true时，训练模型。当flag=false时，预测模型
def create_LSTM(file_dir_list, look_back=10):
    model_dir_list = []
    for file_dir in file_dir_list:
        data = pd.read_csv(file_dir).set_index('Date')
        dataset = data.values
        x_train, y_train = create_dataset(dataset, look_back=look_back)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(Conv1D(32, 5, padding='same', activation='relu', input_shape=(look_back, 1)))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(32, 5, padding='same', activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(32, 2, padding='same', activation='relu'))
        # model.add(Flatten())
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, activation='tanh',  return_sequences=True))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, activation='tanh',  return_sequences=True))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, activation='tanh',  return_sequences=False))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mse', optimizer='RMSProp', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=2)  # verbose=0表示不显示

        model_dir = 'save/model2/' + file_dir.split('\\')[-1].split('.csv')[0] + '_lookback=' + str(look_back) + '.h5'
        model.save(model_dir)
        model_dir_list.append(model_dir)
    return model_dir_list


# 用模型来预测, model_dir_list为模型存的地址，predict_data为需要预测的原始数据
def LSTM_predict(model_dir_list, data_predict, look_back=10):
    data_x, data_y = create_dataset(data_predict.values, look_back=look_back)
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))
    mre = []
    rmse = []
    predict_list = []
    for model_dir in model_dir_list:
        model = load_model(model_dir)
        predict = model.predict(data_x)
        predict_list.append(predict)
        # mre, _ = mean_absolute_error_list(data_y, data_predict)
        mre.append(mean_absolute_error_list(data_y, predict)[0])
        rmse.append(mean_squared_error(predict, data_y))
    return mre, rmse, predict_list


def show_data(ori_data_file, save_pic, range_tuple=None):
    plt.figure(figsize=(15, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.tick_params(labelsize=20)
    plt.xlabel('时刻/min', size=23)
    plt.ylabel('负荷值/A', size=23)
    plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
    plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白

    plt.xticks(rotation=15)
    ori_data = create_dataframe(ori_data_file, 'I')
    if range_tuple is None:
        plt.plot(ori_data)
    else:
        plt.plot(ori_data[range_tuple[0]: range_tuple[1]])
    plt.savefig(save_pic,
                format='png',
                bbox_inches='tight',
                transparent=True)
    plt.show()


if __name__ == '__main__':
    # show_data(r"D:\pycharmPro\markov\2019.8.9—9.19.csv", r'D:\SJTU\机器学习\小论文\数据补全\图\data_ori.png')

    complete_file = r"D:\pycharmPro\markov\2019.9.9-9.19(completed).csv"
    data_complete = create_dataframe(complete_file, 'I')

    lost_file = r"D:\pycharmPro\markov\2019.9.9-9.19(lost).csv"
    data_lost = create_dataframe(lost_file, 'I')

    show_data(lost_file, r'D:\SJTU\机器学习\小论文\数据补全\图\data_loss.png', (5300, 6500))

    missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
    # 分割训练数据
    data_train = data_complete.iloc[:5500]
    # data_test = data_complete.iloc[5500:6400]
    data_test = data_complete.iloc[5500:12700]
    data_predict = data_complete.iloc[12700:12700+800]

    # mre, rmse = get_all_error([0.28], data_train, data_test)
    # print(mre)
    # print(rmse)
    file_dir_list = ['D:\\SJTU\机器学习\\小论文\\数据补全\\data2\\grey_auto_knn(ratio=0.03).csv',
                     'D:\\SJTU\机器学习\\小论文\\数据补全\\data2\\arima(ratio=0.03).csv',
                     'D:\\SJTU\机器学习\\小论文\\数据补全\\data2\\knn(ratio=0.03).csv',
                     'D:\\SJTU\机器学习\\小论文\\数据补全\\data2\\markov(ratio=0.03).csv',
                     'D:\\SJTU\机器学习\\小论文\\数据补全\\data2\\mean(ratio=0.03).csv']
    model_dir_list = ['save/model2/grey_auto_knn(ratio=0.3)_lookback=10.h5', 'save/model2/arima(ratio=0.3)_lookback=10.h5',
                      'save/model2/knn(ratio=0.3)_lookback=10.h5', 'save/model2/markov(ratio=0.3)_lookback=10.h5',
                      'save/model2/mean(ratio=0.3)_lookback=10.h5']

    fill_label = ['GAKNN填补', 'ARIMA填补', 'KNN填补', '马尔科夫填补', '均值填补']
    style = ['-d', '-*', '-s', '-v', '-o']
    _, _, predict_list = LSTM_predict(model_dir_list, data_predict, 10)

    plt.figure(figsize=(30, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    str_index = []
    for i in range(len(fill_label)):
        ori_index = data_predict.iloc[len(data_predict)-len(predict_list[i]):].index
        str_index = [(str(d)).replace('T', ' ')[5:16] for d in list(ori_index.values)]
        # print(str_index)
        # print(list(np.array(predict_list[i]).reshape(-1,)))
        # plt.plot([(str(d)).replace('T', ' ')[5:16] for d in list(ori_index.values)], predict_list[i], label=fill_label[i])
        plt.plot(str_index, list(np.array(predict_list[i]).reshape(-1,)), label=fill_label[i])

    y_real = list(np.array(data_predict.values).reshape(-1,))
    plt.plot(str_index, y_real[len(y_real)-len(predict_list[0]):], label='原始数据')
    print([(str(d)).replace('T', ' ')[5:16] for d in list(data_predict.index.values)])
    print(list(np.array(data_predict.values).reshape(-1,)))
    plt.tick_params(labelsize=20)
    plt.xlabel('时刻', size=23)
    plt.ylabel('负荷值/A', size=23)
    plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
    plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(100))  # 设置刻度密度
    plt.xticks(rotation=30)
    plt.legend(loc='center', bbox_to_anchor=(0.6, 0.9), prop={'size': 20}, ncol=3, frameon=False)
    plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\自动化设备-图\LSTM_predict.png',
                format='png',
                bbox_inches='tight',
                transparent=True)
    plt.show()

    # look_back_result = []
    # mre_list = []
    # rmse_list = []
    # look_back_list = range(10, 11)
    # for look_back in look_back_list:
    #     print('-------------------begin: look_back = %d--------------------' % look_back)
    #     model_dir_list = create_LSTM(file_dir_list, look_back=look_back)
    #     mre, rmse, _ = LSTM_predict(model_dir_list, data_predict, look_back=look_back)
    #     look_back_result.append(look_back)
    #     mre_list.append(mre)
    #     rmse_list.append(rmse)
    #     print('-------------------finish: look_back = %d--------------------\n' % look_back)
    # print(file_dir_list[0].split('\\')[-1].split('(')[-1].split(')')[0])
    # print(look_back_result)
    # print(mre_list)
    # print(rmse_list)

    # missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
    # mre = [[2.076632569476165, 0.16149157164820033, 0.0984679712910605, 0.025759674189353134, 0.05957549285606621],
    #        [3.8090956810737397, 0.20354310581320645, 0.09006069726881974, 0.22405103767362428, 0.053432641711864154],
    #        [5.946868450676342, 0.2505463567086689, 0.09042030273627008, 0.47210593675260104, 0.066797429746322],
    #        [12.319546078919636, 0.13516806960360614, 0.09530446311845336, 0.9680219403789806, 0.056643637956478955],
    #        [17.631671687758658, 0.19490928603628202, 0.1358919026633745, 2.6067210419250952, 0.08790123350239175]]
    # rmse = [[45.945839441977824, 2.8081927247733667, 2.082145573969582, 0.9179511316018377, 1.9965446131443163],
    #         [88.60790535867982, 3.517253409059961, 2.0213539094287345, 2.8052788630021057, 2.0240821484663525],
    #         [148.63332489449448, 4.8880449293335095, 2.20934144428912, 4.663447600956625, 2.2275026022971947],
    #         [325.5544077804964, 3.023678986457206, 2.2294322638898634, 6.840232572696477, 2.1417220196027014],
    #         [479.7988833584617, 3.855068676586253, 3.4035295048725644, 14.699299080293272, 3.2947886098730246]]

    # for i in range(len(mre2)):
    #     mre2[i][2] = mre[i][0]
    #     mre2[i][-1] = mre[i][1]

    # fill_label = ['均值填补', '马尔科夫填补', 'KNN填补', 'ARIMA填补', '灰色自适应K-NN算法']
    # style = ['-o', '-v', '-s', '-*', '-d']

    '''
    包含所有的值
    '''
    # plt.figure(figsize=(16, 8))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
    # plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
    # for i in range(len(fill_label)):
    #     plt.plot(missing_ratio, [x[i] for x in mre], style[i], markersize=15, label=fill_label[i])
    # plt.tick_params(labelsize=23)
    # plt.ylim((0, 1))
    # plt.xlabel("缺失率", size=23)
    # plt.ylabel("MRE", size=23)
    # plt.legend(loc='upper right', prop={'size': 23}, ncol=1)  # ncol=n设为n列
    # plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\MRE.png',
    #             format='png',
    #             bbox_inches='tight',
    #             transparent=True)
    # plt.close()
    #
    # plt.figure(figsize=(16, 8))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
    # plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
    # for i in range(len(fill_label)):
    #     plt.plot(missing_ratio, [x[i] for x in rmse], style[i], markersize=15, label=fill_label[i])
    # plt.tick_params(labelsize=23)
    # plt.ylim((1.5, 8))
    # plt.xlabel("缺失率", size=23)
    # plt.ylabel("RMSE", size=23)
    # plt.legend(loc='upper right', prop={'size': 23}, ncol=1)
    # plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\RMSE.png',
    #             format='png',
    #             bbox_inches='tight',
    #             transparent=True)
    # plt.close()

    # '''
    # 不包含均值填补
    # '''
    # plt.figure(figsize=(16, 8))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
    # plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
    # for i in range(1, len(fill_label)):
    #     plt.plot(missing_ratio, [x[i] for x in mre], style[i], markersize=15, label=fill_label[i])
    # plt.tick_params(labelsize=23)
    # plt.xlabel("缺失率", size=23)
    # plt.ylabel("MRE", size=23)
    # plt.legend(loc='best', prop={'size': 23}, ncol=2)  # ncol=2设为2列
    # plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\MRE_no_mean.png',
    #             format='png',
    #             bbox_inches='tight',
    #             transparent=True)
    # plt.close()
    #
    # plt.figure(figsize=(16, 8))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
    # plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
    # for i in range(1, len(fill_label)):
    #     plt.plot(missing_ratio, [x[i] for x in rmse], style[i], markersize=15, label=fill_label[i])
    # plt.tick_params(labelsize=23)
    # plt.xlabel("缺失率", size=23)
    # plt.ylabel("RMSE", size=23)
    # plt.legend(loc='best', prop={'size': 23}, ncol=2)
    # plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\RMSE_no_mean.png',
    #             format='png',
    #             bbox_inches='tight',
    #             transparent=True)
    # plt.close()

