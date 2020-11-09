from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import defKnn
import math
from datetime import datetime

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
    '''

    :param data_test:
    :return:
    '''
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


def get_k_fill():
    complete_file = r"file\2019.9.9-9.19(completed).csv"
    data_complete = create_dataframe(complete_file, 'I')

    lost_file = r"file\2019.9.9-9.19(lost).csv"
    data_lost = create_dataframe(lost_file, 'I')

    look_back = 13  # 14
    missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
    # missing_ratio = [0.03]
    # 分割训练数据
    data_train = data_complete.iloc[:5500]
    data_test = data_complete.iloc[5500:6400]
    train_x, train_y = create_dataset(data_train.values, look_back)

    df_mre = pd.DataFrame(index=range(3, 21))
    df_rmse = pd.DataFrame(index=range(3, 21))
    rmse_all = []
    mre_all = []
    k_list = list(range(4, 21))
    for ratio in missing_ratio:
        rmse_list = []
        mre_list = []
        data_fill_plt = None
        for n_neighbors in k_list:
            data_cut, r = create_miss_data(data_test, ratio, 10)
            # print(r)

            # 数据分割，得到数据缺失部分
            # data_cut = data_lost.iloc[5500:6400]
            values = data_cut.values.flatten()
            lost_index = get_lost_index(data_cut)
            # print(lost_index)

            # 利用自己的knn
            knn2 = defKnn.KNNClassifier(n_neighbors)
            knn2.fit(train_x, train_y)

            rmse = 0
            mre = 0
            sum = 0
            data_fill = data_cut.copy()
            for i in range(len(lost_index)):
                for j in range(lost_index[i][0], lost_index[i][1]+1):
                    sum += 1
                    look_back_x = np.array(data_fill.values[j-look_back:j])
                    look_back_x = look_back_x.reshape(1, -1)
                    # print(look_back_x)
                    # val = knn.predict(look_back_x)
                    val = knn2.predict(look_back_x)
                    data_fill.loc[j:j+1, 'y'] = val
                    mre += abs((val-data_test.iloc[j]['y'])/data_test.iloc[j]['y'])
                    rmse += pow(val-data_test.iloc[j]['y'], 2)

            mre_list.append(float(mre / sum))
            rmse_list.append(float(math.sqrt(rmse/sum)))
            # k_list.append(n_neighbors)
            if n_neighbors == 6:
                data_fill_plt = data_fill.copy()
        if len(rmse_all) == 0:
            rmse_all = rmse_list
        else:
            rmse_all = [i + j for i, j in zip(rmse_all, rmse_list)]
        # plt.figure(0, figsize=(18, 9))
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        # plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
        # plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
        # plt.plot([(str(d)).replace('T', ' ')[5:16] for d in list(data_test.index.values)], data_test['y'].values, "black", linewidth=4, linestyle='-', label='真实值')
        # plt.plot([(str(d)).replace('T', ' ')[5:16] for d in list(data_fill_plt.index.values)], data_fill_plt['y'].values, "black", linewidth=2, linestyle='--', label='填补值')
        # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))  # 设置刻度密度
        # plt.tick_params(labelsize=60)
        # plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
        # plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白
        # ax = plt.gca()
        # ax.spines['bottom'].set_linewidth(4)   # 设置底部坐标轴的粗细
        # ax.spines['left'].set_linewidth(4)   # 设置左边坐标轴的粗细
        # ax.tick_params(width=4)  # 设置刻度线的粗细（竖着的）
        # plt.xticks(rotation=30)
        # plt.xlabel("时刻", size=80)
        # plt.ylabel("真实值/A,填补值/A", size=80)
        # plt.legend(loc='upper center', prop={'size': 60}, bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)  # ncol=n设为n列
        # plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\自动化设备-图\fill'+'(ratio='+str(ratio)+').png',
        #             format='png',
        #             bbox_inches='tight',
        #             transparent=True)
        # plt.show()
        # plt.close()

        # plt.figure(1, figsize=(12, 6))
        # plt.plot(data_cut.index, data_cut['y'], label='miss')
        # plt.figure(2, figsize=(12, 6))
        # plt.plot(data_test.index, data_test['y'], label='real')
        # plt.legend()
        # plt.show()

        # print(mre_list)
        # df_mre['ratio=' + str(ratio)] = mre_list
        # df_rmse['ratio=' + str(ratio)] = rmse_list
        # df_mre.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data_MRE.csv')
        # df_rmse.to_csv(r'D:\SJTU\机器学习\小论文\数据补全\data_RMSE.csv')
        #
        # plt.figure(0, figsize=(16, 8))
        # plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
        # plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
        # plt.tick_params(labelsize=23)
        # plt.plot(np.array(k_list).astype(dtype=np.str), mre_list)
        # plt.xlabel("K", size=23)
        # plt.ylabel("MRE", size=23)
        # # plt.legend()
        # plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\缺失率='+str(ratio)+'_MRE.png',
        #             format='png',
        #             bbox_inches='tight',
        #             transparent=True)
        # plt.close()
        # print('缺失率='+str(ratio)+'_MRE.png 已完成')

        #  以下为RMSE的图形展示
        plt.figure(figsize=(15, 9))
        plt.plot(np.array(k_list).astype(dtype=np.str), rmse_list, 'black')
        plt.xlabel("K值", size=23)
        plt.ylabel('${E_{RMSE}}$/A', size=23)
        # plt.ylim(1.95, 2.10)  # 缺失率=0.03
        # plt.ylim(1.98, 2.30)  # 缺失率=0.06
        # plt.ylim(2.15, 2.30)  # 缺失率=0.1
        # plt.ylim(3.00, 3.50)  # 缺失率=0.3
        # plt.xticks(range(2, 20, 3))
        ax = plt.gca()
        # ax为两条坐标轴的实例
        # ax.spines['bottom'].set_linewidth(4)   # 设置底部坐标轴的粗细
        # ax.spines['left'].set_linewidth(4)   # 设置左边坐标轴的粗细
        # ax.xaxis.set_major_locator(MultipleLocator(3))
        # ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 缺失率=0.03/0.1
        # ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 缺失率=0.06
        # ax.yaxis.set_major_locator(MultipleLocator(0.25))  # 缺失率=0.3
        # plt.show()
        # fig = plt.gcf()
        # plt.savefig(r'picture\(K)缺失率='+str(ratio)+'_RMSE.png',
        #             format='png',
        #             bbox_inches='tight',
        #             transparent=True)
        # plt.show()
        # plt.close()
        print('缺失率=' + str(ratio) + '_RMSE.png 已完成')

    plt.figure(figsize=(15, 9))
    plt.tick_params(labelsize=23)
    plt.plot(np.array(k_list).astype(dtype=np.str), rmse_all, 'black')
    plt.xlabel("K值", size=23)
    plt.ylabel('${E_{RMSE}}$/A', size=23)
    plt.savefig(r'picture\(K)缺失率汇总_RMSE.png',
                format='png',
                bbox_inches='tight',
                transparent=True)


def get_lookback_fill():
    complete_file = r"file\2019.9.9-9.19(completed).csv"
    data_complete = create_dataframe(complete_file, 'I')

    lost_file = r"file\2019.9.9-9.19(lost).csv"
    data_lost = create_dataframe(lost_file, 'I')

    missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
    # missing_ratio = [0.03]
    # 分割训练数据
    data_train = data_complete.iloc[:5500]
    data_test = data_complete.iloc[5500:6400]

    df_mre = pd.DataFrame(index=range(3, 21))
    df_rmse = pd.DataFrame(index=range(3, 21))
    look_back_list = list(range(10, 45))

    for ratio in missing_ratio:
        rmse_list = []
        mre_list = []
        k_list = []
        data_fill_plt = None
        # 利用自己的knn
        for look_back in look_back_list:
            train_x, train_y = create_dataset(data_train.values, look_back)
            data_cut, r = create_miss_data(data_test, ratio, 10)
            # print(r)

            # 数据分割，得到数据缺失部分
            # data_cut = data_lost.iloc[5500:6400]
            values = data_cut.values.flatten()
            lost_index = get_lost_index(data_cut)

            # 利用自己的knn
            knn2 = defKnn.KNNClassifier(7)
            knn2.fit(train_x, train_y)

            rmse = 0
            mre = 0
            sum = 0
            data_fill = data_cut.copy()
            for i in range(len(lost_index)):
                for j in range(lost_index[i][0], lost_index[i][1]+1):
                    sum += 1
                    look_back_x = np.array(data_fill.values[j-look_back:j])
                    look_back_x = look_back_x.reshape(1, -1)
                    # print(look_back_x)
                    # val = knn.predict(look_back_x)
                    val = knn2.predict(look_back_x)
                    data_fill.loc[j:j+1, 'y'] = val
                    mre += abs((val-data_test.iloc[j]['y'])/data_test.iloc[j]['y'])
                    rmse += pow(val-data_test.iloc[j]['y'], 2)

            mre_list.append(float('%.2f' % (mre / sum)))
            rmse_list.append(float('%.2f' % math.sqrt(rmse/sum)))
            data_fill_plt = data_fill.copy()

        plt.figure(figsize=(15, 9))
        plt.plot(np.array(look_back_list), rmse_list, 'black')
        plt.xlabel("输入特征维度", size=23)
        plt.ylabel('${E_{RMSE}}$/A', size=23)
        # plt.ylim(5.17, 5.19)  # 缺失率=0.03
        # plt.ylim(1.98, 2.30)  # 缺失率=0.06
        # plt.ylim(2.15, 2.30)  # 缺失率=0.1
        # plt.ylim(5.17, 5.19)  # 缺失率=0.3
        # plt.xticks(range(2, 20, 3))
        ax = plt.gca()
        # ax为两条坐标轴的实例
        # ax.spines['bottom'].set_linewidth(2)   # 设置底部坐标轴的粗细
        # ax.spines['left'].set_linewidth(2)   # 设置左边坐标轴的粗细
        # ax.xaxis.set_major_locator(MultipleLocator(3))
        # ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 缺失率=0.03/0.1
        # ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 缺失率=0.06
        # ax.yaxis.set_major_locator(MultipleLocator(0.25))  # 缺失率=0.3
        # plt.show()
        # fig = plt.gcf()
        plt.savefig(r'picture\(输入维度)缺失率='+str(ratio)+'_RMSE.png',
                    format='png',
                    bbox_inches='tight',
                    transparent=True)
        # plt.show()
        # plt.close()
        print('(输入维度)缺失率=' + str(ratio) + '_RMSE.png 已完成')


def get_value_fill():
    complete_file = r"file\2019.9.9-9.19(completed).csv"
    data_complete = create_dataframe(complete_file, 'I')

    lost_file = r"file\2019.9.9-9.19(lost).csv"
    data_lost = create_dataframe(lost_file, 'I')

    look_back = 13  # 14
    missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
    # missing_ratio = [0.03]
    # 分割训练数据
    data_train = data_complete.iloc[:5500]
    data_test = data_complete.iloc[5500:6400]
    train_x, train_y = create_dataset(data_train.values, look_back)

    df_mre = pd.DataFrame(index=range(3, 21))
    df_rmse = pd.DataFrame(index=range(3, 21))
    value_list = list(np.arange(0.15, 0.6, 0.05))

    for ratio in missing_ratio:
        rmse_list = []
        mre_list = []
        k_list = []
        data_fill_plt = None
        # 利用自己的knn
        for value in value_list:
            data_cut, r = create_miss_data(data_test, ratio, 10)
            # print(r)

            # 数据分割，得到数据缺失部分
            # data_cut = data_lost.iloc[5500:6400]
            values = data_cut.values.flatten()
            lost_index = get_lost_index(data_cut)

            # 利用自己的knn
            knn2 = defKnn.KNNClassifier(7, value)
            knn2.fit(train_x, train_y)

            rmse = 0
            mre = 0
            sum = 0
            data_fill = data_cut.copy()
            for i in range(len(lost_index)):
                for j in range(lost_index[i][0], lost_index[i][1]+1):
                    sum += 1
                    look_back_x = np.array(data_fill.values[j-look_back:j])
                    look_back_x = look_back_x.reshape(1, -1)
                    # print(look_back_x)
                    # val = knn.predict(look_back_x)
                    val = knn2.predict(look_back_x)
                    data_fill.loc[j:j+1, 'y'] = val
                    mre += abs((val-data_test.iloc[j]['y'])/data_test.iloc[j]['y'])
                    rmse += pow(val-data_test.iloc[j]['y'], 2)

            mre_list.append(float('%.2f' % (mre / sum)))
            rmse_list.append(float('%.2f' % math.sqrt(rmse/sum)))
            data_fill_plt = data_fill.copy()

        plt.figure(figsize=(15, 9))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
        plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
        plt.tick_params(labelsize=23)
        plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
        plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白
        plt.plot(np.array(value_list), rmse_list, 'black')
        plt.xlabel("阈值", size=23)
        plt.ylabel('${E_{RMSE}}$/A', size=23)
        # plt.ylim(5.17, 5.19)  # 缺失率=0.03
        # plt.ylim(1.98, 2.30)  # 缺失率=0.06
        # plt.ylim(2.15, 2.30)  # 缺失率=0.1
        # plt.ylim(5.17, 5.19)  # 缺失率=0.3
        # plt.xticks(range(2, 20, 3))
        ax = plt.gca()
        # ax为两条坐标轴的实例
        # ax.spines['bottom'].set_linewidth(2)   # 设置底部坐标轴的粗细
        # ax.spines['left'].set_linewidth(2)   # 设置左边坐标轴的粗细
        # ax.xaxis.set_major_locator(MultipleLocator(3))
        # ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 缺失率=0.03/0.1
        # ax.yaxis.set_major_locator(MultipleLocator(0.1))  # 缺失率=0.06
        # ax.yaxis.set_major_locator(MultipleLocator(0.25))  # 缺失率=0.3
        # plt.show()
        # fig = plt.gcf()
        plt.savefig(r'picture\(自适应)缺失率='+str(ratio)+'_RMSE.png',
                    format='png',
                    bbox_inches='tight',
                    transparent=True)
        # plt.show()
        # plt.close()
        print('(自适应)缺失率=' + str(ratio) + '_RMSE.png 已完成')


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.spines.top'] = False  # 去掉顶部轴，必须放在plot之前
    plt.rcParams['axes.spines.right'] = False  # 去掉右部轴
    plt.tick_params(labelsize=23)
    plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
    plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白

    get_k_fill()
    # get_value_fill()
    # get_lookback_fill()