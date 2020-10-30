import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker

def _grc(x, _x_train):
    min = float("inf")
    max = 0
    alpa = 0.5
    for j in range(_x_train.shape[0]):
        temp = np.abs(np.array(x) - np.array(_x_train[j]))
        max_temp = np.max(temp)
        min_temp = np.min(temp)
        if max_temp > max:
            max = max_temp
        if min_temp < min:
            min = min_temp
    print("min")
    print(min)
    print("max")
    print(max)
    cof = []
    for i in range(_x_train.shape[0]):
        temp = []
        for j in range(_x_train.shape[1]):
            temp.append((min + alpa * max) / (abs(_x_train[i][j] - x[j]) + alpa * max))
        # print(temp)
        cof.append(np.array(temp).mean())
    return cof


x = [29.879999, 29.879999, 29.879999, 29.879999, 29.879999, 29.879999, 29.879999,
     29.879999, 29.879999, 29.879999, 29.879999, 29.879999, 29.879999, 29.879999]
x_train = [[29.76, 29.76, 29.76, 29.76, 29.76, 29.52, 29.52, 29.52, 29.52,
            29.52, 29.52, 29.52, 29.52, 29.52],
           [30.960001, 30.960001, 30.960001, 29.76, 29.76, 29.76,
            29.76, 29.76, 29.76, 29.76, 29.76, 29.76,
            29.76, 29.76],
           [30.960001, 30.960001, 30.960001, 30.960001, 29.76, 29.76,
            29.76, 29.76, 29.76, 29.76, 29.76, 29.76,
            29.76, 29.76],
           [29.76, 29.76, 29.76, 29.76, 29.52, 29.52,
            29.52, 29.52, 29.52, 29.52, 29.52, 29.52,
            29.52, 31.920002],
           [29.279999, 29.279999, 29.279999, 29.279999, 31.679998, 29.279999,
            29.52, 29.52, 29.52, 29.52, 29.639999, 29.639999,
            29.639999, 29.639999],
           [30.960001, 30.960001, 30.960001, 30.960001, 30.960001, 29.76,
            29.76, 29.76, 29.76, 29.76, 29.76, 29.76,
            29.76, 29.76],
           [29.279999, 29.279999, 29.279999, 29.279999, 29.279999, 31.679998,
            29.279999, 29.52, 29.52, 29.52, 29.52, 29.639999,
            29.639999, 29.639999],
           [32.16, 29.76, 29.76, 29.76, 29.76, 29.76, 29.52, 29.52, 29.52,
            29.52, 29.52, 29.52, 29.52, 29.52]]
x = np.array(x)
x_train = np.array(x_train)
print(x_train)
print(_grc(x, np.array(x_train)))

begin = np.random.randint(10, high=40, size=1)

print(int(begin+2))


# 以下为lookback为10
all_data_mre = [[3.533840551972389, 4.3134987354278564, 6.4364954829216, 5.025150626897812, 7.0522211492061615],
                [2.9076196253299713, 4.3967872858047485, 5.572567880153656, 3.445187956094742, 3.7285909056663513],
                [2.3042958229780197, 3.8314972072839737, 3.130148723721504, 2.2059855982661247, 4.983729124069214],
                [2.6708807796239853, 2.920106239616871, 4.120416939258575, 5.96974715590477, 8.08897316455841],
                [3.1243911013007164, 5.3427837789058685, 3.0822990462183952, 6.7419275641441345, 6.409325450658798]]
all_data_rmse = [[0.8951619239076275, 1.224562525435362, 3.119847247737196, 1.9796240936896066, 2.683046633814922],
                 [0.6186719931525703, 1.5534944448295296, 1.6127089689672007, 0.7714026079221378, 0.9392889511190258],
                 [0.4870453595921553, 1.0035628602813904, 0.7388996193972254, 0.5965575963664826, 1.437172106715925],
                 [0.5828246245821712, 0.6362502890719673, 1.1978452094565313, 2.0976767935051406, 4.737500695649613],
                 [0.7115892226207783, 1.4192416306965063, 0.665054489434959, 2.600067942897122, 2.094776597808432]]

# 以下为lookback为6
# all_data_mre = [[3.235149011015892, 3.0865345150232315, 3.0782967805862427, 5.692587047815323, 5.510665476322174],
#                 [3.5067372024059296, 5.357145518064499, 4.792933538556099, 3.4127939492464066, 4.596593603491783],
#                 [3.0485251918435097, 3.7895947694778442, 4.779594764113426, 2.271895855665207, 2.9399655759334564],
#                 [3.3850111067295074, 3.3536482602357864, 3.8267944008111954, 1.5956588089466095, 4.548113793134689],
#                 [4.612954705953598, 3.699292242527008, 5.474206805229187, 5.560799688100815, 4.33211475610733]]
# all_data_rmse = [[0.7183761642204055, 0.7560749636114413, 0.716584363087497, 2.098811701088089, 2.008388958953164],
#                  [0.8086247665046851, 1.6396383552884177, 1.3353311815257936, 0.8925041520483656, 1.0719855363779598],
#                  [0.7560580259619573, 0.9084925078498783, 1.448114208991964, 0.4845423493524957, 0.6168664731162062],
#                  [0.7885555185628185, 0.7901565660966058, 0.9027218369458424, 0.4026037342429724, 1.2978655561323478],
#                  [1.3748342206616704, 0.8737881523705168, 1.6016061239350208, 2.1938269112549653, 1.0861916083430605]]

missing_ratio = [0.03, 0.06, 0.1, 0.2, 0.3]
fill_label = ['GAKNN填补', 'ARIMA填补', 'KNN填补', '马尔科夫填补', '均值填补']
style = ['-d', '-*', '-s', '-v', '-o']

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

plt.figure(0, figsize=(30, 9))
for i in range(len(fill_label)):
    plt.plot(missing_ratio, [x[i]/100 for x in all_data_mre], style[i], color='black', linewidth=4, markersize=30, label=fill_label[i])
plt.tick_params(labelsize=60)
plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白
plt.xlabel('缺失率', size=80)
plt.ylabel('${E_{MAPE\_LSTM}}$', size=80)
plt.xlim(right=0.302)
plt.ylim(top=0.09)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.03))  # 设置刻度密度
ax = plt.gca()
ax.spines['bottom'].set_linewidth(4)   # 设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(4)   # 设置左边坐标轴的粗细
ax.tick_params(width=4)  # 设置刻度线的粗细（竖着的）
# bbox_to_anchor=(num1, num2)，第一个向右移动，第二个向上移动，ncol=5表示5列，frameon=False除去方框
plt.legend(loc='center', bbox_to_anchor=(0.352, 0.87), prop={'size': 48}, ncol=2, frameon=False)
plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\自动化设备-图\LSTM_MAPE.png',
            format='png',
            bbox_inches='tight',
            transparent=True)

# plt.figure(1, figsize=(30, 9))
# for i in range(len(fill_label)):
#     plt.plot(missing_ratio, [x[i] for x in all_data_rmse], style[i], color='black', linewidth=4, markersize=30, label=fill_label[i])
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.tick_params(labelsize=60)
# plt.autoscale(enable=True, axis='x', tight=True)  # 去掉坐标边缘的留白
# plt.autoscale(enable=True, axis='y', tight=True)  # 去掉坐标边缘的留白
# plt.xlabel('缺失率', size=80)
# plt.ylabel('${E_{RMSE\_LSTM}}$/A', size=80)
# plt.xlim(right=0.302)
# plt.ylim(top=5)
# plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2.5))  # 设置刻度密度
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(4)   # 设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(4)   # 设置左边坐标轴的粗细
# ax.tick_params(width=4)  # 设置刻度线的粗细（竖着的）
# # bbox_to_anchor=(num1, num2)，第一个向右移动，第二个向上移动，ncol=5表示5列，frameon=False除去方框
# plt.legend(loc='center', bbox_to_anchor=(0.2, 0.8), prop={'size': 48}, ncol=1, frameon=False)
# plt.savefig(r'D:\SJTU\机器学习\小论文\数据补全\图\自动化设备-图\LSTM_RMSE.png',
#             format='png',
#             bbox_inches='tight',
#             transparent=True)
plt.show()
