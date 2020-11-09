import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k, value=None):
        """初始化knn分类器"""
        assert k >= 1, "k must be valid"
        self.k = k  # knn中的k
        self._x_train = None  # 训练数据集在类中，用户不能随意操作，故设置为私有
        self._y_train = None
        self._grc_cof = 0  # 灰色关联系数初始化
        self._value = value

    def fit(self, x_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert self.k <= x_train.shape[0], "the size of X_train must be at least k."
        self._x_train = x_train
        self._y_train = y_train
        return self  # 模仿sklearn，调用fit函数会返回自身

    def predict(self, x_predict):
        """给定待预测数据集X_predict,跟sklearn一样，要求用户传来的是数组格式的数据，
           返回表示X_predict的结果向量"""
        assert self._x_train is not None and self._y_train is not None,\
            "must fit before predict!"
        assert x_predict.shape[1] == self._x_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x, self._value) for x in x_predict]
        return np.array(y_predict)  # 返回的结果也遵循sklearn

    def _predict(self, x, value=None):
        """给定单个待预测的数据x,返回x_predict的预测结果值"""

        # 先判断x是合法的
        assert x.shape[0] == self._x_train.shape[1],\
            "the feature number of x must be equal to X_train"
        # 计算新来的数据与整个训练数据的距离
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._x_train]

        nearest = np.argsort(distances)  # 对距离排序并返回对应的索引
        if value is not None:
            for i in nearest[:self.k]:
                if distances[i] > value:
                    self.k = i
                    if i == 0:
                        self.k = 1
                    break
        topK_y = [self._y_train[i].tolist() for i in nearest[:self.k]]  # 返回最近的k个距离对应的分类
        topK_x = [self._x_train[i].tolist() for i in nearest[:self.k]]

        cof = self._grc(topK_x, x)
        self._grc_cof += np.sum(cof)
        # print("cof")
        # print(cof)
        sum_cof = np.sum(cof)
        return np.dot(cof, topK_y)/sum_cof if sum_cof != 0 else np.mean(topK_y)

    # 获得灰色关联系数(grey relation cofficient)，x为预测值，k_y为前k个值
    # 返回灰色关联系数array
    def _grc(self, k_x, x):
        k_x = np.array(k_x)
        x = np.array(x)
        min = float("inf")
        max = 0
        alpa = 0.5
        for j in range(k_x.shape[0]):
            temp = np.abs(x - k_x[j])
            max_temp = np.max(temp)
            min_temp = np.min(temp)
            if max_temp > max:
                max = max_temp
            if min_temp < min:
                min = min_temp
        cof = []
        for i in range(k_x.shape[0]):
            temp = []
            for j in range(k_x.shape[1]):
                if abs(k_x[i][j]-x[j])+alpa*max == 0:
                    temp.append(1)
                else:
                    temp.append((min+alpa*max)/(abs(k_x[i][j]-x[j])+alpa*max))
            cof.append(np.array(temp).mean())
        return np.array(cof)

    def get_grc_cof(self):
        return self._grc_cof

    def __repr__(self):
        return "KNN(k=%d)" % self.k
