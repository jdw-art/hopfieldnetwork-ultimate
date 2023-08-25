from __future__ import print_function, division
import numpy as np


class HopfieldNetwork:
    def __init__(self, N=100, filepath=None):
        '''
        初始化
        :param N: 网络中神经元个数
        :param filepath: 读取文件地址
        '''
        if not filepath:  # create new hopfield network with N neurons
            self.initialize_new_network(N)
        else:  # load hopfield network from file
            self.load_network(filepath)

    def initialize_new_network(self, N):
        '''
        初始化网络
        :param N: 神经元个数
        :return:
        '''
        self.N = N  # number of neurons
        self.w = np.zeros((N, N))  # weight matrix
        # 保存已经记忆过的模式
        self.xi = np.empty((N, 0), dtype="int8")  # array with saved patterns
        # 神经元状态，初始值都为-1，表示当前神经元未被激活
        self.S = -1 * np.ones(N, dtype="int8")  # state of the neurons
        self.p = 0  # number of saved patterns
        self.t = 0  # time steps

    def load_network(self, filepath):
        '''
        从文件中加载网络
        :param filepath:
        :return:
        '''
        npzfile = np.load(filepath)
        self.w = npzfile["arr_0"]
        self.xi = npzfile["arr_1"]
        self.N = self.w.shape[0]
        self.S = -1 * np.ones(self.N, dtype="int8")
        self.p = self.xi.shape[1]
        self.t = 0

    def save_network(self, filepath):
        '''
        保存网络参数
        :param filepath:
        :return:
        '''
        np.savez(filepath, self.w, self.xi)

    def train_pattern(self, input_pattern):
        '''
        训练单个模式并更新网络，根据输入模式更新网络
        :param input_pattern:
        :return:
        '''
        # 构建新的Hebb权重矩阵，用于更新网络权重
        self.w += construct_hebb_matrix(input_pattern)
        # 将输入的模式保存到已训练的模式数组
        self.xi = np.column_stack((self.xi, input_pattern))
        # 根据更新后的模式数组的形状，更新已保存的模式数量
        self.p = self.xi.shape[1]

    def remove_pattern(self, i):
        '''
        从网络中删除已保存的模式
        :param i: 将要删除的模式的索引
        :return:
        '''
        if i < self.p:
            # 根据指定索引的模式构建出 Hebbian 权重矩阵的一部分，并将其从原有的权重矩阵 self.w 中减去
            self.w -= construct_hebb_matrix(self.xi[:, i])
            # 从已保存的模式数组中删除指定索引的模式列
            self.xi = np.delete(self.xi, i, axis=1)
            # 根据更新后的模式数组的形状，更新已保存的模式数量
            self.p = self.xi.shape[1]
        else:
            print("There is no pattern to remove!")

    def set_initial_neurons_state(self, S_initial):  # uses S_initial in place
        '''
        设置初始化神经元状态
        :param S_initial:
        :return:
        '''
        if len(S_initial.shape) != 1 or S_initial.shape[0] != self.N:
            # S_initial的形状不是一维或者S_initial的长度和网络中神经元个数不同
            raise ValueError(
                "Unexpected shape/size of initial neuron state: {}".format(
                    S_initial.shape
                )
            )
        self.t = 0  # reset timer1 for new initial state vector
        self.S = S_initial  # set new initial neuron state

    def update_neurons(self, iterations, mode, run_max=False):
        '''
        更新神经元状态
        :param iterations: 迭代次数
        :param mode: 同步更新或者异步更新
        :param run_max: 表示是否运行到状态不再改变为止
        :return:
        '''
        self.t += iterations    # 将迭代次数添加到时间步长 t 上，表示进行了多少个时间步
        if mode == "async":     # 异步更新
            for _ in range(iterations):
                for i in np.random.permutation(self.N):  # semi-random 在神经元数量范围内，以半随机的顺序遍历神经元
                    self.S[i] = sign_0(np.dot(self.w[i, :], self.S))    # 更新神经元状态，超过阈值为1否则为-1
            if run_max:
                while True:
                    last_S = np.copy(self.S)        # 复制当前神经元状态
                    for i in np.random.permutation(self.N):  # semi-random 在神经元数量范围内，以半随机的顺序遍历神经元
                        self.S[i] = sign_0(np.dot(self.w[i, :], self.S))    # 更新神经元状态，超过阈值为1否则为-1
                    if np.array_equal(last_S, self.S):  # 检查上一次的神经元状态和当前状态是否相等，如果相等表示状态不再改变
                        return
                    self.t += 1

        elif mode == "sync":    # 同步更新
            for _ in range(iterations):
                # 通过将权重矩阵与神经元状态进行矩阵乘法，并通过 sign_0 函数取值，更新所有神经元的状态
                self.S = sign_0(np.dot(self.w, self.S))
            if run_max:
                while True:
                    second_last_S = np.copy(self.S)
                    for i in range(2):              # 进行两次循环，用于检查状态的振荡情况
                        last_S = np.copy(self.S)
                        self.S = sign_0(np.dot(self.w, self.S))
                        if np.array_equal(last_S, self.S):
                            return
                        self.t += 1
                    if np.array_equal(second_last_S, self.S):
                        # 检查两次循环后的神经元状态是否相等，如果相等表示状态振荡
                        # print('Reached oscillating neuron state.')
                        return  # break if oscillating

    def update_neurons_with_finite_temp(self, iterations, mode, beta):
        '''
        使用非线性方式计算更新神经元状态
        :param iterations: 迭代次数
        :param mode: 更新方式：同步、异步
        :param beta: 非线性参数，计算神经元状态
        :return:
        '''
        self.t += iterations
        if mode == "async":     # 异步更新
            for _ in range(iterations):
                for i in np.random.permutation(self.N):  # semi-random
                    self.S[i] = (2 * (1 / (1 + np.exp(-2 * beta * np.dot(self.w[i, :], self.S))) >= np.random.rand(1)) - 1)
        elif mode == "sync":
            for _ in range(iterations):
                self.S = (2 * (1 / (1 + np.exp(-2 * beta * np.dot(self.w, self.S))) >= np.random.rand(self.N)) - 1)
        else:
            raise ValueError("Unkown mode: {}".format(mode))

    def compute_energy(self, S):
        '''
        计算能量函数
        :param S:
        :return:
        '''
        return -0.5 * np.einsum("i,ij,j", S, self.w, S)

    def check_stability(self, S):  # stability condition
        '''
        检查神经元状态是否猫族稳定条件的函数
        稳定条件是通过将权重矩阵与神经元状态相乘，然后应用符号函数sign_0确定
        如果经过这个运算后得到的结果与原始状态 S 完全相同，那么就认为该状态是稳定的
        :param S: 给定神经元的原始状态
        :return:
        '''
        return np.array_equal(S, sign_0(np.dot(self.w, S)))


def construct_hebb_matrix(xi):
    '''
    根据输入模式xi构造Hebbian权重矩阵
    :param xi:
    :return:
    '''
    n = xi.shape[0]             # n表示模式的维度
    if len(xi.shape) == 1:      # 输入模式是一维向量
        w = np.outer(xi, xi) / n  # p = 1 使用外积运算构造权重矩阵，并对其进行归一化
    elif len(xi.shape) == 2:    # 需要记忆的模式是一个矩阵（二维向量）
        w = np.einsum("ik,jk", xi, xi) / n  # p > 1 使用张量乘法构造权重矩阵，同时进行归一化
    else:
        raise ValueError("Unexpected shape of input pattern xi: {}".format(xi.shape))
    np.fill_diagonal(w, 0)  # set diagonal elements to zero 将权重矩阵的对角线元素设置为0，表示不与自身进行连接
    return w


def hamming_distance(x, y):
    '''
    计算两个序列之间的汉明距离，也就是两个序列在相同位置上不同元素的数量
    :param x:
    :param y:
    :return:
    '''
    return np.sum(x != y)


def sign_0(array):  # x=0 -> sign_0(x) = 1
    # 神经元状态大于0将其设为1，否则设为-1
    return np.where(array >= -1e-15, 1, -1)  # machine precision: null festhalten
