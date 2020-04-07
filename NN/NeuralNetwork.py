import numpy as np


def tanh(x):
    return np.tanh(x)


# 一阶导数
def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)


# 逻辑函数
def logistic(x):
    return 1/(1 + np.exp(-x))


# 逻辑函数的导数
def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):  # __init__构造函数，self指向当前对象的指针，相当于this，layers每层里面有多少个神经元，是一个list，activation激活函数，默认tanh
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):  # 训练，epochs前后更新次数
        X = np.atleast_2d(X)  # 至少二维
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])  # shape返回矩阵的尺寸
            a = [X[i]]  # 随机抽取的一行

            for l in range(len(self.weights)):  # going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  # Computer the node value for each layer (O_i) using activation function
            error = y[i] - a[-1]  # Computer the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])]  # For output layer, Err calculation (delta is updated error)

            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer 倒数第一 层
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer

                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))  # 更新过的误差
            deltas.reverse()
            for i in range(len(self.weights)):  # 权重更新
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)  # 内积

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x  # 0行到最后一行
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
