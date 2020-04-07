from sklearn import svm
import numpy as np

x = [[2, 0], [1, 1], [2, 3]]  # 三个点
y = [0, 0, 1]  # 分别为0类，0类，1类
clf = svm.SVC(kernel='linear')  # 线性核函数
clf.fit(x, y)  # 建立模型

print(clf)  # 会打印模型的一些参数

# get support vectors支持向量
print(clf.support_vectors_)
# get indices of support vectors 支持向量下标
print(clf.support_)
# get number of support vectors for each class支持向量的个数
print(clf.n_support_)
# 预测
new_x = [0, 1]
new_x = np.array(new_x).reshape(1, -1)
pre_y = clf.predict(new_x)
print(pre_y)
