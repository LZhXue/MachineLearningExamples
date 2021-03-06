from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
# from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'E:/Pre_Study_of_Postguaduate/MachineLearningExamples/DecisionTree/AllElectronicscsv.csv', 'rt')
reader = csv.reader(allElectronicsData)  # 按行读取
headers = next(reader)

print(headers)  # 打印第一行

featureList = []  # 特征值list
labelList = []  # 类别list

for row in reader:  # 读一行
    labelList.append(row[len(row)-1])  # 每一行的最后一个值
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vetorize features
vec = DictVectorizer()  # 实例化
dummyX = vec.fit_transform(featureList) .toarray()  # 格式转化

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')  # 度量标准entropy信息熵
clf = clf.fit(dummyX, dummyY)  # 建模
print("clf: " + str(clf))


# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX.reshape(1, -1))
print("predictedY: " + str(predictedY))
