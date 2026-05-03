import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1.加载数据集
dataset = pd.read_csv('../data/train.csv')

# 测试图像
# digit = dataset.iloc[10,1:].values
# # 灰度图像
# plt.imshow(digit.reshape(28,28),cmap='gray')
# plt.show()

# 划分数据集
X = dataset.drop("label", axis=1)
y = dataset["label"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 3.特征工程:归一化处理
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义模型和训练
model = LogisticRegression()
model.fit(x_train, y_train)

# 模型评估
score = model.score(x_test, y_test)
print(score)

# 预测
# 转换成二维矩阵
plt.imshow(dataset.iloc[123, 1:].values.reshape(28, 28), cmap="gray")
plt.show()
print(model.predict(dataset.iloc[123, 1:].values.reshape(1, -1)))