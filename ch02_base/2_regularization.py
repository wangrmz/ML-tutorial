import numpy as np # 处理数组和数学运算
import matplotlib.pyplot as plt  # 画图
# 模型选择，划分训练集，测试集
from sklearn.model_selection import train_test_split
# 线性回归模型
# lasso回归，岭回归
from sklearn.linear_model import LinearRegression,Lasso,Ridge
# 均方误差损失函数 计算均方误差（MSE），用来衡量预测值与真实值的差距。
from sklearn.metrics import mean_squared_error
# 构建多项式特征 可以生成多项式特征（本例未使用，但可以用于解决欠拟合）。
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import rcParams  # 字体

'''
机器学习中最经典的一个反例：用一条直线去拟合一个非线性的曲线（带噪声的正弦波），结果就是“欠拟合”
'''

'''
1.生成数据
2.划分训练集和测试集（验证集）
3.定义模型（线性回归模型）
4.训练模型
5.预测结果，计算误差（损失）
'''

# 设置中文字体（让图上的文字显示中文）
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题


# 1.生成数据
'''
np.linspace(-3, 3, 300)：在区间 [-3, 3] 内均匀生成 300 个等间距的数（包含端点）。
.reshape(-1, 1)：将一维数组（形状 (300,)）重塑为列向量，形状变为 (300, 1)。
这样每一行是一个样本，每个样本只有一个特征（特征值是从 -3 到 3 的线性序列）。
'''
X = np.linspace(-3, 3, 300).reshape(-1, 1)

'''
np.sin(X)：对 X 中每个值计算正弦函数，结果形状也是 (300, 1)。这给出了一个平滑的非线性曲线。
np.random.uniform(-0.5, 0.5, 300)：从均匀分布 U(-1, 1) 中随机抽取 300 个噪声值，形状为 (300,)。
.reshape(-1, 1)：将噪声也变成 (300, 1) 的列向量，以便与 np.sin(X) 逐元素相加。
最终 Y = 正弦信号 + 随机噪声，噪声范围在 [-0.5, 0.5] 之间，噪声让数据点不再光滑，更接近真实世界采集的数据。。
'''
# 使用正态分布
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)
print(X.shape)
print(y.shape)

# 画出散点图（2*3个子图）
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].plot(X, y, "yo")
ax[0,1].plot(X, y, "yo")
ax[0,2].plot(X, y, "yo")
# plt.show()

# 2.划分训练集和测试集（验证集）
# 将 300 个样本随机分成 80% 训练（240个）和 20% 测试（60个）。
# 训练集用来教模型，测试集用来评估模型对未知数据的表现。
trainX,testX,trainY,testY = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.定义模型（线性回归模型）
# 这就是一个标准的线性回归模型，数学形式为 y=wx+b。
model = LinearRegression()

# 过拟合（20次多项式）
poly20 = PolynomialFeatures(degree=20)
x_train = poly20.fit_transform(trainX)
x_test = poly20.fit_transform(testX)
print(x_train.shape)
print(x_test.shape)

model.fit(x_train,trainY)

print(model.coef_)
print(model.intercept_)

# 5.预测结果，计算误差（损失）
# 用训练好的模型分别预测训练集和测试集的 y 值。
# mean_squared_error 计算均方误差，数值越小，模型越好

y_pred1 = model.predict(x_test)
# 损失函数，真实值，预测值
test_loss1 = mean_squared_error(testY, y_pred1) # 测试误差


# 画出拟合曲线，并写出训练误差
ax[0,0].plot(X,model.predict(poly20.fit_transform(X)), "r") # 画出模型预测的直线（红色）
ax[0,0].text(-3,1,f"测试误差:{test_loss1:.4f}")
# 画出所有系数的直方图

ax[1,0].bar(np.arange(21), model.coef_.reshape(-1))

# plt.show()

# 二、加L正则化项（Lasso回归）

# 3.定义模型（线性回归模型）
# 这就是一个标准的线性回归模型，数学形式为 y=wx+b。
lasso = Lasso(alpha=0.01)

lasso.fit(x_train,trainY)

print(lasso.coef_)
print(lasso.intercept_)

# 5.预测结果，计算误差（损失）
# 用训练好的模型分别预测训练集和测试集的 y 值。
# mean_squared_error 计算均方误差，数值越小，模型越好

y_pred2 = lasso.predict(x_test)
# 损失函数，真实值，预测值
test_loss2 = mean_squared_error(testY, y_pred2) # 测试误差


# 画出拟合曲线，并写出训练误差
ax[0,1].plot(X,lasso.predict(poly20.fit_transform(X)), "r") # 画出模型预测的直线（红色）
ax[0,1].text(-3,1,f"测试误差:{test_loss2:.4f}")
# 画出所有系数的直方图

ax[1,1].bar(np.arange(21), lasso.coef_.reshape(-1))



# 三、加L2正则化项（岭回归）

# 3.定义模型（线性回归模型）
# 这就是一个标准的线性回归模型，数学形式为 y=wx+b。
ridge = Ridge(alpha=1)
ridge.fit(x_train,trainY)

print(ridge.coef_)
print(ridge.intercept_)

# 5.预测结果，计算误差（损失）
# 用训练好的模型分别预测训练集和测试集的 y 值。
# mean_squared_error 计算均方误差，数值越小，模型越好

y_pred3 = ridge.predict(x_test)
# 损失函数，真实值，预测值
test_loss3 = mean_squared_error(testY, y_pred3) # 测试误差


# 画出拟合曲线，并写出训练误差
ax[0,2].plot(X,ridge.predict(poly20.fit_transform(X)), "r") # 画出模型预测的直线（红色）
ax[0,2].text(-3,1,f"测试误差:{test_loss3:.4f}")
# 画出所有系数的直方图

ax[1,2].bar(np.arange(21), ridge.coef_.reshape(-1))


plt.show()

