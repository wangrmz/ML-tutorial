import pandas as pd
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.linear_model import LinearRegression, SGDRegressor  # 线性回归-正规方程，线性回归-随机梯度下降
from sklearn.metrics import mean_squared_error  # 均方误差

# 1 加载数据集
advertising = pd.read_csv("../data/advertising.csv")
advertising.drop(advertising.columns[0], axis=1, inplace=True)
advertising.dropna(inplace=True)
advertising.info()
print(advertising.head())

# 2 划分训练集与测试集
# 特征和标签分开
X = advertising.drop("Sales", axis=1)
y = advertising["Sales"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 3 特征工程： 标准化
preprocessor = StandardScaler()
x_train = preprocessor.fit_transform(x_train)  # 计算训练集的均值和标准差，并标准化训练集
x_test = preprocessor.transform(x_test)  # 使用训练集的均值和标准差对测试集标准化

# 4 使用正规方程法拟合线性回归模型
# 4.1 正规方程法
normal_equation = LinearRegression()
normal_equation.fit(x_train, y_train)
print("正规方程法解得模型系数:", normal_equation.coef_)
print("正规方程法解得模型偏置:", normal_equation.intercept_)

# 4.2 使用随机梯度下降法拟合线性回归模型
gradient_descent = SGDRegressor()
gradient_descent.fit(x_train, y_train)
print("随机梯度下降法解得模型系数:", gradient_descent.coef_)
print("随机梯度下降法解得模型偏置:", gradient_descent.intercept_)

# 5 使用均方误差评估模型 MSE
print("正规方程法均方误差:", mean_squared_error(y_test, normal_equation.predict(x_test)))
print("随机梯度下降法均方误差:", mean_squared_error(y_test, gradient_descent.predict(x_test)))
