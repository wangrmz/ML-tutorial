import numpy as np

# 定义损失函数
def J(beta):
    """目标函数"""
    # return np.sum((X @ beta - y) ** 2, axis=0).reshape(-1, 1) / n
    return np.sum((X @ beta - y) ** 2) / n

# 定义
def gradient(beta):
    """梯度"""
    return X.T @ (X @ beta - y) / n * 2


# 1.定义数据
X = np.array([[5], [8], [10], [12], [15], [3], [7], [9], [14], [6]])  # 自变量，每周学习时长
y = np.array([[55], [65], [70], [75], [85], [50], [60], [72], [80], [58]])  # 因变量，数学考试成绩

n = X.shape[0]  # 样本数

# 2. 数据处理，X增加一列1
X = np.hstack((np.ones((n, 1)), X))

# 3.初始化参数以及超参数
alpha = 0.01
iter = 10000

beta = np.array([[1],[1]])

# 停止两个方面：1是到达迭代次数，2是梯度
# 重复迭代
for i in range(iter):
    # 4.计算梯度
    grad = gradient(beta)

    # 5.更新参数
    beta = beta - alpha * grad   # 更新参数

    # 每迭代10次打印一次当前的参数值和损失值
    if i % 10 == 0:
        print(f"beta={beta.reshape(-1)}\tJ={J(beta)}")

