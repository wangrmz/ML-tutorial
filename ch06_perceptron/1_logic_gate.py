# 定义
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     result = w1 * x1 + w2 * x2
#     if result > theta:
#         return 1
#     else:
#         return 0

import numpy as np


# ND 门：权重 [0.5, 0.5]，偏置 -0.7 → 输出只有当 x1=1 且 x2=1 时为 1，其余为 0。
#
# NAND 门：权重 [-0.5, -0.5]，偏置 0.7 → 输出与 AND 相反。
#
# OR 门：权重 [0.5, 0.5]，偏置 -0.2 → 只要有一个输入为 1 即输出 1。
#
# XOR 门：通过组合 NAND(x1, x2) 和 OR(x1, x2) 作为 AND 的输入，符合异或逻辑（相同为 0，不同为 1）。



def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    # 直接使用矩阵运算
    res = w @ x + b
    if res <=  0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    # 直接使用矩阵运算
    res = w @ x + b
    if res <=  0:
        return 0
    else:
        return 1

# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    # 直接使用矩阵运算
    res = w @ x + b
    if res <=  0:
        return 0
    else:
        return 1

# 异或门
def XOR(x1, x2):
   s1 = NAND(x1, x2)
   s2 =OR(x1, x2)
   y = AND(s1, s2)
   return y


# 测试
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
print(AND(0, 0))

print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
print(NAND(0, 0))

print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
print(OR(0, 0))


print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
print(XOR(0, 0))
