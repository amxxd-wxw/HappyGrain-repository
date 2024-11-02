# Pytorch框架  构建tensor变量进行求导
import numpy as np
import torch
from torch.nn import Parameter


# 定义向量值边缘的实值标量函数  函数 1
def function2(x):
    func_value = torch.tensor(0.0)
    for n in range(1, 10):
        func_value += n*x[n-1]**4
    return func_value


# 创建一个十维向量变量
x_initial_value = torch.tensor(np.ones(9))  # 变量的初始值
# 向量值变元x
x = Parameter(x_initial_value)
# 定义实值标量函数
f_x = function2(x)

# 计算梯度
f_x.backward()

# 打印 x 的梯度
#print("x 的梯度:", x.grad)
# 打印结果
#print("f(x):", f_x)

def calculate_alpha(x_k, g_k, d_k):
    "计算下降步长alpha的函数。这里采用了Armijo-Goldstein准则"
    # 初始化步长因子 alpha ，以及参数 rho
    alpha, rho = 1, 0.3
    # 用来判断是否继续进行迭代
    is_iteration = True
    while is_iteration:
        # 计算下降量 s_k
        s_k = alpha * d_k
        # 准则2.5.2   rule1应该小于等于0
        rule1 = function2(x_k + s_k) - (function2(x_k) + rho * torch.dot(g_k, s_k))
        # 准则2.5.3   rule2应该大于等于0
        rule2 = function2(x_k + s_k) - (function2(x_k) + (1 - rho) * torch.dot(g_k, s_k))
        # print(rule1)
        if rule1 <= 0 and rule2 >= 0:
            is_iteration = False
        else:
            if rule1 > 0:
                alpha = alpha * 0.5
            elif rule2 < 0:
                alpha = alpha / 0.5
        print(alpha)
    return alpha

# 下降方向选择1：负梯度方向
def negative_gradient():
    "负梯度方向下降法"
    epsilon = 0.01
    # 用来判断是否继续进行迭代
    is_iteration = True
    while is_iteration:
        # 计算目标函数
        f_x = function2(x)
        #print(f_x)
        # 计算此时的梯度
        f_x.backward()
        g_k = x.grad
        # print("Gradient:", g_k)

        # 选择下降方向 d_k 负梯度方向
        d_k = -g_k

        # 计算步长 alpha
        alpha = calculate_alpha(x.data, g_k, d_k)

        # 更新 x_value，并重新设置 requires_grad=True 以支持梯度计算
        x_value = (x.data + alpha * d_k).detach().requires_grad_(True)
        x.data = x_value
        # 判断是否满足终止条件
        if torch.norm(g_k, p=2) < epsilon:
            is_iteration = False


negative_gradient()
print(x.data)