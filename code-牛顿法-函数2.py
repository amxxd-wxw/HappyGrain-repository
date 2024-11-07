###-------------------------Pytorch框架  构建tensor变量进行求导---------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch

###-------------------------构建目标函数---------------------------
# 定义向量值边缘的实值标量函数  函数 2
def function2(x):
    func_value = torch.tensor(0.0)
    for n in range(1, 10):
        func_value += n*(x[n-1]**4)
    return func_value

###-------------------------计算一维搜索步长-----------------------
def calculate_alpha(x_k, g_k, d_k):
    "计算下降步长alpha的函数。这里采用了简单准则"
    # 初始化步长因子 alpha ，以及参数 rho
    alpha, rho = 1, 0.3
    # 用来判断是否继续进行迭代
    is_iteration = True
    while is_iteration:
        # 计算下降量 s_k
        s_k = alpha * d_k
        # 准则2.5.2   rule1应该小于等于0
        rule1 = function2(x_k + s_k) - (function2(x_k) + rho * torch.dot(g_k, s_k))
        if rule1 <= 0:
            #满足条件，即终止迭代
            is_iteration = False
        else:
            #不满足条件，缩小alpha
            alpha = alpha * 0.5
    return alpha

###-------------------------牛顿法方向---------------------------
def Newton_method_direction():
    "牛顿方向下降法"
    epsilon = 0.0001  # 梯度范数收敛范围
    d_list = [-g_list[-1]]  # 初始化下降方向d_0
    iter_num = 0  # 记录迭代轮数

    while iter_num < 1000:
        print("迭代次数：", iter_num)
        x_k = x_list[-1]
        g_k = g_list[-1]
        hessian_matrix_k = hessian_matrix_list[-1]

        # 判断是否终止迭代
        if torch.norm(g_k, p=2) < epsilon:
            break

        iter_num += 1

        # 计算下降步长 alpha
        d_k = d_list[-1]
        alpha = calculate_alpha(x_k, g_k, d_k)

        # 计算第k+1步的x值
        x_k_next = (x_k + alpha * d_k).detach().requires_grad_(True)
        x_list.append(x_k_next)

        # 计算新的目标函数值
        f_k = function2(x_k_next)
        f_list.append(f_k)
        print('f_k:', f_k)

        # 计算新的梯度 g_k_next
        f_k.backward()
        g_k_next = x_k_next.grad
        g_list.append(g_k_next)

        # 计算新的海森矩阵
        hessian_matrix_k_next = torch.autograd.functional.hessian(function2, x_k_next)
        hessian_matrix_list.append(hessian_matrix_k_next)

        # 修正牛顿法：如果海森矩阵不正定，则下降方向改为负梯度方向
        try:
            # Cholesky 分解判断正定性
            L = torch.linalg.cholesky(hessian_matrix_k_next)
            # 使用逆矩阵计算下降方向 d_k_next
            L_inv = torch.linalg.inv(L)
            d_k_next = -torch.matmul(L_inv.T, torch.matmul(L_inv, g_k_next))  # 使用牛顿方向
        except RuntimeError:
            # 如果海森矩阵不正定，改为负梯度方向
            d_k_next = -g_k_next

        d_list.append(d_k_next)

###---------------------------定义自变量---------------------------
# 向量值变元x，创建一个十维向量变量。初始值为全 1 向量
x_initial_value = torch.tensor(np.ones(9))
x = x_initial_value.requires_grad_(True)

###---------------------------计算目标函数值，以及对变量求导示例---------------------------
# 定义实值标量函数
f_x = function2(x)
# 计算梯度
f_x.backward()

###----------------------------
# 用于存储结果。x_list存储每一轮迭代得到的x，g_list存储每一轮对x的梯度
x_list = []
x_list.append(x)           #存储 x 的初始值
g_list = []
g_list.append(x.grad)      #存储 g 的初始值
f_list = []
f_list.append(f_x)            #存储f 的初始值

hessian_matrix_list = []
hessian_matrix_list.append(torch.autograd.functional.hessian(function2, x))   #存储海森矩阵的初始值

Newton_method_direction()

#绘图代码
plt.plot([f.detach().numpy() for f in f_list])  # 使用 detach() 转换为 NumPy 数组
plt.title("Newton_method_direction f2")
plt.xlabel("Index")
plt.ylabel("f_x Value")
plt.grid()
plt.show()

#输出最后一轮迭代得到的 x
print(x_list[-1])
