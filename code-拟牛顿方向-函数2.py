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

###-------------------------拟牛顿方法---------------------------
def renew_H_Matrix(H_k, s_k, y_k):
    "用于拟牛顿法中更秩二矫正的H矩阵更新"
    # 计算分子和分母
    # s_k s_k^T，外积
    s_k_outer = torch.ger(s_k, s_k)
    # s_k^T y_k
    y_k_dot_s_k = torch.dot(s_k, y_k)

    # 检查分母是否为零，以避免除零错误
    if y_k_dot_s_k == 0:
        raise ValueError("s_k^T y_k 为零，无法更新 H_k。")

    # 第一个项: s_k s_k^T / s_k^T y_k
    term1 = s_k_outer / y_k_dot_s_k

    # 计算第二项
    # 计算 H_k y_k
    H_y_k = torch.mv(H_k, y_k)
    # H_k y_k y_k^T H_k
    y_k_outer_H_y_k = torch.ger(H_y_k, H_y_k)
    # y_k^T H_k y_k
    y_k_dot_H_y_k = torch.dot(y_k, H_y_k)
    # 检查分母是否为零，以避免除零错误
    if y_k_dot_H_y_k == 0:
        raise ValueError("y_k^T H_k y_k 为零，无法更新H_k。")
    # 第二项: H_k y_k y_k^T H_k / y_k^T H_k y_k
    term2 = y_k_outer_H_y_k / y_k_dot_H_y_k

    # 更新 H_k
    H_k_plus_1 = H_k + term1 - term2
    return H_k_plus_1

def pseudo_Newtonian_direction():
    "拟牛顿方向下降法"
    # 梯度范数收敛范围
    epsilon = 0.0001
    # 初始化 H_0。H_list用于存储每一轮迭代得到的矩阵 H
    H_list = [torch.eye(len(x_list[-1]))]
    #记录迭代轮数 iter_num
    iter_num = 0
    while iter_num < 1000:
        print("迭代次数：", iter_num)
        # 第k步的梯度
        x_k = x_list[-1]
        g_k = g_list[-1]
        # 判断是否终止迭代
        if torch.norm(g_k, p=2) < epsilon:
            break  # 跳出循环
        iter_num += 1
        # 计算下降方向 d_k
        #第k步的矩阵H
        H_k = H_list[-1].double()
        #第k步的下降方向d_k
        d_k = torch.matmul(-H_k, g_k)
        # 计算下降步长 alpha
        alpha = calculate_alpha(x_k, g_k, d_k)
        # 计算第k+1步的x值：x_k_next，并确保启用 requires_grad=True.
        '''
        这里 detach() 将 x_k + alpha * d_k 的计算图分离，防止其与前一步的计算图连接，然后重新启用 requires_grad=True
        '''
        x_k_next = (x_k + alpha * d_k).detach().requires_grad_(True)
        # 将第k+1步的x值：x_k_next 存储到x_list
        x_list.append(x_k_next)
        #计算此时的目标函数值
        f_k = function2(x_k_next)
        f_list.append(f_k)
        print('f_k:',f_k)
        # 计算新的梯度 g_k_next
        f_k.backward()
        g_k_next = x_k_next.grad
        # 将第k+1步的梯度保存到g_list
        g_list.append(g_k_next)
        # 开始求解第k+1步的矩阵H: H_k+1
        s_k = x_list[-1] - x_list[-2]
        y_k = g_list[-1] - g_list[-2]
        H_k_next = renew_H_Matrix(H_k, s_k, y_k)
        # 第k+1步的矩阵H: H_k+1 保存到H_list中
        H_list.append(H_k_next)

###---------------------------定义自变量---------------------------
# 变量的初始值。全 1 向量
x_initial_value = torch.tensor(np.ones(9))
# 向量值变元x，创建一个十维向量变量
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

pseudo_Newtonian_direction()

#绘图代码
plt.plot([f.detach().numpy() for f in f_list])  # 使用 detach() 转换为 NumPy 数组
plt.title("pseudo_Newtonian_direction f2")
plt.xlabel("Index")
plt.ylabel("f_x Value")
plt.grid()
plt.show()


#输出最后一轮迭代得到的 x
print(x_list[-1])