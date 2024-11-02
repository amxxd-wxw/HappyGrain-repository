###-------------------------Pytorch框架  构建tensor变量进行求导---------------------------
import numpy as np
import torch

###-------------------------构建目标函数---------------------------
# 定义向量值边缘的实值标量函数  函数 1
def function1(x):
    func_value = torch.tensor(0.0)
    for n in range(0, 9):
        func_value += (1 - x[n]) ** 2 + 100 * (x[n + 1] - x[n] ** 2) ** 2
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
        rule1 = function1(x_k + s_k) - (function1(x_k) + rho * torch.dot(g_k, s_k))
        if rule1 <= 0:
            #满足条件，即终止迭代
            is_iteration = False
        else:
            #不满足条件，缩小alpha
            alpha = alpha * 0.5
    return alpha

###-------------------------负梯度方向方法---------------------------
def conjugate_gradient_direction():
    "负梯度方向下降法"
    # 梯度范数收敛范围
    epsilon = 0.00001
    # 初始化下降方向d_0。d_list用于存储每一轮迭代得到的下降方向d_k
    d_list = [-g_list[-1]]
    #记录迭代轮数 iter_num
    iter_num = 0

    while True:
        print("迭代次数：", iter_num)
        # 第k步的梯度
        x_k = x_list[-1]
        g_k = g_list[-1]
        # 判断是否终止迭代
        if torch.norm(g_k, p=2) < epsilon:
            break  # 跳出循环
        iter_num += 1
        #第k步的下降方向
        d_k = d_list[-1]
        # 计算下降步长 alpha
        alpha = calculate_alpha(x_k, g_k, d_k)
        # 计算第k+1步的x值：x_k_next，并确保启用 requires_grad=True.
        '''
        这里 detach() 将 x_k + alpha * d_k 的计算图分离，防止其与前一步的计算图连接，然后重新启用 requires_grad=True
        '''
        x_k_next = (x_k + alpha * d_k).detach().requires_grad_(True)
        # 将第k+1步的x值：x_k_next 存储到x_list
        x_list.append(x_k_next)
        # 计算此时的目标函数值
        f_k = function1(x_k_next)
        print('f_k:', f_k)
        # 计算新的梯度 g_k_next
        f_k.backward()
        g_k_next = x_k_next.grad
        # 将第k+1步的梯度保存到g_list
        g_list.append(g_k_next)
        #开始求解第k+1步的下降方向d_k_next
        d_k_next = -g_k_next
        d_list.append(d_k_next)


###---------------------------定义自变量---------------------------
# 向量值变元x，创建一个十维向量变量。初始值为全零向量
x_initial_value = torch.tensor(np.zeros(10))
x = x_initial_value.requires_grad_(True)

###---------------------------计算目标函数值，以及对变量求导示例---------------------------
# 定义实值标量函数
f_x = function1(x)
# 计算梯度
f_x.backward()

###----------------------------
# 用于存储结果。x_list存储每一轮迭代得到的x，g_list存储每一轮对x的梯度
x_list = []
x_list.append(x)           #存储 x 的初始值
g_list = []
g_list.append(x.grad)      #存储 g 的初始值

conjugate_gradient_direction()

#输出最后一轮迭代得到的 x
print(x_list[-1])
