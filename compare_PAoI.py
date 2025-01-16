# q1 = 0.99, q2= 0.01
#   x1 --> our proposed
#   x2 --> local
#   x3 --> edge
import numpy as np

from main_final import *

local_value = sum(const_term1)  # value of x2_Task
# print(local_value)

p_edge = [max(a, b) for a, b in zip(p_acc, p_prime)]
# print(p_edge)
Task_edge = np.zeros(M)
for i in range(M):
    Task_edge[i] = const_term1[i]+judge_term2[i]+M*func(p_edge[i], d[i], alpha[i], sigma, B, q1, q2, C)
# print(sum(Task_edge))   # value of x3_Task

var_x1 = np.var(Task[min_Task_index])
# print(var_x1)
var_x2 = np.var(const_term1)
# print(var_x2)
var_x3 = np.var(Task_edge)
# print(var_x3)

x1_Task_PAoI = 92.64015445105699
x2_Task_PAoI = 111.96393987545582
x3_Task_PAoI = 92.64015445105699
x4_Task_PAoI = 104.05091764001696

x1_Task_PAoI_var = 30
x2_Task_PAoI_var = 0
x3_Task_PAoI_var = 30
x4_Task_PAoI_var = 15

Task = [x1_Task_PAoI/30, x2_Task_PAoI/30, x3_Task_PAoI/30, x4_Task_PAoI/30]
Var = [x1_Task_PAoI_var/30, x2_Task_PAoI_var/30, x3_Task_PAoI_var/30, x4_Task_PAoI_var/30]

import matplotlib.pyplot as plt

# 生成示例数据
x = ['IJCS', 'GLCS', 'GECS', 'RROS']
y1 = Task
y2 = Var
# 创建图像和轴
fig, ax1 = plt.subplots()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Scheme Performance Comparison", font1, labelpad=10)
plt.ylabel('Task Value', font1, labelpad=10)
x_len = np.arange(len(x))
total_width, n = 0.4, 2
width = total_width / n
xticks = x_len - (total_width - width) / 2
# 绘制第一组数据（y1）
ax1.bar(xticks, y1, color='blue', width=.8 * width, label='y1', edgecolor='black', linewidth=1)
plt.axhline(y=x1_Task_PAoI/30, color='blue', linestyle='--', linewidth=1)
ax1.set_ylabel('Average value of PAoI (s)', font1, labelpad=10, color='blue')
plt.yticks(np.arange(2.4, 4, 0.3), color='blue', family='Times New Roman', fontsize=15)
plt.ylim(2.4, 3.9)
# 创建第二个坐标轴
ax2 = ax1.twinx()
# 绘制第二组数据（y2）
ax2.bar(xticks + width, y2, color='green', width=.8 * width, label='y2', edgecolor='black', linewidth=1)
ax2.set_ylabel('Offload ratio', font1, labelpad=10, color='green')
plt.yticks(np.arange(0, 1.01, 0.2), color='green', family='Times New Roman', fontsize=15)
plt.ylim(0, 1)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, position: '{0:.0f}%'.format(y*100)))

# 调整x轴的刻度
ax1.set_xticks(x_len, x, family='Times New Roman', fontsize=15)
# 显示图像
plt.tick_params(axis='both', pad=7)

plt.tight_layout()
plt.grid(axis='y', linestyle=':', linewidth=1)
# plt.savefig("D:/python_code/access_v3/results/Fig7_bar_compare_PAoI.pdf")
plt.show()
