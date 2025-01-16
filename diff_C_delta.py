from scipy.special import lambertw
from solve_function import *


def diff_C_delta(delta_C_delta):
    M = 30
    C = 0.5  # datasize 0.5MB
    C_delta = 2000 + delta_C_delta  # CPU cycles  2000Megacycles
    epsilon = 10 ** (-11)
    f = np.linspace(1.1, 1.2, M)  # local computing frequency (GHz)
    q1 = 0.8  # weighted parameter for PAoI
    q2 = 0.2  # weighted parameter for Energy consumption
    lamda = np.linspace(0.4, 0.6, M)
    miu = np.linspace(1.3, 1.1, M)
    beta = []
    a = np.zeros(M)  # offload indicator

    d = np.linspace(25, 30, M)
    alpha = np.linspace(3.2, 3.5, M)
    sigma = 10 ** -5  # -30dBm////////////
    theta_threshold = 10 ** 0  # theta_threshold: -10dB/////////////
    B = 30  # total bandwidth = 1500 MHz
    acc_threshold = 0.85  # //////////////////
    a1 = 0.04592165
    a2 = 0.97662984
    c1 = 10.33466385
    c2 = -6.89161717

    for i in range(M):
        rho = lamda[i] / miu[i]
        z = (-1 / rho) * math.exp(-1 / rho)
        beta_value = -rho * lambertw(z)
        beta_real = np.real(beta_value)
        beta.append(beta_real)

    # judgement1: The second term of Task_i, -const1
    const_term1 = []
    judge_term2 = []
    p_acc = []
    hdN_arr = []
    x_threshold = solve_x(acc_threshold)  # get x_threshold
    for i in range(M):
        ct1 = q1 * (1 / lamda[i] + C_delta / (f[i] * 1000)) + q2 * epsilon * ((f[i] * 1000) ** 2) * C_delta
        jt2 = q1 / (miu[i] * (1 - beta[i])) - q1 * C_delta / (f[i] * 1000) - q2 * epsilon * (
                (f[i] * 1000) ** 2) * C_delta
        pacc = -theta_threshold * sigma * ((d[i]) ** (alpha[i])) / math.log(x_threshold)
        hdN = ((d[i]) ** (-alpha[i])) / sigma
        const_term1.append(ct1)
        judge_term2.append(jt2)
        p_acc.append(pacc)
        hdN_arr.append(hdN)

    # a=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    # ================================================================================
    # 需要判断function3和-judge_term2/M_prime的关系

    p_prime = []
    x_1_arr = []
    for i in range(M):
        x_1 = solve_lnx(hdN_arr[i] * q1 / q2)
        x_1_arr.append(x_1)
        x_2 = x_1 / hdN_arr[i]
        p_prime.append(x_2)

    # print(hdN_arr)
    # print(p_prime)
    # print(p_acc)

    # ==================================================================
    # p_acc---->accuracy constraint, p_prime---->min for function3
    # const1---->-judge_term2/M_prime

    power_for_user = np.ones(M) * 10 ** -13
    a1 = np.zeros(M)
    l1 = np.zeros(M)
    b = np.zeros((M, M))
    a_final = np.zeros((M, M))
    p_final = np.zeros((M, M))
    power_final = np.zeros((M, M))
    for u in range(1, M + 1):  # 循环有多少用户参与offload，从1到M
        l = []
        for i in range(M):
            if judge_term2[i] < 0:
                if func(p_prime[i], d[i], alpha[i], sigma, B, q1, q2, C) < -judge_term2[i] / u:
                    if p_prime[i] >= p_acc[i]:
                        a[i] = 1
                        power_for_user[i] = p_prime[i]
                    else:
                        if func(p_acc[i], d[i], alpha[i], sigma, B, q1, q2, C) < -judge_term2[i] / u:
                            a[i] = 1
                            power_for_user[i] = p_acc[i]
            l_value = (func(power_for_user[i], d[i], alpha[i], sigma, B, q1, q2, C) + judge_term2[i] / u) * a[i]
            l.append(l_value)
        power_final[u - 1] = power_for_user
        new_index, new_value = smallest_nonzero(l, u)

        for ii, xx in enumerate(new_index):
            a1[xx] = 1
        for i in range(M):
            l1[i] = l[i] * a1[i]
        a_final[u - 1] = a1
        p_final[u - 1] = l1
    # print(a_final)
    # print(p_final)
    # print(power_final)
    # 研究不同M下的TASK值
    Task = np.zeros((M, M))
    for i in range(M):
        Task[i] = const_term1 + p_final[i] * sum(a_final[i])
    # print(Task)

    Task_compare = np.zeros(M)
    for i in range(M):
        Task_compare[i] = sum(Task[i])
    # print(Task_compare)

    min_Task_value = np.min(Task_compare)  # our proposed
    min_Task_index = np.argmin(Task_compare)
    local_value = sum(const_term1)  # local computing
    p_edge = [max(a, b) for a, b in zip(p_acc, p_prime)]
    Task_edge = np.zeros(M)
    for i in range(M):
        Task_edge[i] = const_term1[i] + judge_term2[i] + M * func(p_edge[i], d[i], alpha[i], sigma, B, q1, q2, C)
    edge_value = sum(Task_edge)  # edge computing

    rand = []
    for i in range(M):
        if i < M / 2:
            rand.append(const_term1[i])
        else:
            r = q1 * (1 / lamda[i] + 1 / (miu[i] * (1 - beta[i])) + C / (
                    B / M * 2 * math.log2(1 + p_acc[i] * (d[i] ** -alpha[i]) / sigma)
            )) + q2 * p_acc[i] * C / (
                        B / M * 2 * math.log2(1 + p_acc[i] * (d[i] ** -alpha[i]) / sigma)
                )
            rand.append(r)
    rand_value = sum(rand)
    return min_Task_value, local_value, edge_value, min_Task_index + 1, rand_value


our_proposed = []
local = []
edge = []
num_edge = []
ran = []

for i in range(7):
    delta = i * 500 - 1500
    # print(2000+delta)
    a, b, c, d, e = diff_C_delta(delta)
    our_proposed.append(a / 30)
    local.append(b / 30)
    edge.append(c / 30)
    num_edge.append(d/30)
    ran.append(e/30)

# print(our_proposed)
# print(local)
# print(edge)

x = np.linspace(500, 3500, 7)

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
font_xy = {'family': 'Times New Roman', 'size': '15'}
font1 = {'family': 'Times New Roman', 'weight': 'bold', 'size': '15'}
plt.xlabel("Required number of CPU cycles (Megacycles)", font1, labelpad=10)
ax1.set_ylabel('Average value of Task', font1, labelpad=10)
plt.xticks(np.arange(500, 3501, 500), family='Times New Roman', fontsize=15)
plt.yticks(np.arange(1.8, 4.4, 0.5), family='Times New Roman', fontsize=15)
plt.xlim(500, 3500)

lns1 = ax1.plot(x, our_proposed, color='r', marker='s', markersize=8, label="IJCS", linewidth=2)
lns2 = ax1.plot(x, local, color='g', marker='^', markersize=8, label="GLCS", linewidth=2)
lns3 = ax1.plot(x, edge, color='c', marker='o', markersize=8, label="GECS", linewidth=2)
lns5 = ax1.plot(x, ran, color='pink', marker='p', markersize=8, label="RROS", linewidth=2)

plt.ylim(1.8, 4.3)
# plt.legend(['IJCS', 'GLCS', 'GECS'], prop=font_xy)
plt.grid(axis='both', linestyle=':', linewidth=1)

ax2 = ax1.twinx()
lns4 = ax2.plot(x, num_edge, linestyle='--', color='blue', marker='D', label='Offload ratio', linewidth=2)
ax2.set_ylabel('Offload ratio', font1, labelpad=10, color='blue')
plt.yticks(np.arange(0, 1.01, 0.2), color='blue', family='Times New Roman', fontsize=15)
plt.ylim(0, 1)

lns = lns1 + lns2 + lns3 + lns5 + lns4
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc='upper left', prop=font_xy)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, position: '{0:.0f}%'.format(y*100)))
# plt.gca().yaxis.get_offset_text().set_fontsize(15)

# plt.minorticks_on()
# plt.tick_params(left=True, bottom=False, which='minor')

plt.tick_params(axis='both', pad=7)

plt.tight_layout()
# plt.savefig("D:/python_code/access_v3/results/Fig11_compare_Cdelta.pdf")

plt.show()
