import math
import numpy as np
from scipy.special import lambertw
from solve_function import *
import sympy
from sympy import exp

M = 30
C = 0.5  # datasize 0.5MB
C_delta = 2000  # CPU cycles  2000Megacycles
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
    jt2 = q1 / (miu[i] * (1 - beta[i])) - q1 * C_delta / (f[i] * 1000) - q2 * epsilon * ((f[i] * 1000) ** 2) * C_delta
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
# pu = np.zeros((M, M))
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

min_Task_value = np.min(Task_compare)
min_Task_index = np.argmin(Task_compare)


# print('Task value = ', min_Task_value)
# print('number of edge computing = ', min_Task_index+1)

#  计算AoI和能耗


def func12(p, d, alpha, sigma, B, C):
    SNR = p * (d ** -alpha) / sigma
    R = B * math.log2(1 + SNR)
    function1 = C / R
    function2 = p * C / R
    return function1, function2


AoI = []
EC = []
T = []
for i in range(M):
    func1, func2 = func12(power_final[min_Task_index][i], d[i], alpha[i], sigma, B, C)

    aoi = 1 / lamda[i] + C_delta / (f[i] * 1000) + a_final[min_Task_index][i] * (
            1 / (miu[i] * (1 - beta[i])) - C_delta / (f[i] * 1000) + sum(a_final[min_Task_index]) * func1
    )
    # print(aoi)
    ec = epsilon * ((f[i] * 1000) ** 2) * C_delta + a_final[min_Task_index][i] * (
            -epsilon * ((f[i] * 1000) ** 2) * C_delta + sum(a_final[min_Task_index]) * func2
    )
    # print(ec)
    AoI.append(aoi)
    EC.append(ec)
    temp1 = q1*aoi+q2*ec

    T.append(temp1)
    total = q1*(1/lamda[i]+C_delta / (f[i] * 1000))+q2*epsilon * ((f[i] * 1000) ** 2)*C_delta\
            + a_final[min_Task_index]*(
        sum(a_final[min_Task_index])*(q1*func1+q2*func2)+q1/(miu[i] * (1 - beta[i]))
        -q1*C_delta / (f[i] * 1000)-q2*epsilon * ((f[i] * 1000) ** 2)*C_delta
            )


# print(AoI)
# print(sum(AoI)/30)
# print(EC)
# print(sum(EC)/30)
# print(0.8*sum(AoI)+0.2*sum(EC))
#
# print(min_Task_value)

#  all local computing
AoI_LC = []
EC_LC = []
AoI_EC = []
EC_EC = []
for i in range(M):
    func1, func2 = func12(power_final[min_Task_index][i], d[i], alpha[i], sigma, B, C)

    aoi_lc = 1/lamda[i]+C_delta/(f[i] * 1000)
    AoI_LC.append(aoi_lc)
    ec_lc = epsilon * ((f[i] * 1000) ** 2) * C_delta
    EC_LC.append(ec_lc)
    aoi_ec = 1/lamda[i] + 1 / (miu[i] * (1 - beta[i])) + 30*func1
    AoI_EC.append(aoi_ec)
    ec_ec = 30 * func2
    EC_EC.append(ec_ec)
# print('='*10)
# print(sum(AoI_LC)/30)
# print(sum(EC_LC)/30)
# print('='*10)
# print(sum(AoI_EC)/30)
# print(sum(EC_EC)/30)

# rand = []
# for i in range(M):
#     if i < M / 2:
#         rand.append(const_term1[i])
#     else:
#         r = q1 * (1 / lamda[i] + 1 / (miu[i] * (1 - beta[i])) + C / (
#                 B / M * 2 * math.log2(1 + p_acc[i] * (d[i] ** -alpha[i]) / sigma)
#         )) + q2 * p_acc[i] * C / (
#                     B / M * 2 * math.log2(1 + p_acc[i] * (d[i] ** -alpha[i]) / sigma)
#             )
#         rand.append(r)
# rand_value = sum(rand)
# print(rand_value)
# print(rand_value/30)

