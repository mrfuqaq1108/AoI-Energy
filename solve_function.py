import math
import numpy as np


#  求解方程(1+x)ln(1+x)-x=t, x范围是[0, 10000]
def solve_lnx(t):
    def function1(s):
        return (1 + s) * math.log(1 + s) - s

    x_min = 0
    x_max = 10000
    for ite in range(10000):
        x_new = (x_min + x_max) / 2
        # b = function1(x_new)
        if np.fabs(function1(x_new) - t) < math.pow(10, -8) or (iter == 9999):
            return x_new
        elif function1(x_new) < t:
            x_min = x_new
        elif function1(x_new) > t:
            x_max = x_new


#  求解方程y~x
def solve_x(acc):
    def accu(x):
        a1 = 0.04592165
        a2 = 0.97662984
        c1 = 10.33466385
        c2 = -6.89161717
        y = a1 + (a2 - a1) / (1 + math.exp(-(c1 * x + c2)))
        return y

    x_min = 0
    x_max = 1
    for ite in range(10000):
        x_new = (x_min + x_max) / 2
        if np.fabs(accu(x_new) - acc) < math.pow(10, -14) or (iter == 9999):
            return x_new
        elif accu(x_new) < acc:
            x_min = x_new
        elif accu(x_new) > acc:
            x_max = x_new


# x = solve_x(0.8)
# print(x)
def func(p, d, alpha, sigma, B, q1, q2, C):
    SNR = p * (d ** -alpha) / sigma
    R = B * math.log2(1 + SNR)
    function = q1 * C / R + q2 * p * C / R
    return function


def smallest_nonzero(arr,M):
    arr = np.array(arr)
    value = []
    sort_index = np.argsort(arr)
    for i in range(len(arr)):
        value.append(arr[sort_index[i]])
    non_zero_count = np.count_nonzero(value)
    new_index = sort_index[:(min(M,non_zero_count))]
    new_value = value[:(min(M,non_zero_count))]
    return new_index, new_value
