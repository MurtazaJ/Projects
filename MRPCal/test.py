from sys import maxunicode
from matplotlib.cbook import maxdict
import numpy as np


# def solution(a):
#     # a = sorted(a) 
#     a_set = set(a)
#     maximum = 0
#     for i in range(1, max(a_set)+1):
#         if i not in a_set:
#             return i
#     if (max(a_set)+ 1) <= 0:
#         return 1
#     else:
#         return max(a_set)+ 1
    


#     def solution(N,K):
#         first_digit = int(math.log10(n))
#         n = int(n/pow(10, digits))


def solution(N, K):
    digit_100 = int(N / 100)
    digit_10 = int(N % 100 / 10)
    digit_1 = N % 10
    
    increase_100 = min(K, 9 - digit_100)
    K = K - increase_100
    increase_10 = min(K, 9 - digit_10)
    K = K - increase_10
    increase_1 = min(K, 9 - digit_1)
    
    new_digit_100 = digit_100 + increase_100
    new_digit_10 = digit_10 + increase_10
    new_digit_1 = digit_1 + increase_1
    
    return new_digit_100 * 100 + new_digit_10 * 10 + new_digit_1
    # for i in range(0 , len(a)):
    #     if (max + 1) not in a_set:
    #         break
    #     elif a[i] > max:
    #         max = a[i]
           
    # if max <= 0:
    #     return 1
    # else:
    #     return max + 1
  

print(solution(100, 0) )
