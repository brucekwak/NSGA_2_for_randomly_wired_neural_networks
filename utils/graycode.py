from __future__ import division
import random
import warnings


def check_Upper(gy, num_graph):
    tmp = ''
    for i in gy:
        tmp += str(i)

    gray_len = len(str(grayCode(num_graph)))

    decimal = graydecode(int(tmp))
    if decimal > num_graph - 1:
        return num2gray(num_graph-1,gray_len)
    else :
        return gy

def grayCode(n):
    # Decimal to binary graycode
    # Right Shift the number
    # by 1 taking xor with
    # original number
    grayval = n ^ (n >> 1)

    return int(bin(grayval)[2:])

def graydecode(binary):
    # binary -> decimal

    binary1 = binary
    decimal, i, n = 0, 0, 0
    while (binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1

    # Taking xor until
    # n becomes zero
    inv = 0
    while (decimal):
        inv = inv ^ decimal;
        decimal = decimal >> 1;

    return inv

def num2gray(n, gray_len):
    gy = str(grayCode(n))

    if len(gy) < gray_len:
        gy = '0' * (gray_len - len(gy)) + gy

    return gy


############################
# Crossover - for gray encoding
############################
def cxgray(ind1, ind2, cxpb, num_graph):
    gray_len = len(str(grayCode(num_graph-1)))
    
    for stage_i in range(3):  # 0, 1, 2
        # stage_i 에서
        if random.random() < cxpb:
            #  e.g. stage 1 => temp[0:7], temp[7:14], temp[14:21] 
            ind1_stage_i = ind1[stage_i*7:(stage_i+1)*7]
            ind2_stage_i = ind2[stage_i*7:(stage_i+1)*7]

            cx_point = random.randint(1, 5)  # 1 ~ 5
            ind1_stage_i[cx_point:], ind2_stage_i[cx_point:] = ind2_stage_i[cx_point:], ind1_stage_i[cx_point:]
            
            ind1[stage_i*7:(stage_i+1)*7] = ind1_stage_i
            ind2[stage_i*7:(stage_i+1)*7] = ind2_stage_i

    return ind1, ind2


############################
# Mutate - for gray encoding
############################
# 크로모좀 내에서 스테이지마다, mutpb 의 확률로 random flip 적용
def mutgray(individual, mutpb, num_graph):
    gray_len = len(str(grayCode(num_graph-1)))
    
    size = len(individual)  # 인덱스 0 ~ (size-1)
    
    for i in range(3):
        if random.random() < mutpb:
            flip_idx = random.randint(gray_len * i, gray_len * (i + 1) - 1)  # 0:6, 7:13, 14:20
            # bit flip
            if individual[flip_idx] == 0:
                individual[flip_idx] = 1
            elif individual[flip_idx] == 1:
                individual[flip_idx] = 0

    return individual,