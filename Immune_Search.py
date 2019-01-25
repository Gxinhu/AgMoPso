import copy
import pygmo as pg
import numpy as np
import math
import random
import funcition
import main

def immune_search(A, NC, N, pc, pm, x_min, x_max, f_num, fun):
    E = CloneOperator(A, NC, N)
    S = []
    for i in range(len(E)):
        k = random.randint(0, 1)
        if random.random() < pc:
            C = np.zeros((2, len(E[0].x)))
            j = random.randint(0, len(E) - 1)
            C[0], C[1] = SBX(E[i].x, E[j].x,x_min,x_max)
            if random.random() < pm:
                x=PM(C[k],x_min,x_max)
                S.append(funcition.partical(x, fun, len(E[0].x)))
            else:
                S.append(funcition.partical((C[k]), fun, len(E[0].x)))
        elif (random.random() < pm):
            S.append(funcition.partical(PM(E[i].x, x_min, x_max), fun, len(E[0].x)))
        else:
            S.append(E[i])
    return S


def CloneOperator(A, NC, N):
    PC = copy.deepcopy(A)
    fitnesss=[]
    for i in range(len(PC)):
        fitnesss.append(PC[i].fitness)
    CDA=pg.crowding_distance(fitnesss)
    for i in range(len(PC)):
        PC[i].crowding_distance=CDA[i]
    # PC = crowding_distance(PC, f_num)
    PC = sort_by_crowding_distance(PC)
    max_=0
    for i in range(len(PC)):
        if PC[i].crowding_distance!=math.inf:
            if PC[i].crowding_distance>max_:
                max_=PC[i].crowding_distance
    for i in range(len(PC)):
        if PC[i].crowding_distance==math.inf:
                PC[i].crowding_distance=2*max_
    if (len(PC) > NC):
        PC = PC[:NC]
    E = []
    sum = 0
    for i in range(len(PC)):
        sum += PC[i].crowding_distance
    for i in range(len(PC)):
        try:
            q = math.ceil(N * PC[i].crowding_distance / sum)
        except ValueError:
            print(A)
        for j in range(q):
            E.append(PC[i])
    return E


# def crowding_distance(PC, f_num):

#     for dim in range(f_num):
#         new_list = sort_by_coordinate(PC, dim)
#         new_list[0].crowding_distance = math.inf
#         new_list[-1].crowding_distance = math.inf
#         max_distance = new_list[0].fitness[dim] - new_list[-1].fitness[dim]
#         for i in range(1, len(new_list) - 1):
#             distance = new_list[i + 1].fitness[dim] - new_list[i - 1].fitness[dim]
#             if max_distance == 0:
#                 new_list[i].crowding_distance = 0
#             else:
#                 new_list[i].crowding_distance += distance / max_distance
#     return new_list
#
#
# def sort_by_coordinate(population, dim):  # selection sort, which can be replaced with quick sort
#     p_list = []
#     for p in population:
#         p_list.append(p)
#
#     for i in range(0, len(p_list) - 1):
#         for j in range(i + 1, len(p_list)):
#             if p_list[i].fitness[dim] < p_list[j].fitness[dim]:
#                 temp = p_list[i]
#                 p_list[i] = p_list[j]
#                 p_list[j] = temp
#
#     return p_list


def sort_by_crowding_distance(population):  # selection sort, which can be replaced with quick sort
    p_list = []
    for p in population:
        p_list.append(p)

    for i in range(0, len(p_list) - 1):
        for j in range(i + 1, len(p_list)):
            if p_list[i].crowding_distance < p_list[j].crowding_distance:
                temp = p_list[i]
                p_list[i] = p_list[j]
                p_list[j] = temp

    return p_list


def SBX(E1, E2,x_min,x_max):
    u = random.random()
    n = 1
    if u <= 0.5:
        beta = (2 * u) ** (1 / (n + 1))
    else:
        beta = (1 / (2 * (1 - u))) ** (1 / (1 + n))
    C1 = 0.5 * ((1 + beta) * E1 + (1 - beta) * E2)
    C2 = 0.5 * ((1 - beta) * E1 + (1 + beta) * E2)
    C1=np.clip(C1,a_min=x_min,a_max=x_max)
    C2 =np.clip(C2, a_min=x_min, a_max=x_max)
    return C1, C2


def PM(C,x_min,x_max):
    yita = 1
    u2 = random.random()
    if (u2 <0.5):
        delta = float((2 * u2) ** (1 / (yita + 1)) - 1)
    else:
        delta = float(1 - (2 * (1 - u2)) ** (1 / (yita + 1)))
    off1 = C + delta
    off1 = np.clip(off1, a_min=x_min, a_max=x_max)
    return off1[0]
