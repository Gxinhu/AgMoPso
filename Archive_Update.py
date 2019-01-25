import Immune_Search
import pygmo as pg
import math
import numpy as np
def archive_update(S,A,N,f_num):
    for i in range(len(S)):
        rmlist = []
        for j in range(len(A)):
            flag=Dominate(S[i],A[j])
            if flag==1:
                rmlist.append(A[j])
            elif flag==-1:
                break
        for j in range(len(rmlist)):
            A.remove(rmlist[j])
        if flag!=-1:
            A.append(S[i])
            if len(A)>N:
                fitnesss=[]
                for i in range(len(A)):
                    fitnesss.append(A[i].fitness)
                CDA=pg.crowding_distance(fitnesss)
                for i in range(len(A)):
                    A[i].crowding_distance=CDA[i]
                A=sort_by_crowding_distance(A)
                max_ = 0
                for i in range(len(A)):
                    if A[i].crowding_distance != math.inf:
                        if A[i].crowding_distance > max_:
                            max_ = A[i].crowding_distance
                for i in range(len(A)):
                    if A[i].crowding_distance == math.inf:
                        A[i].crowding_distance = 2 * max_
                del A[-1]
    return A

#     for i in range(len(S)):
#         mark = []
#         for j in range(len(A)):
#             flag=CheckDominance(S[i],A[j])
#             if flag==1:
#                 mark.append(A[j])
#             else:
#                 break
#         if len(mark)>0:
#             for k in mark:
#                 if k in A:
#                     A.remove(k)
#         if flag!=-1:
#             A.append(S[i])
#             if len(A)>N:
#                 A=crowding_distance(A)
#                 A=Immune_Search.sort_by_crowding_distance(A)
#                 for i in range(len(A)):
#                     if A[i].crowding_distance != math.inf:
#                         for j in range(len(A)):
#                             if A[j].crowding_distance == math.inf:
#                                 A[j].crowding_distance = 2 * A[i].crowding_distance
#                 del A[-1]
#
#     return A
#
# def crowding_distance(A_):
#     fitnesss=[]
#     for i in range(len(A_)):
#         fitnesss.append(A_[i].fitness)
#     CDA=pg.crowding_distance(fitnesss)
#     for i in range(len(A_)):
#         A_[i].crowding_distance=CDA[i]
#     return Immune_Search.sort_by_crowding_distance(A_)

def Dominate(y1, y2):
    less = 0  # y1的目标函数值小于y2个体的目标函数值数目
    equal = 0  # y1的目标函数值等于y2个体的目标函数值数目
    greater = 0  # y1的目标函数值大于y2个体的目标函数值数目
    for i in range(len(y1.fitness)):
        if y1.fitness[i] > y2.fitness[i]:
            greater = greater + 1
        elif y1.fitness[i] == y2.fitness[i]:
            equal = equal + 1
        else:
            less = less + 1
    if (greater == 0 and equal != len(y1.fitness)):
        return 1  # y1支配y2返回正确
    elif (less == 0 and equal != len(y1.fitness)):
        return -1  # y2支配y1返回false
    else:
        return 0

def CheckDominance(s,a):
    less=0
    great=0
    for k in range(len(s.fitness)):
        if (s.fitness[k]<=a.fitness[k]):
            less+=1
        if s.fitness[k]>=a.fitness[k]:
            great+=1
    if less==len(s.fitness):
        return 1
    elif great==len(s.fitness) :
        return -1
    return 0
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