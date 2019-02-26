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
        C = np.zeros((2, len(E[0].x)))
        j = random.randint(0, len(E) - 1)
        C[0], C[1] = SBX(E[i].x, E[j].x, x_min, x_max,pc)
        S.append(funcition.partical(PM(C[k],x_min,x_max,pm),fun,len(A[0].x)))
    return S


def CloneOperator(A, NC, N):
    PC = copy.deepcopy(A)
    fitnesss = []
    for i in range(len(PC)):
        fitnesss.append(PC[i].fitness)
    if len(fitnesss)>1:
        CDA = pg.crowding_distance(fitnesss)
        for i in range(len(PC)):
            PC[i].crowding_distance = CDA[i]
        # PC = crowding_distance(PC, f_num)
        PC = sort_by_crowding_distance(PC)
    else:
        PC[0].crowding_distance=math.inf
    if len(PC)>NC:
        PC=PC[:NC]
    max_distance = 1.0
    min_distance=0.0
    k=0
    while k<len(PC):
        if PC[k].crowding_distance!=math.inf:
            max_distance=2*PC[k].crowding_distance
            min_distance=PC[-1].crowding_distance
            for i in range(k):
                PC[i].crowding_distance=2*PC[k].crowding_distance
            break
        k+=1
    if PC[0].crowding_distance==math.inf:
        for i in range(len(PC)):
            PC[i].crowding_distance=1.0
    E = []
    sum = 0
    for i in range(len(PC)):
        sum += PC[i].crowding_distance
        if PC[i].crowding_distance>max_distance:
            PC[i].crowding_distance=max_distance
        if PC[i].crowding_distance<min_distance:
            PC[i].crowding_distance = min_distance
    for i in range(len(PC)):
        q = math.ceil(N * PC[i].crowding_distance / sum)
        if sum==0:
            q = math.ceil(N/ len(PC))
        for j in range(q):
            E.append(PC[i])
    return E


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


def SBX(E1, E2, x_min, x_max,pc):
    off1=E1
    off2 =E2
    EPS=1.0e-14
    n = 20
    if random.random()<=pc:
        for i in range(len(E1)):
            valueX1=E1[i]
            valueX2=E2[i]
            if random.random()<=0.5:
                if abs(valueX1-valueX2>EPS):
                    if(valueX1<valueX2):
                        y1=valueX1
                        y2=valueX2
                    else:
                        y1 = valueX2
                        y2 = valueX1
                    yL=x_min[i]
                    yU=x_max[i]
                    rand=random.random()
                    beta = 1.0 + (2.0 * (y1 - yL) / (y2 - y1))
                    alpha = 2.0 - pow(beta, -(n + 1.0))
                    if rand<=1/alpha:
                        betaq=pow(rand*alpha,1/(n+1))
                    else:
                        betaq = pow(1/(2-rand * alpha), 1 / (n + 1))
                    c1=0.5*((y1+y2)-betaq*(y2-y1))
                    beta = 1.0 + (2.0 * (yU - y2) / (y2 - y1))
                    alpha = 2.0 - pow(beta, -(n + 1.0))
                    if rand <= 1 / alpha:
                        betaq = pow(rand * alpha, 1 / (n + 1))
                    else:
                        betaq = pow(1 / (2 - rand * alpha), 1 / (n + 1))
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    if (c1 < yL):
                        c1 = yL

                    if (c2 < yL):
                        c2 = yL

                    if (c1 > yU):
                        c1 = yU

                    if (c2 > yU):
                        c2 = yU
                    if (random.random() <= 0.5):
                        off1[i]= c2
                        off2[i] =c1
                    else :
                        off1[i]= c1
                        off2[i]= c2
                else:
                    off1[i]=valueX1
                    off2[i]= valueX2
            else:
                off1[i]= valueX2
                off2[i]= valueX1
    return off1,off2
def PM(C, x_min, x_max,pm):
    yita = 20
    off=C
    for i in range(len(C)):
        if random.random()<=pm:
            y=C[i]
            yl=x_min[i]
            yu=x_max[i]
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            rnd = random.random()
            mut_pow = 1.0 / (yita + 1.0)
            if (rnd <= 0.5):
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (yita + 1.0)))
                deltaq = pow(val, mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (yita+ 1.0)))
                deltaq = 1.0 - (pow(val, mut_pow))
            y = y + deltaq * (yu - yl)
            if (y < yl):
                y = yl
            if (y > yu):
                y = yu
            off[i]=y
    return off