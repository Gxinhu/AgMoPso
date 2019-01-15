import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import math
import funcition
import pygmo as pg
import copy
def initialization(N,f_num,x_num,x_min,x_max,fun):
    lambda_ = np.zeros((1,f_num))
    P = []

    # 种群初始化以及产生labmda(权值向量)
    for i in range(N):
        temp=np.zeros((f_num))
        x =np.zeros(x_num)
        for j in range(x_num):
            x[j]= x_min[0, j] + (x_max[0, j] - x_min[0, j]) * random.random()
        P.append(funcition.partical(x,fun,x_num))
        if (f_num == 2):
            temp[0]=float(i) / N
            temp[1]=1.0 - float(i) / N
        elif (f_num == 3):
            temp[0]=np.array([float(i) / N])
            temp[1]=np.array([1.0 - float(i) / N])
            temp[2]=np.array([1.0 - float(N - i - 1) / N])
        lambda_=np.concatenate((lambda_,np.array([temp])))
    return P, lambda_
def Caculateminobj(P,N,f_num):
    z_star=np.zeros(f_num)
    min_1=P[0].fitness[0]
    min_2=P[0].fitness[1]
    for i in range(N):
        if P[i].fitness[0]<min_1:
            min_1=P[i].fitness[0]
            z_star[0]=P[i].fitness[0]
        if P[i].fitness[1]<min_2:
            min_2=P[i].fitness[1]
            z_star[1]=P[i].fitness[1]

    return z_star
def look_neighbor(lambda_, T):
    B = []
    for i in range(len(lambda_)):
        temp = []
        for j in range(len(lambda_)):
            distance = np.sqrt((lambda_[i][0] - lambda_[j][0]) ** 2 + (lambda_[i][1] - lambda_[j][1]) ** 2)
            temp.append(distance)
        l = np.argsort(temp)
        B.append(l[:T])
    return B




#----------------------------------
#-----------------参数输入--------------
def init(N,T,fun,f_num, x_num, x_min, x_max, PP ):
    A=[]
    P,lambda_=initialization(N,f_num,x_num,x_min,x_max,fun)
    P_fitness=np.zeros((N,f_num))
    for i in range(N):
        for j in range(f_num):
            P_fitness[i][j]=P[i].fitness[j]
    z_star=Caculateminobj(P,N,f_num)
    B=look_neighbor(lambda_,T)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(P_fitness)
    for i in range(N):
        if(dc[i]!=0):
            A.append(P_fitness[i])
    return P,A,B,z_star
