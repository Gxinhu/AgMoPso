from mpl_toolkits.mplot3d import Axes3D

import Initialization
import funcition
import Immune_Search
import Archive_Update
import random
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import math


def main(fun):
    N = 100 # 种群规模
    NC = int(N / 5)  # 免疫搜索用的种群规模
    T = 20  # 邻域规模
    #fun = 'ZDT1'  # 测试函数DTLZ2
    f_num, x_num, x_min, x_max, PP = funcition.funcitions(fun)
    max_gen = 3000*N  # 最大进化代数
    pc = 0.9  # 交叉概率
    w = 0.1
    pm = 1 / x_num  # 变异概率
    P, A, B, z_star, lambda_ = Initialization.init(N, T, fun, f_num, x_num, x_min, x_max, PP)
    gen = 0
    while gen < max_gen:
        S = Immune_Search.immune_search(A, NC, N, pc, pm, x_min, x_max, f_num, fun)
        gen +=len(S)
        A = Archive_Update.archive_update(S, A, N, f_num)
        # z_star =  Initialization.Caculateminobj(A, len(A), f_num)
        # d1,leader=find_leader(N,A,lambda_,z_star)
        # for i in range(N):
        #     sign1=1
        #     sign2=1
        #     pbest = A[int(leader[i])]
        #     l = random.randint(0, len(A) - 1)
        #     gbest = A[l]
        #     ll=random.randint(0,len(B[i])-1)
        #     lbest=A[int(leader[B[i][ll]])]
        #     if pbi(lbest.fitness, lambda_[i],z_star) > pbi(P[i].fitness,lambda_[i],z_star):
        #         sign1 = -1.0
        #     if pbi(gbest.fitness, lambda_[i], z_star) > pbi(P[i].fitness, lambda_[i], z_star):
        #         sign2 = -1.0
        #     v_=np.zeros(len(P[i].v))
        #     for k in range(len(P[i].x)):
        #         C1=random.random()
        #         R1=random.uniform(1.5,2.0)
        #         v_[k] = w * P[i].v[k] + C1*R1* (pbest.x[k] - P[i].x[k]) +sign1*0.5 * (lbest.x[k]-P[i].x[k])+sign2*0.5*(gbest.x[k]-P[i].x[k])
        #         if v_[k]>x_max[k]:
        #             v_[k]=x_min[k]
        #         if (v_[k])<x_min[k]:
        #             v_[k]=x_min[k]
        #     P[i].v=v_
        #     P[i].x = P[i].x + P[i].v
        #     P[i].x = np.clip(P[i].x, a_min=x_min, a_max=x_max)
        #     P[i].fitness = funcition.partical(P[i].x, fun, x_num).fitness
        #     gen += 1
        #     z_star = updatereference(P[i], f_num,z_star)
        print(gen, len(A))
        # A = Archive_Update.archive_update(P, A, N, f_num)
    show(A, f_num)
    # --------------------Coverage(C-metric)---------------------
    PP = np.loadtxt('%s.pf' % (fun))
    B =A
    print(IGD(B, PP))


def IGD(A, PP):
    sums = 0
    min = math.inf
    for i in range(len(PP)):
        for j in range(len(A)):
            temp = math.sqrt(sum((PP[i] - A[j].fitness) ** 2))
            if temp < min:
                min = temp
        sums += min
        min = math.inf
    return sums / len(PP)


def Tchebycheff(x, lamb, z):
    temp = []
    for i in range(len(x)):
        temp.append(np.abs(x[i] - z[i]) * lamb[i])
    return np.max(temp)


def pbi(fitness, lambda_, z_star):
    Normlambda = math.sqrt(sum(lambda_ ** 2))
    d1 = math.sqrt(sum(((fitness - z_star) * lambda_)) ** 2) / Normlambda

    d2 = math.sqrt(sum((fitness - z_star - (d1 / Normlambda) * lambda_) ** 2))
    return d1, d1 + 0.5 * d2


def show(A, f_num):
    x = []
    y = []
    z = []
    if f_num == 2:
        for i in range(len(A)):
            x.append(A[i].fitness[0])
            y.append(A[i].fitness[1])
        plt.scatter(x, y, marker='o', color='red', s=40)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.show()
        plt.close('all')
    elif f_num == 3:
        for i in range(len(A)):
            x.append(A[i].fitness[0])
            y.append(A[i].fitness[1])
            z.append(A[i].fitness[2])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c='r')
        plt.show(elev=45, azim=45)
        plt.close("all")

def updatereference(P,f_num,z_stars):
    z_star = np.zeros(f_num)
    if f_num==2:
        min_1 = z_stars[0]
        min_2 = z_stars[1]
        if P.fitness[0] < min_1:
            z_star[0] = P.fitness[0]
        if P.fitness[1] < min_2:
            z_star[1] = P.fitness[1]
    if f_num==3:
        min_1 = z_stars[0]
        min_2 = z_stars[1]
        min_3=z_stars[2]
        if P.fitness[0] < min_1:
            z_star[0] = P.fitness[0]
        if P.fitness[1] < min_2:
            z_star[1] = P.fitness[1]
        if P.fitness[2]<min_3:
            z_stars[2]=P.fitness[2]


    return z_star


def Dominate(x, y, min=True):
    if min:

        for i in range(len(x.fitness)):
            if x.fitness[i] > y.fitness[i]:
                return False
        return True
    else:
        for i in range(len(x.fitness)):
            if x.fitness[i] < y.fitness[i]:
                return False
        return True
def find_leader(N,A,lambda_,z_star):
    d1=np.zeros(N)
    leader_ind=np.zeros(N)
    for i in range(N):
        best_ind=-1
        minFit=math.inf
        for j in range(len(A)):
            d,fitnesse=pbi(A[j].fitness,lambda_[i], z_star)
            if fitnesse<minFit:
                minFit=fitnesse
                best_ind=j
                bestd=d
        leader_ind[i]=best_ind
        d1[i]=bestd
    return d1,leader_ind

if __name__ == '__main__':
    # for i in range(1,8):
    #     if i !=9:
    #         fun="DTLZ"+str(i)
    #         main(fun)
    fun="UF1"
    main(fun)
