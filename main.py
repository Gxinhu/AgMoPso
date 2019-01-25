import Initialization
import funcition
import Immune_Search
import Archive_Update
import random
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import math
def main():
    N = 100  # 种群规模
    NC=int(N/5)  #免疫搜索用的种群规模
    T = 20  # 邻域规模
    fun = 'ZDT3'  # 测试函数DTLZ2
    f_num, x_num, x_min, x_max, PP =funcition.funcitions(fun)
    max_gen = 25000 # 最大进化代数
    pc = 0.9  # 交叉概率
    w=0.2
    pm = 1 / x_num  # 变异概率
    P, A, B, z_star,lambda_=Initialization.init(N,T,fun,f_num, x_num, x_min, x_max, PP )
    gen=0
    while gen<max_gen:
        S=Immune_Search.immune_search(A,NC,N,pc,pm,x_min,x_max,f_num,fun)
        gen+=len(S)
        A=Archive_Update.archive_update(S,A,N,f_num)
        for i in range(N):
            m = random.randint(0, T - 1)
            l = random.randint(0, T - 1)
            y1,y2=sbxpm(P[B[i][m]],P[B[i][l]],pc,pm,x_min,x_max,fun)
            if Dominate(y1,y2):
                y=y1
            else:
                y=y2
            for j in range(len(z_star)):
                if y.fitness[j]<z_star[j]:
                    z_star[j]=y.fitness[j]
            for j in range(len(B[i])):
                Ta = pbi(P[B[i][j]].fitness, lambda_[B[i][j]], z_star)
                Tb = pbi(y.fitness, lambda_[B[i][j]], z_star)
                if Tb <=Ta:
                    P[B[i][j]] = y
            # if A == []:
            #     A.append(y)
            # else:
            #     dominateY = False
            #     rmlist = []
            #     for j in range(len(A)):
            #         if Dominate(y, A[j]):
            #             rmlist.append(A[j])
            #         elif Dominate(A[j], y):
            #             dominateY = True
            #
            #     if dominateY == False:
            #         A.append(y)
            #         for j in range(len(rmlist)):
            #             A.remove(rmlist[j])

            gen+=1
        A = Archive_Update.archive_update(P, A, N, f_num)
        print(gen)
    show(A,f_num)





        # z_star=Initialization.Caculateminobj(A,len(A),f_num)
        # for i in range(N):
        #     pbest=A[0]
        #     for j in range(2,len(A)):
        #         d1_1,temp_a=pbi(pbest.fitness,lambda_[i],z_star)
        #         d1_2,temp_b=pbi(A[j].fitness,lambda_[i],z_star)
        #         if(temp_a>temp_b):
        #             pbest=A[j]
        #     # k=random.randint(0,T-1)
        #     # lbest=A[B[i][k]]
        #     l=random.randint(0,len(A)-1)
        #     gbest=A[l]
        #     P[i].v=w*P[i].v+d1_1*(pbest.x-P[i].x)+0.5*(gbest.x-P[i].x)
        #     P[i].x=P[i].x+P[i].v
        #     P[i].x=np.clip(P[i].x, a_min=x_min, a_max=x_max)[0]
        #     P[i].fitness=funcition.partical(P[i].x,fun,x_num).fitness
        #     gen+=1
        # A=Archive_Update.archive_update(P,A,N,f_num)
        #z_star = Initialization.Caculateminobj(A, len(A), f_num)
# --------------------Coverage(C-metric)---------------------
    PP=np.loadtxt('ZDT1.txt')
    B = A
    number = 0
    for i in range(len(B)):
        nn = 0
        for j in range(len(PP)):
            if (pg.pareto_dominance(PP[j], B[i].fitness)):
                nn = nn + 1  # B[i]被A支配的个体数目+1
        if (nn != 0):
            number = number + 1
    C_AB = float(number / len(B))
    print("C_AB：%2f" % C_AB)


def Tchebycheff(x, lamb, z):
    temp = []
    for i in range(len(x)):
        temp.append(np.abs(x[i] - z[i]) * lamb[i])
    return np.max(temp)
def pbi(fitness,lambda_,z_star):
    Normlambda=math.sqrt(sum(lambda_**2))
    d1=math.sqrt(sum(((fitness-z_star)*lambda_))**2)/Normlambda

    d2=math.sqrt(sum((fitness-z_star-(d1/Normlambda)*lambda_)**2))
    return d1,d1+0.5*d2
def show(A,f_num):
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
        plt.close()
def sbxpm(x,y,pc,pm,x_min,x_max,fun):
    k = random.randint(0, 1)
    l=abs(k-1)
    if random.random() < pc:
        C = np.zeros((2, len(x.x)))
        C[0], C[1] = Immune_Search.SBX(x.x, y.x, x_min, x_max)
        if random.random() < pm:
            z = Immune_Search.PM(C[k], x_min, x_max)
            return funcition.partical(z, fun, len(x.x)),funcition.partical(C[l], fun, len(x.x))
        else:
            return funcition.partical(C[0], fun, len(x.x)),funcition.partical(C[1], fun, len(x.x))
    elif (random.random() < pm):
        return funcition.partical(Immune_Search.PM(x.x, x_min, x_max), fun, len(x.x)),funcition.partical(y.x, fun, len(x.x))
    else:
        return x,y
def Dominate(x,y,min=True):
    if min:

        for i in range(len(x.fitness)):
            if x.fitness[i]>y.fitness[i]:
                return False
        return True
    else:
        for i in range(len(x.fitness)):
            if x.fitness[i]<y.fitness[i]:
                return False
        return True
if __name__ == '__main__':
    main()