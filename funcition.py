import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import math


def funcitions(fun):
    if fun == 'ZDT1':
        f_num = 2 # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt1 = np.loadtxt('ZDT1.txt')
        plt.scatter(zdt1[:, 0], zdt1[:, 1], marker='o', color='green', s=40)
        PP = zdt1
    elif fun == 'ZDT2':
        f_num = 2  # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.zeros(x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('ZDT2.txt')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'ZDT3':
        f_num = 2  # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt3 = np.loadtxt('ZDT3.txt')
        plt.scatter(zdt3[:, 0], zdt3[:, 1], marker='o', color='green', s=40)
        PP = zdt3
    elif fun == 'ZDT4':
        f_num = 2  # 目标函数个数
        x_num = 10  # 决策变量个数
        x_min = np.array([0, -5,-5,-5,-5,-5,-5,-5,-5,-5], dtype=float)  # 决策变量的最小值
        x_max = np.array([1, 5,5,5,5,5,5,5,5,5], dtype=float)  # 决策变量的最大值
        zdt4 = np.loadtxt('ZDT4.txt')
        plt.scatter(zdt4[:, 0], zdt4[:, 1], marker='o', color='green', s=40)
        PP = zdt4
    elif fun == 'ZDT6':
        f_num = 2  # 目标函数个数
        x_num = 10  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones( x_num)  # 决策变量的最大值
        zdt6 = np.loadtxt('ZDT6.txt')
        plt.scatter(zdt6[:, 0], zdt6[:, 1], marker='o', color='green', s=40)
        PP = zdt6
    elif fun == 'DTLZ1':
        f_num =3  # 目标函数个数
        x_num = 12  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones( x_num) # 决策变量的最大值
        dtlz1 = np.loadtxt('DTLZ1.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz1[:, 0], dtlz1[:, 1], dtlz1[:, 2], c='g')
        PP = dtlz1
    elif fun == 'DTLZ2':
        f_num = 3  # 目标函数个数
        x_num = 10 # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        dtlz2 = np.loadtxt('DTLZ2.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz2[:, 0], dtlz2[:, 1], dtlz2[:, 2], c='g')
        PP = dtlz2
    elif fun == 'DTLZ3':
        f_num = 3  # 目标函数个数
        x_num = 12  # 决策变量个数
        x_min = np.zeros(x_num)  # 决策变量的最小值
        x_max = np.ones( x_num)  # 决策变量的最大值
        dtlz3 = np.loadtxt('DTLZ3.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz3[:, 0], dtlz3[:, 1], dtlz3[:, 2], c='g')
        PP = dtlz3
    elif fun == 'DTLZ4':
        f_num = 3  # 目标函数个数
        x_num = 12  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones( x_num)  # 决策变量的最大值
        dtlz4 = np.loadtxt('DTLZ4.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz4[:, 0], dtlz4[:, 1], dtlz4[:, 2], c='g')
        PP = dtlz4
    elif fun == 'DTLZ5':
        f_num = 3  # 目标函数个数
        x_num = 12  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        dtlz5 = np.loadtxt('DTLZ5.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz5[:, 0], dtlz5[:, 1], dtlz5[:, 2], c='g')
        PP = dtlz5
    elif fun == 'DTLZ6':
        f_num = 3  # 目标函数个数
        x_num = 12  # 决策变量个数
        x_min = np.zeros(x_num) # 决策变量的最小值
        x_max = np.ones( x_num)  # 决策变量的最大值
        dtlz6 = np.loadtxt('DTLZ6.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz6[:, 0], dtlz6[:, 1], dtlz6[:, 2], c='g')
        PP = dtlz6
    elif fun == 'DTLZ7':
        f_num = 3  # 目标函数个数
        x_num = 22  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones( x_num)  # 决策变量的最大值
        dtlz7 = np.loadtxt('DTLZ7.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz7[:, 0], dtlz7[:, 1], dtlz7[:, 2], c='g')
        PP = dtlz7
    elif fun == 'WFG1':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros(x_num) # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG1.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG2':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros(x_num)  # 决策变量的最小值
        x_max = np.ones( x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG2.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG3':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG3.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG4':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG4.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG5':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG5.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG6':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG6.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG7':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG7.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG8':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG8.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'WFG9':
        f_num = 2  # 目标函数个数
        x_num = 8  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('WFG9.3D.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF1':
        f_num = 2  # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.full( x_num,-1)  # 决策变量的最小值
        x_min[0]=0
        x_max = np.ones( x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('UF1.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF2':
        f_num = 2  # 目标函数个数
        x_num = 30 # 决策变量个数
        x_min = np.full(x_num, -1)  # 决策变量的最小值
        x_min[0] = 0
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('UF2.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF3':
        f_num = 2  # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.zeros( x_num)  # 决策变量的最小值
        x_max = np.ones(x_num)  # 决策变量的最大值
        zdt2 = np.loadtxt('UF3.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF4':
        f_num = 2  # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF4.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF5':
        f_num = 2  # 目标函数个数
        x_num = 30  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF5.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF6':
        f_num = 2  # 目标函数个数
        x_num = 30   # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF6.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF7':
        f_num = 2  # 目标函数个数
        x_num = 30   # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF7.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF8':
        f_num = 2  # 目标函数个数
        x_num = 30   # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF8.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'UF9':
        f_num = 2  # 目标函数个数
        x_num = 30   # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF9.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2

    elif fun == 'UF10':
        f_num = 2  # 目标函数个数
        x_num = 30   # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('UF10.pf')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2

    return f_num, x_num, x_min, x_max, PP


class partical():
    def __init__(self, x, fun, x_num):
        self.x = x
        if (fun == 'ZDT1'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** (0.5))
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'ZDT2'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** 2)
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'ZDT3'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** (0.5) - (f1 / g) * math.sin(10 * math.pi * f1))
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'ZDT4'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + (x[i + 1]) ** 2 - 10 * math.cos(4 * math.pi * x[i + 1])
            g = float(1 + (len(x)-1)* 10 + sum1)
            f2 = g * (1 - (f1 / g) ** (0.5))
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'ZDT6'):
            f1 = float(1 - math.exp(-4 * x[0]) * (math.sin(6 * math.pi * x[0])) ** 6)
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * ((sum1 / (x_num - 1)) ** (0.25)))
            f2 = g * (1 - (f1 / g) ** 2)
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'DTLZ1'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i + 2] - 0.5))
            g = float(100 * (x_num - 2 + sum1))
            f1 = float(0.5*(1 + g) * x[0] * x[1])
            f2 = float(0.5*(1 + g) * x[0] * (1 - x[1]))
            f3 = float(0.5*(1 + g) * (1 - x[0]))
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v =np.zeros(x_num)
        elif (fun == 'DTLZ2'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2]-0.5) ** 2
            g = float(sum1)
            f1 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.cos(0.5 * math.pi * x[1]))
            f2 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.sin(0.5 * math.pi * x[1]))
            f3 = float((1 + g) * math.sin(0.5 * math.pi * x[0]))
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'DTLZ3'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i + 2] - 0.5))
            g = float(100 * (x_num - 2) + 100 * sum1)
            f1 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.cos(0.5 * math.pi * x[1]))
            f2 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.sin(0.5 * math.pi * x[1]))
            f3 = float((1 + g) * math.sin(0.5 * math.pi * x[0]))
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'DTLZ4'):
            sum1 = 0.0
            a=100
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2] - 0.5) ** 2
            g = float(sum1)
            f1 = float((1 + g) * math.cos(0.5  * pow(x[0],a) * math.pi ) * math.cos(0.5 * math.pi * pow(x[1],a)))
            f2 = float((1 + g) * math.cos(0.5 * math.pi * pow(x[0],a)) * math.sin(0.5 * math.pi   * pow(x[1],a)))
            f3 = float((1 + g) * math.sin(0.5 * math.pi * pow(x[0],a)))
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'DTLZ5'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2] - 0.5) ** 2
            g = float(sum1)
            t=math.pi/(4*(1+g))
            sita1=0.5 * math.pi*x[0]
            sita2=t*(1+2*x[1]*g)
            f1 = float((1 + g) * math.cos( sita1) * math.cos(sita2))
            f2 = float((1 + g) * math.cos(sita1) * math.sin( sita2))
            f3 = float((1 + g) * math.sin(sita1))
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'DTLZ6'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2]) ** 0.1
            g = float(sum1)
            t = math.pi / (4 * (1 + g))
            sita1 = 0.5 * math.pi * x[0]
            sita2 = t * (1 + 2 * x[1] * g)
            f1 = float((1 + g) * math.cos(sita1) * math.cos(sita2))
            f2 = float((1 + g) * math.cos(sita1) * math.sin(sita2))
            f3 = float((1 + g) * math.sin(sita1))
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'DTLZ7'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 +x[i + 2]
            g = float(1+9*sum1/22)
            f1 = x[0]
            f2 = x[1]
            h=3-(f1/(1+g))*(1+math.sin(3*math.pi*f1))-(f2/(1+g))*(1+math.sin(3*math.pi*f2))
            f3 = (1+g)*h
            self.fitness = [f1, f2, f3]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif(fun=='UF1'):
            count1=0
            count2=0
            sum1=0
            sum2=0
            for i in range(2,x_num):
                try:
                    yj=x[i-1]-math.sin(6*math.pi*x[0]+i*math.pi/x_num)
                except IndexError:
                    print(i,x)
                yj=yj*yj
                if i%2==0:
                    sum2+=yj
                    count2+=1
                else:
                    sum1+=yj
                    count1+=1
            f1=x[0]+2*sum1/count1
            f2=1-math.sqrt(x[0])+2.0*sum2/count2
            self.fitness=[f1,f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'UF2'):
            count1 = 0
            count2 = 0
            sum1 = 0
            sum2 = 0
            for j in range(2, x_num):
                yj = x[j - 1] - (0.3 * x[0] * x[0] * math.cos(24 * math.pi * x[0] + 4 * j * math.pi / x_num) + 0.6 * x[
                0]) * math.sin(6.0 * math.pi * x[0] + j * math.pi / x_num)
                if j % 2 == 0:

                    sum2 += yj*yj
                    count2 += 1
                else:
                    sum1 += yj*yj
                    count1 += 1
            f1 = x[0] + 2 * sum1 / count1
            f2 = 1 - math.sqrt(x[0]) + 2.0 * sum2 / count2
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'UF3'):
            count1 = 0
            count2 = 0
            sum1 = 0
            sum2 = 0
            prod1=1
            prod2=1
            for j in range(2, x_num):
                yj = x[j - 1] - math.pow(x[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (x_num - 2.0)))
                pj = math.cos(20.0 * yj * math.pi / math.sqrt(j))
                if j % 2 == 0:
                    prod2*=pj
                    sum2 += yj * yj
                    count2 += 1
                else:
                    sum1 += yj * yj
                    prod1*=pj
                    count1 += 1
            f1 = x[0] + 2.0*(4.0*sum1 - 2.0*prod1 + 2.0) /count1
            f2 = 1.0 - math.sqrt(x[0]) + 2.0*(4.0*sum2 - 2.0*prod2 + 2.0) / count2
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)
        elif (fun == 'UF4'):
            count1 = 0
            count2 = 0
            sum1 = 0
            sum2 = 0
            prod1=1
            prod2=1
            for j in range(2, x_num):
                yj = x[j - 1] - math.pow(x[0], 0.5 * (1.0 + 3.0 * (j - 2.0) / (x_num - 2.0)))
                pj = math.cos(20.0 * yj * math.pi / math.sqrt(j))
                if j % 2 == 0:
                    prod2*=pj
                    sum2 += yj * yj
                    count2 += 1
                else:
                    sum1 += yj * yj
                    prod1*=pj
                    count1 += 1
            f1 = x[0] + 2.0*(4.0*sum1 - 2.0*prod1 + 2.0) /count1
            f2 = 1.0 - math.sqrt(x[0]) + 2.0*(4.0*sum2 - 2.0*prod2 + 2.0) / count2
            self.fitness = [f1, f2]
            self.crowding_distance = 0
            self.v = np.zeros(x_num)



