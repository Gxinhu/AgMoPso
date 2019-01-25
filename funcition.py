import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import math

def funcitions(fun):
    if fun == 'ZDT1':
        f_num = 2;  # 目标函数个数
        x_num = 30;  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt1 = np.loadtxt('ZDT1.txt')
        plt.scatter(zdt1[:, 0], zdt1[:, 1], marker='o', color='green', s=40)
        PP = zdt1
    elif fun == 'ZDT2':
        f_num = 2;  # 目标函数个数
        x_num = 10;  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt2 = np.loadtxt('ZDT2.txt')
        plt.scatter(zdt2[:, 0], zdt2[:, 1], marker='o', color='green', s=40)
        PP = zdt2
    elif fun == 'ZDT3':
        f_num = 2;  # 目标函数个数
        x_num = 30;  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt3 = np.loadtxt('ZDT3.txt')
        plt.scatter(zdt3[:, 0], zdt3[:, 1], marker='o', color='green', s=40)
        PP = zdt3
    elif fun == 'ZDT4':
        f_num = 2;  # 目标函数个数
        x_num = 10;  # 决策变量个数
        x_min = np.array([[0, -5, -5, -5, -5, -5, -5, -5, -5, -5]], dtype=float)  # 决策变量的最小值
        x_max = np.array([[1, 5, 5, 5, 5, 5, 5, 5, 5, 5]], dtype=float)  # 决策变量的最大值
        zdt4 = np.loadtxt('ZDT4.txt')
        plt.scatter(zdt4[:, 0], zdt4[:, 1], marker='o', color='green', s=40)
        PP = zdt4
    elif fun == 'ZDT6':
        f_num = 2;  # 目标函数个数
        x_num = 10;  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        zdt6 = np.loadtxt('ZDT6.txt')
        plt.scatter(zdt6[:, 0], zdt6[:, 1], marker='o', color='green', s=40)
        PP = zdt6
    elif fun == 'DTLZ1':
        f_num = 3;  # 目标函数个数
        x_num = 10;  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        dtlz1 = np.loadtxt('DTLZ1.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz1[:, 0], dtlz1[:, 1], dtlz1[:, 2], c='g')
        PP = dtlz1
    elif fun == 'DTLZ2':
        f_num = 3;  # 目标函数个数
        x_num = 10;  # 决策变量个数
        x_min = np.zeros((1, x_num))  # 决策变量的最小值
        x_max = np.ones((1, x_num))  # 决策变量的最大值
        dtlz2 = np.loadtxt('DTLZ2.txt')
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(dtlz2[:, 0], dtlz2[:, 1], dtlz2[:, 2], c='g')
        PP = dtlz2
    return f_num, x_num, x_min, x_max, PP
class partical():
    def __init__(self,x,fun,x_num):
        self.x=x
        if (fun == 'ZDT1'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** (0.5))
            self.fitness = [f1, f2]
            self.crowding_distance=0
            self.v=np.random.uniform(low=0,high=1,size=(x_num,))
        elif (fun == 'ZDT2'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** 2)
            self.fitness = [f1, f2]
            self.crowding_distance = 0
        elif (fun == 'ZDT3'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * (sum1 / (x_num - 1)))
            f2 = g * (1 - (f1 / g) ** (0.5) - (f1 / g) * math.sin(10 * math.pi * f1))
            self.fitness = [f1, f2]
            self.crowding_distance = 0
        elif (fun == 'ZDT4'):
            f1 = float(x[0])
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + (x[i + 1]) ** 2 - 10 * math.cos(4 * math.pi * x[i + 1])
            g = float(1 + 9 * 10 + sum1)
            f2 = g * (1 - (f1 / g) ** (0.5))
            self.fitness = [f1, f2]
            self.crowding_distance = 0
        elif (fun == 'ZDT6'):
            f1 = float(1 - math.exp(-4 * x[0]) * (math.sin(6 * math.pi * x[0])) ** 6)
            sum1 = 0.0
            for i in range(x_num - 1):
                sum1 = sum1 + x[i + 1]
            g = float(1 + 9 * ((sum1 / (x_num - 1)) ** (0.25)))
            f2 = g * (1 - (f1 / g) ** 2)
            self.fitness = [f1, f2]
            self.crowding_distance = 0
        elif (fun == 'DTLZ1'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i + 2] - 0.5))
            g = float(100 * (x_num - 2) + 100 * sum1)
            f1 = float((1 + g) * x[0] * x[1])
            f2 = float((1 + g) * x[0] * (1 - x[1]))
            f3 = float((1 + g) * (1 - x[0]))
            self.f = [f1, f2, f3]
            self.crowding_distance = 0
        elif (fun == 'DTLZ2'):
            sum1 = 0.0
            for i in range(x_num - 2):
                sum1 = sum1 + (x[i + 2]) ** 2
            g = float(sum1)
            f1 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.cos(0.5 * math.pi * x[1]))
            f2 = float((1 + g) * math.cos(0.5 * math.pi * x[0]) * math.sin(0.5 * math.pi * x[1]))
            f3 = float((1 + g) * math.sin(0.5 * math.pi * x[0]))
            self.f = [f1, f2, f3]