import numpy as np
import pygmo as pg
import math
import funcition
def crowding_distance(PC, f_num):
    for dim in range(f_num):
        new_list = sort_by_coordinate(PC, dim)
        new_list[0].crowding_distance = math.inf
        new_list[-1].crowding_distance = math.inf
        max_distance = new_list[0].fitness[dim] - new_list[-1].fitness[dim]
        for i in range(1, len(new_list) - 1):
            distance = new_list[i - 1].fitness[dim] - new_list[i + 1].fitness[dim]
            if max_distance == 0:
                new_list[i].crowding_distance = 0
            else:
                new_list[i].crowding_distance += distance / max_distance
    return new_list


def sort_by_coordinate(population, dim):  # selection sort, which can be replaced with quick sort
    p_list = []
    for p in population:
        p_list.append(p)

    for i in range(0, len(p_list) - 1):
        for j in range(i + 1, len(p_list)):
            if p_list.fitness[dim] < p_list.fitness[dim]:
                temp = p_list[i]
                p_list[i] = p_list[j]
                p_list[j] = temp

    return p_list
c=pg.crowding_distance([[0,0],[-1,1],[2,-2]])
x=[]
for i in [0,0],[-1,1],[2,-2]:
    x.append(funcition.partical(i, 'ZDT1', 30))
d=crowding_distance(x,2)
