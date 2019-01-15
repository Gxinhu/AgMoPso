import Initialization
import funcition
import Immune_Search
def main():
    N = 300  # 种群规模
    NC=int(N/5)  #免疫搜索用的种群规模
    T = 20  # 邻域规模
    fun = 'ZDT2'  # 测试函数DTLZ2
    f_num, x_num, x_min, x_max, PP =funcition.funcitions(fun)
    max_gen = 250  # 最大进化代数
    pc = 1  # 交叉概率
    pm = 1 / x_num  # 变异概率
    P, A, B, z_star=Initialization.init(N,T,fun,f_num, x_num, x_min, x_max, PP )
    gen=0
    while gen<max_gen:
        S=Immune_Search.immune_search(A,NC,N)







if __name__ == '__main__':
    main()