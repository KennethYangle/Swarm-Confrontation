import numpy as np
from scipy.optimize import linear_sum_assignment


# Recursive Hungarian Algorithm for Dynamic Environments
class RHA2:
    def __init__(self, agent_pos, agent_energy, task_pos, task_importance):
        self.agent_pos = agent_pos
        self.agent_energy = agent_energy
        self.task_pos = task_pos
        self.task_importance = task_importance

        col = []
        for i in range(len(self.task_pos)):
            if all(self.task_pos[i] == np.array([-1, -1, -1])):
                col.append(i)
        self.task_pos = np.delete(self.task_pos, col, axis=0)
        self.task_importance = np.delete(self.task_importance, col, axis=0)

        self.m = len(self.agent_pos)
        self.n = len(self.task_pos)
        self.oo = 0x3f3f3f3f

    def deal(self):
        # 初始化
        useful_m = self.m
        max_mn = max(self.m, self.n)
        cost = np.zeros([max_mn, max_mn])
        for i in range(self.m):
            for j in range(self.n):
                dis = np.linalg.norm(self.agent_pos[i]-self.task_pos[j])
                if dis > self.agent_energy[i]:
                    cost[i][j] = self.oo
                else:
                    cost[i][j] = 1 / (self.task_importance[j] - dis)    # cost为重要度-距离，因为越小越好所以取倒数
        uavs_pos_record = []
        uavs_pos_record.append(self.agent_pos.copy())

        # 开始
        # 1. 飞机数小于等于任务数，多轮分配，一个飞机执行多个任务
        if self.m <= self.n:
            while useful_m <= self.n:
                # 添加虚拟agent，补齐cost矩阵
                for i in range(self.m, self.n):
                    for j in range(self.n):
                        cost[i][j] = self.oo

                col_del = []
                row_ind, col_ind = linear_sum_assignment(cost)
                for i in range(self.m):
                    if cost[i][col_ind[i]] < self.oo:
                        # 更新能量
                        self.agent_energy[i] -= np.linalg.norm(self.agent_pos[i]-self.task_pos[col_ind[i]])
                        # 更新位置
                        self.agent_pos[i] = self.task_pos[col_ind[i]]
                        col_del.append(col_ind[i])
                    else:
                        useful_m -= 1
                print(self.agent_energy)
                self.task_pos = np.delete(
                    self.task_pos, col_del, axis=0)     # 更新task
                self.task_importance = np.delete(
                    self.task_importance, col_del, axis=0)     # 更新importance
                self.n = self.n - len(col_del)
                uavs_pos_record.append(self.agent_pos.copy())

                # 更新代价矩阵
                max_mn = max(self.m, self.n)
                cost = np.zeros([max_mn, max_mn])
                for i in range(self.m):
                    for j in range(self.n):
                        dis = np.linalg.norm(self.agent_pos[i]-self.task_pos[j])
                        if dis > self.agent_energy[i]:
                            cost[i][j] = self.oo
                        else:
                            cost[i][j] = 1 / (self.task_importance[j] - dis)

            # 剩余几个任务，不足以分配给所有飞机
            # 添加虚拟task，补齐cost矩阵
            for i in range(self.m):
                for j in range(self.n, self.m):
                    cost[i][j] = self.oo

            row_ind, col_ind = linear_sum_assignment(cost)
            tmp = np.zeros(self.agent_pos.shape)
            for i in range(self.m):
                if col_ind[i] < self.n:
                    tmp[i] = self.task_pos[col_ind[i]]   # 更新agent位置
                else:
                    tmp[i] = self.agent_pos[i]
            # self.agent_pos = self.task_pos[col_ind[:self.m]]
            uavs_pos_record.append(tmp.copy())


        # 2. 飞机数大于任务数，多个飞机执行一个任务
        else:
            k = self.m // self.n
            tmp = np.zeros(self.agent_pos.shape)
            for t in range(k):
                row_ind, col_ind = linear_sum_assignment(cost[t*self.n:(t+1)*self.n,:self.n])
                tmp[t*self.n:(t+1)*self.n] = self.task_pos[col_ind[:]]

            if self.m%self.n != 0:
                cost_res = np.zeros([self.n, self.n])
                cost_res[:self.m%self.n] = cost[k*self.n:, :self.n]
                cost_res[self.m%self.n:] = self.oo * np.ones([self.n-self.m%self.n, self.n])
                row_ind, col_ind = linear_sum_assignment(cost_res)
                tmp[k*self.n:] = self.task_pos[col_ind[:self.m%self.n]]

            self.agent_pos = tmp
            uavs_pos_record.append(tmp.copy())

        return uavs_pos_record


def gen_task_lists(task_pos, uavs_pos_record):
    task_dict = dict()
    for i in range(len(task_pos)):
        task_dict[tuple(task_pos[i])] = i
    task_lists = [[] for i in range(len(uavs_pos_record[0]))]
    for i in range(len(uavs_pos_record[0])):
        for j in range(1, len(uavs_pos_record)):
            t = task_dict[tuple(uavs_pos_record[j][i])]
            if t not in task_lists[i]:
                task_lists[i].append(t)
    return task_lists


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    agent_pos = np.array([(-19, -4, 0), (-1, -1, -1), (-3, 13, 0), (14, -15, 0), (16, 0, 0)])
    agent_energy = np.array([100, 100, 100, 30, 30])
    task_pos = np.array([(-13, 8, 0), (-12, 20, 0), (4, 4, 0), (18, -12, 0), (4, -19, 0),
                          (-19, 11, 0), (19, -8, 0), (-1, 9, 0), (-8, -8, 0), (11, -6, 0),
                          (-18, -17, 0), (-7, -16, 0), (12, 4, 0), (7, -1, 0)])
    task_importance = np.array([110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 200, 201, 202, 203])
    m, n = len(agent_pos), len(task_pos)

    r = RHA2(agent_pos, agent_energy, task_pos, task_importance)
    uavs_pos_record = r.deal()
    print(uavs_pos_record)
    task_lists = gen_task_lists(task_pos, uavs_pos_record)
    print(task_lists)
    path_len = len(uavs_pos_record)
    path_x = np.empty([m, path_len])
    path_y = np.empty([m, path_len])
    for i in range(path_len):
        for j in range(m):
            path_x[j][i] = uavs_pos_record[i][j][0]
            path_y[j][i] = uavs_pos_record[i][j][1]
    for i in range(m):
        plt.plot(path_x[i], path_y[i])
    for i in range(m):
        plt.annotate('U{}'.format(i),
                     xy=(uavs_pos_record[0][i][0],
                         uavs_pos_record[0][i][1]), xycoords='data',
                     xytext=(0, +5), textcoords='offset points', fontsize=10)
    for i in range(n):
        plt.annotate('{}'.format(i),
                     xy=(task_pos[i][0],
                         task_pos[i][1]), xycoords='data',
                     xytext=(0, +5), textcoords='offset points', fontsize=10)
    plt.show()
