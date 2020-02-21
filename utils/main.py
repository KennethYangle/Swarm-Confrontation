import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# 初始位置
agent_pos = np.array([(-19,-4), (-4,4), (-3,13), (14,-15), (16,0)])
task_pos = np.array([(-13,8), (-12,20), (4,4), (18,-12), (4,-19),   (-19,11), (19,-8), (-1,9), (-9,-8), (11,-6),   (-18,-17), (-7,-16), (12,4), (7,-1), (4,6)])

m = agent_pos.__len__()
n = task_pos.__len__()

# 初始化
max_mn = max(m, n)
cost = np.zeros([max_mn, max_mn])
for i in range(m):
    for j in range(n):
        cost[i][j] = math.sqrt((agent_pos[i][0] - task_pos[j][0])*(agent_pos[i][0] - task_pos[j][0]) + (agent_pos[i][1] - task_pos[j][1])*(agent_pos[i][1] - task_pos[j][1]))
uavs_pos_record = []
uavs_pos_record.append(agent_pos)

# 开始
while m < n:
    # 添加虚拟agent，补齐cost矩阵
    for i in range(m, n):
        for j in range(n):
            cost[i][j] = 0x3f3f3f3f
    print("=================================\n")
    print(cost)

    row_ind,col_ind = linear_sum_assignment(cost)
    agent_pos = task_pos[col_ind[:m]]   # 更新agent位置
    task_pos = np.delete(task_pos, col_ind[:m], axis=0)     # 更新task
    n = n - m
    uavs_pos_record.append(agent_pos)
    print(agent_pos)
    print(row_ind)                      # 开销矩阵对应的行索引
    print(col_ind)                      # 对应行索引的最优指派的列索引
    print(cost[row_ind,col_ind])        # 提取每个行索引的最优指派列索引所在的元素，形成数组
    print(cost[row_ind,col_ind].sum())  # 数组求和

    # 更新代价矩阵
    max_mn = max(m, n)
    cost = np.zeros([max_mn, max_mn])
    for i in range(m):
        for j in range(n):
            cost[i][j] = math.sqrt((agent_pos[i][0] - task_pos[j][0])*(agent_pos[i][0] - task_pos[j][0]) + (agent_pos[i][1] - task_pos[j][1])*(agent_pos[i][1] - task_pos[j][1]))

if m >= n:
    # 添加虚拟task，补齐cost矩阵
    for i in range(m):
        for j in range(n, m):
            cost[i][j] = 0x3f3f3f3f
    print("=================================\n")
    print(cost)

    row_ind,col_ind = linear_sum_assignment(cost)
    agent_pos = task_pos[col_ind[:m]]   # 更新agent位置
    uavs_pos_record.append(agent_pos)
    print(agent_pos)
    print(row_ind)                      # 开销矩阵对应的行索引
    print(col_ind)                      # 对应行索引的最优指派的列索引
    print(cost[row_ind,col_ind])        # 提取每个行索引的最优指派列索引所在的元素，形成数组
    print(cost[row_ind,col_ind].sum())  # 数组求和

# 结果
print(uavs_pos_record)
path_len = uavs_pos_record.__len__()
path_x = np.empty([m, path_len])
path_y = np.empty([m, path_len])
for i in range(path_len):
    for j in range(m):
        path_x[j][i] = uavs_pos_record[i][j][0]
        path_y[j][i] = uavs_pos_record[i][j][1]
for i in range(m):
    plt.plot(path_x[i], path_y[i])
for i in range(m):
    plt.annotate('U',
         xy=(uavs_pos_record[0][i][0], uavs_pos_record[0][i][1]), xycoords='data',
         xytext=(0, +5), textcoords='offset points', fontsize=10)
plt.show()
