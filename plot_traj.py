#绘制三角螺旋线
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pickle

f = open('pos_recoder_0.pkl', 'rb')
pos_recoder = pickle.load(f)
ax = plt.axes(projection='3d')

xdata = [a[0] for a in pos_recoder]
ydata = [a[1] for a in pos_recoder]
zdata = [-a[2] for a in pos_recoder]
ax.scatter(xdata, ydata, zdata, linewidths=1)
# #三维线的数据
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# # 三维散点的数据
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter(xdata, ydata, zdata, linewidths=zdata)


plt.show()