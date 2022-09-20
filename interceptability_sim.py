from cProfile import label
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import copy

class RigidBody:
    def __init__(self, x0=[0., 0., 0.], v0=[0., 0., 0.], m=1.0, tau_m=10, tau_tm=10) -> None:
        self.x = np.array(x0)
        self.v = np.array(v0)
        self.at = np.array([0., 0., 0.])
        self.m = m
        self.tau_m = tau_m
        self.tau_tm = tau_tm
        self.g = np.array([0., 9.8, 0.])
        self.k1 = 1.0
        self.k2 = 5.0

        self.traj = copy.deepcopy(self.x)

    def update(self, dt):
        nt = -self.x / np.linalg.norm(self.x)

        self.a_net = self.k1 * (self.k2*nt - self.v)
        self.f_net = self.m * self.a_net
        self.f = sat(self.f_net - self.m*self.g + self.m*self.at, self.tau_m*self.m*np.linalg.norm(self.g))
        self.f_net = self.m*self.g + self.f - self.m*self.at
        self.a_net = self.f_net / self.m

        max_int = -1e10
        for ii in range(3600):
            rad = ii / 3600 * 2 * np.pi
            f = self.tau_m*self.m*np.linalg.norm(self.g) * np.array([np.cos(rad), np.sin(rad), 0])
            f_net = self.m*self.g + f - self.m*self.at
            f_net_hor = nt.dot(f_net) * nt
            f_net_ver = f_net - f_net_hor
            # 条件1：f_net_ver和v在nt两侧；条件2：垂直方向f_net_ver收敛力度大于80倍夹角而小于1（上限，防溢出）；条件3：f_net_hor与nt同向；条件4：找满足前面条件的最大f_net_hor前向收敛力度
            if np.dot(np.cross(f_net_ver, nt), np.cross(self.v, nt)) < 0 and np.linalg.norm(f_net_ver) > min(80*np.arccos(self.v.dot(nt)/np.linalg.norm(self.v)), 1) and f_net_hor[0]/nt[0] > 0 and np.linalg.norm(f_net_hor) > max_int:
                self.f = f
                self.f_net = f_net
                self.a_net = self.f_net / self.m
                max_int = np.linalg.norm(f_net_hor)

        min_esc = 1e10
        for ii in range(3600):
            rad = ii / 3600 * 2 * np.pi
            at = self.tau_tm * np.linalg.norm(self.g) * np.array([np.cos(rad), np.sin(rad), 0])
            esc = nt.dot(self.m*self.g + self.f - self.m*at)
            if esc < min_esc:
                self.at = at
                min_esc = esc
        # print(self.x, self.v, a)


        self.x += self.v * dt
        self.v += self.a_net * dt
        self.traj = np.vstack((self.traj, self.x))

    def plot(self):
        plt.scatter(0, 0, color='#b3b3b3')
        plt.scatter(self.x[0], self.x[1], color='#000000')
        plt.arrow(self.x[0], self.x[1], self.f_net[0], self.f_net[1], width=0.1, color='#983f1d')   # f_net
        plt.text(self.x[0]+self.f_net[0]-0.2, self.x[1]+self.f_net[1]-1, r"${\bf{f}}_{\rm{net}}$")
        plt.arrow(self.x[0], self.x[1], self.f[0]/2, self.f[1]/2, width=0.1, color='#31859b')   # f
        plt.text(self.x[0]+self.f[0]/2-0.2, self.x[1]+self.f[1]/2-1, r"${\bf{f}}$")
        plt.arrow(self.x[0], self.x[1], self.v[0], self.v[1], width=0.1, color='#205867')   # v
        plt.text(self.x[0]+self.v[0]-0.2, self.x[1]+self.v[1]+0.5, r"$\bf{v}$")
        plt.arrow(0, 0, self.at[0], self.at[1], width=0.1, color='#eb7224')   # at
        plt.text(self.at[0]-0.2, self.at[1]+0.5, r"${\bf{a}}_{\rm{t}}$")

        plt.plot(self.traj[:,0], self.traj[:,1])
        scale = max(abs(self.x)//10)*10 + 10
        plt.xlim((-scale, scale))
        plt.ylim((-scale, scale))

def sat(a, s):
    n = np.linalg.norm(a)
    if n > s:
        return a / n * s
    else:
        return a

if __name__ == "__main__":
    x0 = [10., 5., 0.]
    v0 = [-1., 0.5, 0.]
    m = 1.0
    tau_m = 2     # 2可拦截；1.3刚好逃逸；0.8快速逃逸，加速能力小于目标
    tau_tm = 0.5
    dt = 0.01

    rb = RigidBody(x0=x0, v0=v0, m=m, tau_m=tau_m, tau_tm=tau_tm)
    plt.ion()  #打开交互模式
    while True:
        plt.clf()  #清除图像
        rb.update(dt)
        rb.plot()
        plt.pause(dt)
        plt.show()