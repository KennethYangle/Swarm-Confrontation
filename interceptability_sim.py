import numpy as np
import matplotlib.pyplot as plt
import copy
from regex import V0

class RigidBody:
    def __init__(self, x0=[0., 0., 0.], v0=[0., 0., 0.], mu=0, am=10) -> None:
        self.x = np.array(x0)
        self.v = np.array(v0)
        self.mu = mu
        self.am = am

        self.traj = copy.deepcopy(self.x)

    def update(self, dt):
        no = -self.x / np.linalg.norm(self.x)
        theta = np.arctan2(no[1], no[0]) - self.mu

        # v_hor = self.v.dot(no) * no
        # v_ver = self.v - v_hor
        self.a = am * np.array([np.cos(theta), np.sin(theta), 0.])
        # print(self.x, self.v, a)

        self.x += self.v * dt
        self.v += self.a * dt
        self.traj = np.vstack((self.traj, self.x))

    def plot(self):
        plt.scatter(0, 0)
        plt.scatter(self.x[0], self.x[1])
        plt.arrow(self.x[0], self.x[1], self.a[0], self.a[1])   # a
        plt.arrow(self.x[0], self.x[1], self.v[0], self.v[1])   # v
        plt.plot(self.traj[:,0], self.traj[:,1])
        scale = max(abs(self.x)//10)*10 + 10
        plt.xlim((-scale, scale))
        plt.ylim((-scale, scale))

if __name__ == "__main__":
    x0 = [10., 0., 0.]
    v0 = [1., 1., 0.]
    mu = np.pi/6
    am = 10
    dt = 0.01

    rb = RigidBody(x0=x0, v0=v0, mu=mu, am=am)
    plt.ion()  #打开交互模式
    while True:
        plt.clf()  #清除图像
        rb.update(dt)
        rb.plot()
        plt.pause(dt)
        plt.show()
