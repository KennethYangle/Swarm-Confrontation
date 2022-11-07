import numpy as np
import matplotlib.pyplot as plt
import copy

class RigidBody:
    def __init__(self, interceptor="MC", x0=[0., 0., 0.], v0=[0., 0., 0.], m=1.0, tau_m=10, tau_tm=10, fov=np.deg2rad(90), ang_install=np.deg2rad(30)) -> None:
        self.interceptor = interceptor      # MC, FW, or AM
        self.x = np.array(x0)
        self.v = np.array(v0)
        self.a_t = np.array([0., 0., 0.])
        self.a_d = np.array([0., 0., 0.])
        self.a_net = np.array([0., 0., 0.])
        self.a_fic = np.array([0., 0., 0.])
        self.m = m
        self.tau_m = tau_m
        self.tau_tm = tau_tm
        self.fov = fov
        self.ang_install = ang_install
        self.I = 0

        self.g = 9.8
        self.eps = 0.2
        self.penalty = -1e5

        self.traj = copy.deepcopy(self.x)

    def update(self, dt):
        # 0: 初始化
        n_t = np.array([1., 0., 0.])
        ang_n_t = vec2ang(n_t)
        lambda_fov = np.pi/2 - self.fov/2 - self.ang_install
        strategy_pair_stash = []     # 储存策略对，元素为[a_t, a_d, I]

        # step1: 对所有可行的a_t，找出对应的n_td和f_d
        for ii in range(-90,90):
            ang_a_t = ang_n_t + np.pi + np.deg2rad(ii)
            a_t = ang2vec(ang_a_t, self.tau_tm*self.g)

            # 找出此a_t下拦截器策略n_td和f_d
            strategy_interceptor_stash = []     # 储存策略对，元素为[a_t, a_d, I]
            for jj in range(-180, 180):
                # 多旋翼、导弹和固定翼条件不一样
                if (self.interceptor=="MC" or self.interceptor=="AM") and abs(jj) < lambda_fov:
                    continue
                if self.interceptor=="FW" and jj < lambda_fov:
                    continue

                ang_a_d = ang_n_t + np.deg2rad(jj)
                a_d = ang2vec(ang_a_d, self.tau_m*self.g)
                a_net = a_d - a_t
                if a_net[2] * self.v[2] < 0 and abs(a_net[2]) > self.eps:
                    # 满足约束
                    I = a_d[0] / self.tau_m / self.g
                    strategy_interceptor_stash.append([a_t, a_d, I])
                else:
                    # 不满足约束
                    I = self.penalty - abs(a_d[2] + self.v[2])
                    strategy_interceptor_stash.append([a_t, a_d, I])
            
            # 找出n_td*(at) 和 f_d*(at)
            I_max = -1e10
            a_d_star_at = np.array([0., 0., 0.])
            for s in strategy_interceptor_stash:
                if s[2] > I_max:
                    I_max = s[2]
                    a_d_star_at = s[1]
            strategy_pair_stash.append([a_t, a_d_star_at, I_max])

        # step2: 在策略集中找出最差情况的应对措施，记录双方策略
        I_min = 1e10
        a_t_star = np.array([0., 0., 0.])
        a_d_star = np.array([0., 0., 0.])

        for s in strategy_pair_stash:
            if s[2] < I_min:
                I_min = s[2]
                a_t_star = s[0]
                a_d_star = s[1]

        self.a_t = a_t_star
        self.a_d = a_d_star
        self.a_net = self.a_d - self.a_t + self.a_fic
        self.I = I_min

        self.x += self.v * dt
        self.v += self.a_net * dt
        self.traj = np.vstack((self.traj, self.x))

    def plot(self):
        plt.scatter(0, 0, color='#b3b3b3')
        plt.scatter(self.x[0], self.x[2], color='#000000')
        plt.arrow(self.x[0], self.x[2], self.a_net[0], self.a_net[2], width=0.1, color='#983f1d')   # a_net
        plt.text(self.x[0]+self.a_net[0]-0.2, self.x[2]+self.a_net[2]-1, r"${\bf{a}}_{\rm{net}}$")
        plt.arrow(self.x[0], self.x[2], self.a_d[0], self.a_d[2], width=0.1, color='#31859b')   # a_d
        plt.text(self.x[0]+self.a_d[0]-0.2, self.x[2]+self.a_d[2]-1, r"${\bf{a}}_{\rm{d}}$")
        plt.arrow(self.x[0], self.x[2], self.v[0], self.v[2], width=0.1, color='#205867')   # v
        plt.text(self.x[0]+self.v[0]-0.2, self.x[2]+self.v[2]+0.5, r"$\bf{v}$")
        plt.arrow(0, 0, self.a_t[0], self.a_t[2], width=0.1, color='#eb7224')   # a_t
        plt.text(self.a_t[0]-0.2, self.a_t[2]+0.5, r"${\bf{a}}_{\rm{t}}$")
        plt.text(10, 10, "I={}".format(self.I))     #I

        plt.plot(self.traj[:,0], self.traj[:,2])
        scale = max(abs(self.x)//10)*10 + 10
        plt.xlim((-scale, scale))
        plt.ylim((-scale, scale))

def sat(a, s):
    n = np.linalg.norm(a)
    if n > s:
        return a / n * s
    else:
        return a

def vec2ang(a:np.array) -> float:
    """
    xoz平面上向量转欧拉角(俯仰角)
    """
    return np.arctan2(a[2], a[0])

def ang2vec(ang:float, mag:float) -> np.array:
    """
    xoz平面欧拉角转向量, 输入角度和幅值
    """
    a = np.array([0., 0., 0.])
    a[0] = mag * np.cos(ang)
    a[2] = mag * np.sin(ang)
    return a

if __name__ == "__main__":
    # interceptor = "MC"
    # x0 = [-10., 0, 0.]
    # v0 = [1., 0, 0.5]
    # m = 1.0
    # tau_m = 2     # 2可拦截；1.3刚好逃逸；0.8快速逃逸，加速能力小于目标
    # tau_tm = 1
    # fov = np.deg2rad(90)    # 相机视场角
    # ang_install = np.deg2rad(30)    # pi/2-ang_install 为力和光轴夹角，典型值0、pi/2、deg2rad(-15)
    # dt = 0.01

    # interceptor = "FW"
    # x0 = [-10., 0, 0.]
    # v0 = [1., 0, -2]
    # m = 1.0
    # tau_m = 0.2     # 2可拦截；1.3刚好逃逸；0.8快速逃逸，加速能力小于目标
    # tau_tm = 0.1
    # fov = np.deg2rad(90)    # 相机视场角
    # ang_install = np.deg2rad(-15)    # pi/2-ang_install 为力和光轴夹角，典型值0、pi/2、deg2rad(-15)
    # dt = 0.01

    interceptor = "AM"
    x0 = [-10., 0, 0.]
    v0 = [1., 0, -2]
    m = 1.0
    tau_m = 2     # 2可拦截；1.3刚好逃逸；0.8快速逃逸，加速能力小于目标
    tau_tm = 1
    fov = np.deg2rad(90)    # 相机视场角
    ang_install = np.pi/2    # pi/2-ang_install 为力和光轴夹角，典型值0、pi/2、deg2rad(-15)
    dt = 0.01

    rb = RigidBody(interceptor=interceptor, x0=x0, v0=v0, m=m, tau_m=tau_m, tau_tm=tau_tm, fov=fov)
    plt.ion()  #打开交互模式
    while True:
        plt.clf()  #清除图像
        rb.update(dt)
        rb.plot()
        plt.pause(dt)
        plt.show()
