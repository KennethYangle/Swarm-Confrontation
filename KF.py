import numpy as np

class KF:
    def __init__(self, F, H, dt, Q, R):
        self.F = F
        self.H = H
        self.dt = dt
        self.Q = Q
        self.R = R

        self.n_state = self.F.shape[0]
        self.initial()
        self.prediction()

    def initial(self):
        self.x_k_k = np.zeros(self.n_state)
        self.P_k_k = np.identity(self.n_state) * 10
        return self.x_k_k

    def prediction(self):
        self.x_k_kp = self.F.dot(self.x_k_k)
        self.P_k_kp = self.F.dot(self.P_k_k) + self.Q

    def measurement(self, z):
        S_k = self.R + self.H.dot(self.P_k_kp).dot(self.H.T)
        K_k = self.P_k_kp.dot(self.H.T).dot( np.linalg.inv(S_k) )
        self.P_k_k = self.P_k_kp - K_k.dot(self.H).dot(self.P_k_kp)

        nu = np.array(z) - self.H.dot(self.x_k_kp)
        self.x_k_k = self.x_k_kp + K_k.dot(nu)

    def update(self, z):
        self.measurement(z)
        self.prediction()
        return self.x_k_kp
