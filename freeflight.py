import numpy as np
import time

class UAV:
    def __init__(self, home: list, waypoint: list):
        # parameters
        self.l = 5.0
        # status, with np.array
        self.position = np.array(home)
        self.velocity = np.array([0, 0, 0])
        self.waypoint = np.array(waypoint)
        self.ksi = self.filtered(self.position, self.velocity)
        self.arrive = False
        # control command
        self.attraction = np.array([0.0, 0.0, 0.0])
        self.repulsion = np.array([0.0, 0.0, 0.0])

    def filtered(self, pos: np.array, vel: np.array) -> np.array:
        return pos + vel / self.l

    def update(self, pos: list, vel: list):
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self.ksi = self.filtered(self.position, self.velocity)
        if np.linalg.norm(self.position - self.waypoint) <= 0.1:
            self.arrive = True
        else:
            self.arrive = False


class FreeFlight:
    def __init__(self, homes: list, waypoints: list):
        # UAVs status
        self.nums = len(homes)
        self.swarm = list()
        for i in range(self.nums):
            self.swarm.append(UAV(homes[i], waypoints[i]))
        # parameters
        self.rs = 0.6
        self.ra = 2.0
        self.vm = 2.0
        self.k1 = 1.0
        self.k2 = 1.0
        self.eps = 1e-6
        self.eps_s = 1e-6
        # ending flag 
        self.ending = False

    def controller(self) -> list:
        commands = list()
        # for each UAV
        for i in range(self.nums):
            # attractive potential
            ksi_wp = self.swarm[i].ksi - self.swarm[i].waypoint
            self.swarm[i].attraction = -1 * self.sat(self.k1 * ksi_wp, self.vm)

            # repulsive potentials
            self.swarm[i].repulsion = np.array([0.0, 0.0, 0.0])
            for j in range(self.nums):
                if j == i: continue
                ksi_m = self.swarm[i].ksi - self.swarm[j].ksi
                nksi_m = np.linalg.norm(ksi_m)
                den = (1+self.eps)*nksi_m - 2*self.rs*self.s(nksi_m/2/self.rs, self.eps_s)
                num = self.dsigma_m(nksi_m)*den - self.sigma_m(nksi_m)*( 1+self.eps-2*self.rs*self.ds(nksi_m/2/self.rs, self.eps_s) )
                b = self.k2 / nksi_m * num / den**2
                self.swarm[i].repulsion += -1 * b * ksi_m

            commands.append( (self.sat(self.swarm[i].attraction + self.swarm[i].repulsion, self.vm)).tolist() )
        return commands

    def update(self, positions: list, velocitys: list):
        flag = True
        for i in range(self.nums):
            self.swarm[i].update(positions[i], velocitys[i])
            if self.swarm[i].arrive == False:
                flag = False
        self.ending = flag

    def sat(self, v: np.array, vm: float) -> np.array:
        normv = np.linalg.norm(v)
        if normv > vm:
            return vm / normv * v
        else:
            return v

    def s(self, x: float, e: float) -> float:
        x2 = 1 + e / np.tan( 67.5/180*np.pi )
        x1 = x2 - np.sin( 45/180*np.pi ) * e
        if x <= x1:
            return x
        elif x <= x2:
            return 1 - e + np.sqrt( e**2 - (x-x2)**2 )
        else:
            return 1.0

    def ds(self, x: float, e: float) -> float:
        x2 = 1 + e / np.tan( 67.5/180*np.pi )
        x1 = x2 - np.sin( 45/180*np.pi ) * e
        if x <= x1:
            return 1.0
        elif x <= x2:
            return (x2 - x) / np.sqrt( e**2 - (x-x2)**2 )
        else:
            return 0.0

    def sigma_m(self, x: float) -> float:
        d1 = 2 * self.rs
        d2 = self.ra + self.rs
        if x <= d1:
            return 1.0
        elif x <= d2:
            A = -2 / (d1 - d2) ** 3
            B = 3 * (d1 + d2) / (d1 - d2) ** 3
            C = -6 * d1 * d2 / (d1 - d2) ** 3
            D = d2**2 * (3*d1 - d2) / (d1 - d2) ** 3
            return A * x**3 + B * x**2 + C * x + D
        else:
            return 0.0

    def dsigma_m(self, x: float) -> float:
        d1 = 2 * self.rs
        d2 = self.ra + self.rs
        if x <= d1:
            return 0.0
        elif x <= d2:
            A = -2 / (d1 - d2) ** 3
            B = 3 * (d1 + d2) / (d1 - d2) ** 3
            C = -6 * d1 * d2 / (d1 - d2) ** 3
            return 3*A * x**2 + 2*B * x + C
        else:
            return 0.0


if __name__ == "__main__":
    import airsim
    import json

    # get UAVs info from settings file and instantiate FreeFlight object
    settings_file = open("/home/zhenglong/Documents/AirSim/settings.json")
    settings = json.load(settings_file)
    nums = len(settings["Vehicles"])
    homes = list()
    for i in range(nums):
        vehicle_name = "Drone{}".format(i)
        homes.append([settings["Vehicles"][vehicle_name]["X"],
                      settings["Vehicles"][vehicle_name]["Y"],
                      settings["Vehicles"][vehicle_name]["Z"]])
    waypoints = list()
    for i in range(nums):
        waypoints.append(homes[(i+2)%nums])
    ff = FreeFlight(homes, waypoints)

    # airsim API
    client = airsim.MultirotorClient()
    client.confirmConnection()
    for i in range(nums):
        vehicle_name = "Drone{}".format(i)
        client.enableApiControl(True, vehicle_name=vehicle_name)
        client.armDisarm(True, vehicle_name=vehicle_name)
        client.takeoffAsync(vehicle_name=vehicle_name)

    # main loop
    while True:
        if ff.ending:
            break
        
        commands = ff.controller()
        for i in range(nums):
            vehicle_name = "Drone{}".format(i)
            vx = commands[i][0]
            vy = commands[i][1]
            vz = commands[i][2]
            client.moveByVelocityAsync(vx, vy, vz, 1, vehicle_name = vehicle_name)
        
        for i in range(nums):
            print("UAV{} attraction: {}, repulsion: {}".format(i, ff.swarm[i].attraction, ff.swarm[i].repulsion))

        positions = list()
        velocitys = list()
        for i in range(nums):
            vehicle_name = "Drone{}".format(i)
            kinematics = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
            positions.append([
                kinematics.position.x_val + homes[i][0], 
                kinematics.position.y_val + homes[i][1],
                kinematics.position.z_val + homes[i][2]
            ])
            velocitys.append([
                kinematics.linear_velocity.x_val,
                kinematics.linear_velocity.y_val,
                kinematics.linear_velocity.z_val
            ])
        ff.update(positions, velocitys)

    time.sleep(2)
    client.reset()
    