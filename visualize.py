import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from new_recursive_hungarian import RHA2

class Drone:
    def __init__(self, name, position, energy):
        self.name = name
        self.position = np.array(position, dtype=np.float64)
        self.energy = energy

    def __str__(self):
        return "name: {}; position: {}; energy: {}".format(self.name, self.position, self.energy)

    def update(self, interval, task_list, intruders_position):
        pass


class Intruder:
    def __init__(self, name, intrude_tactics, position, motion_model, motion, velocity, importance, entry_time=0):
        self.name = name
        self.intrude_tactics = intrude_tactics
        self.position = np.array(position, dtype=np.float64)
        self.motion_model = motion_model
        self.motion = motion
        self.velocity = velocity
        self.importance = importance
        self.entry_time = entry_time
        self.direction = np.array([0,0,0], dtype=np.float64)

        if self.intrude_tactics.intrude_mode == "Respective":
            if self.motion_model == "Linear":
                self.direction = self.motion.direction
            elif self.motion_model == "WPs":
                self.direction = self.motion.WPs[(self.motion.cnt_wp+1)%self.motion.num_wp] - self.motion.WPs[self.motion.cnt_wp]
                self.direction /= np.linalg.norm(self.direction)
        elif self.intrude_tactics.intrude_mode == "Assemble":
            self.motion_model = "Linear"
            self.direction = self.intrude_tactics.assembly_point - self.position
            self.direction /= np.linalg.norm(self.direction)
        elif self.intrude_tactics.intrude_mode == "Parallel":
            self.motion_model = "Linear"
            self.direction = self.intrude_tactics.direction
        else:
            raise TypeError("Invalid intrude mode: {}".format(self.intrude_tactics.intrude_mode))

    def __str__(self):
        return "name: {}, position: {}, motion_model: {}, motion: {}, velocity: {}, importance: {}, entry_time: {}".format(self.name, self.position, self.motion_model, self.motion, self.velocity, self.importance, self.entry_time)

    def update(self, interval):
        if self.motion_model == "Linear":
            self.position += interval * self.velocity * self.direction
        elif self.motion_model == "RandomWalk":
            self.position += interval * self.motion.step * np.random.uniform(-1, 1, 3)
        elif self.motion_model == "WayPoint":
            self.position += interval * self.velocity * self.direction
        else:
            raise TypeError("Invalid motion mode: {}".format(self.motion_model))


class IntrudeTactics:
    def __init__(self, intrude_mode, assembly_point, direction, velocity):
        if intrude_mode not in ["Respective", "Assemble", "Parallel"]:
            raise TypeError("Invalid intrude mode: {}".format(intrude_mode))
        self.intrude_mode = intrude_mode
        self.assembly_point = np.array(assembly_point, dtype=np.float64)
        self.direction = np.array(direction, dtype=np.float64) / np.linalg.norm(direction)
        self.velocity = velocity


class Motions:
    def __init__(self, name, direction=[1,0,0], step=0, WPs=[]):
        self.name = name
        self.direction = np.array(direction, dtype=np.float64) / np.linalg.norm(direction)
        self.step = step
        self.WPs = np.array(WPs, dtype=np.float64)
        self.num_wp = len(self.WPs)
        self.cnt_wp = 0


class Theater:
    def __init__(self, drones, intruders):
        self.num_drone = len(drones)
        self.num_intruder = len(intruders)
        self.drones = drones
        self.intruders = intruders
        self.task_list = list()
        self.is_reallocate = True
        self.ax = plt.axes(projection='3d')

    def __str__(self):
        prt = "[drones:]\n"
        prt += "\n".join([d.__str__() for d in self.drones])
        prt += "\n[intruders:]\n"
        prt += "\n".join([d.__str__() for d in self.intruders])
        return prt

    def step(self, fps=10):
        """
        function: 每一帧调用该函数一次
        :param fps: 帧率
        :return: None
        """
        # 迭代时间
        self.interval = 1 / fps
        start_time = time.time()
        # 更新敌我位置
        self.update()
        # 需要再分配时计算
        if self.is_reallocate:
            allocator = RHA2(self.get_agent_pos(), self.get_agent_energy(), self.get_task_pos(), self.get_task_importance())
            uavs_pos_record = allocator.deal()
        # 画图
        self.render(uavs_pos_record)
        # 维持帧率
        sleep_time = start_time + self.interval - time.time()
        if sleep_time > 0: plt.pause(sleep_time)

    def update(self):
        """
        function: 更新敌人位置和我方位置
        :return: None
        """
        # 更新敌人位置
        for i in range(self.num_intruder):
            self.intruders[i].update(self.interval)
        # 更新我方位置
        for i in range(self.num_drone):
            self.drones[i].update(self.interval, self.task_list, self.get_task_pos)

    def render(self, uavs_pos_record):
        """
        function: 画图
        :return: None
        """
        print(self)
        print("="*10)
        self.ax.cla()
        task_pos = self.get_task_pos()
        task_importance = self.get_task_importance()
        self.ax.scatter(task_pos[:,0], task_pos[:,1], task_pos[:,2], task_importance)

    def get_agent_pos(self):
        """
        function: 聚合我方位置信息，供RHA2使用
        :return: agent_pos: np.array()
        """
        agent_pos = []
        for i in range(self.num_drone):
            agent_pos.append(self.drones[i].position)
        return np.array(agent_pos)

    def get_agent_energy(self):
        """
        function: 聚合我方能量信息，供RHA2使用
        :return: agent_energy: np.array()
        """
        agent_energy = []
        for i in range(self.num_drone):
            agent_energy.append(self.drones[i].energy)
        return np.array(agent_energy)

    def get_task_pos(self):
        """
        function: 聚合敌方位置信息，供RHA2使用
        :return: task_pos: np.array()
        """
        task_pos = []
        for i in range(self.num_intruder):
            task_pos.append(self.intruders[i].position)
        return np.array(task_pos)

    def get_task_importance(self):
        """
        function: 聚合敌方重要度信息，供RHA2使用
        :return: task_importance: np.array()
        """
        task_importance = []
        for i in range(self.num_intruder):
            task_importance.append(self.intruders[i].importance)
        return np.array(task_importance)


def main(args):
    # 读配置文件
    config_file = open(args.config_file)
    config = json.load(config_file)
    print(json.dumps(config, indent=4))
    # 生成舞台
    drones = list()
    for key, value in config["Vehicles"].items():
        drones.append(Drone(key, value["Position"], value["Energy"]))

    config_tactics = config["IntrudeTactics"]
    intrude_tactics = IntrudeTactics(config_tactics["IntrudeMode"], config_tactics.get("AssemblyPoint", [0,0,0]),
        config_tactics.get("Direction", [1,0,0]), config_tactics.get("Velocity", 0))

    motions = dict()
    for key, value in config["Motions"].items():
        motions[key] = Motions(key, value.get("Direction", [1,0,0]), value.get("Step", 0), value.get("WPs", []))

    intruders = list()
    for key, value in config["Intruders"].items():
        intruders.append(Intruder(key, intrude_tactics, value["Position"], value["MotionModel"], 
            motions[value["MotionParams"]], value["Velocity"], value["Importance"], value.get("EntryTime", 0)))
    theater = Theater(drones, intruders)
    print(theater)
    # 表演开始
    while True:
        theater.step()
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", dest="config_file", default="./config.json")
    args = parser.parse_args()
    print(args)
    main(args)
