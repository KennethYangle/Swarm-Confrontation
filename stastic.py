import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import airsim
import time
from scipy import io

class IBVS:
    def __init__(self, size, target, cam_roll=0, cam_pitch=0, cam_yaw=0, T_cb=np.array([0,0,0]), is_filter=False):
        # 图像相关
        self.width = size[0]
        self.height = size[1]
        self.low = target[0]
        self.high = target[1]
        self.u0 = self.width/2
        self.v0 = self.height/2
        self.x0 = self.u0
        self.y0 = self.v0
        self.f = self.width/2
        # 相机相关
        self.cam_roll = cam_roll
        self.cam_pitch = cam_pitch
        self.cam_yaw = cam_yaw
        self.R_c0b = np.array([[0,0,1], [1,0,0], [0,1,0]])
        self.T_cb = T_cb
        # 开始结束标志
        self.is_finished = False
        self.min_prop = 0.00001
        self.max_prop = 0.3
        # yawrateVzController
        self.kx = 0.1
        self.kz = 0.01
        self.velocity = 10
        # 图像滤波
        self.is_filter = is_filter

    def calc_centroid(self, image_bgr):
        image_hue = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        th = cv2.inRange(image_hue, self.low, self.high)
        dilated = cv2.dilate(th, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        # cv2.imshow("dilated", dilated)
        M = cv2.moments(dilated, binaryImage=True)
        if M["m00"] >= self.min_prop * self.height * self.width:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cent = [cx, cy]
        else: 
            cent = [-1, -1]
            self.is_finished = True
        if M["m00"] >= self.max_prop * self.height * self.width:
            self.is_finished = True
        return cent

    def yawrateVzController(self, cent, angle):
        """
        - function: 通常使用的线速度加偏航角速度控制方法
        - params:
            - cent: 目标在图像上的坐标
            - q: 飞机在世界坐标系的四元数
        - return:
            - cmd: [vx, vy, vz, yawrate]，世界坐标系下的线速度与偏航角速度
        """
        ex, ey = cent[0] - self.width/2, cent[1] - self.height/2
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        # print(pitch, roll, yaw)
        return self.velocity*np.cos(yaw), \
               self.velocity*np.sin(yaw), \
               self.kz*ey, \
               self.kx*ex


if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    # client.takeoffAsync().join()

    width, height = 640, 480
    low = np.array([0, 170, 100])
    high = np.array([17, 256, 256])


    # 仿真100次
    simulation_result = []
    for it in range(100):

        servo = IBVS([width, height], [low, high])
        client.simSetCameraOrientation("0", airsim.to_quaternion(servo.cam_pitch, servo.cam_roll, servo.cam_yaw))
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(-20+5*np.random.random(), -2*np.random.random(), -2), airsim.to_quaternion(0, 0, 0)), True)

        while not servo.is_finished:
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            response = responses[0]
            if response is None:
                print("Camera is not returning image, please check airsim for error messages")
                sys.exit(0)
            else:
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
                image_bgr = img1d.reshape(height, width, 3)

            # 获取图上目标坐标和飞机相关状态
            cent = servo.calc_centroid(image_bgr)
            # print("cent: {}".format(cent))
            kinematics = client.simGetGroundTruthKinematics()
            q = kinematics.orientation
            vcy = kinematics.linear_velocity.z_val

            # 速度控制
            cmd = servo.yawrateVzController(cent_flt, q) if servo.is_filter else servo.yawrateVzController(cent, q)
            # print(cmd)
            client.moveByVelocityAsync(cmd[0], cmd[1], cmd[2], 1, 
                drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, 
                yaw_mode = airsim.YawMode(True, cmd[3]))

            cv2.imshow("img", image_bgr)
            cv2.waitKey(1)

        print("#Simulation {}#: last_cent: {}".format(it, cent))
        simulation_result.append(cent)

    io.savemat("./data/simulation_result.mat", {"matrix": simulation_result})

    airsim.wait_key('Press any key to reset')
    client.reset()
    client.simSetCameraOrientation("0", airsim.to_quaternion(0, 0, 0))
