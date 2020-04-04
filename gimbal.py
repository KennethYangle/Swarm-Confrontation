import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import airsim

class IBVS:
    def __init__(self, size, target):
        # 图像相关
        self.width = size[0]
        self.height = size[1]
        self.low = target[0]
        self.high = target[1]
        # 开始结束标志
        self.is_finished = False
        self.min_prop = 0.00001
        self.max_prop = 0.3
        # yawrateVzController
        self.kx = 0.1
        self.kz = 0.01
        self.velocity = 10
        # angluarVelocityThrottleController
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = 0.02, 0.001, 2, 10, 0.01, -1
        self.theta_d, self.phi_d = -np.pi/6, 0
        self.hover = 0.594
        self.g = 9.8

    def calc_centroid(self, image_bgr):
        image_hue = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        th = cv2.inRange(image_hue, self.low, self.high)
        dilated = cv2.dilate(th, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        cv2.imshow("dilated", dilated)
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
        print(pitch, roll, yaw)
        return self.velocity*np.cos(yaw), \
               self.velocity*np.sin(yaw), \
               self.kz*ey, \
               self.kx*ex

    def angluarVelocityThrottleController(self, cnet, q, vcy, R_cb=np.array([[0,0,1], [1,0,0], [0,1,0]])):
        """
        - function: 2526中角速率控制方法
        - params: 
            - cnet: 目标在图像上的坐标
            - q: 飞机在世界坐标系的四元数
            - vcy: 飞机竖直方向上速度
            - R_cb: 相机系到机体系旋转矩阵。默认光轴与机头方向一致，图像平面水平为xc，光轴为zc，机体系为FRD
        - return:
            -cmd: [roll_rate, pitch_rate, yaw_rate, throttle], 三轴角速度和油门, FRD
        """
        def yawrate_saturation(yr):
            if yr > 1: return 1
            elif yr < -1: return -1
            else: return yr
        ex, ey = cent[0] - self.width/2, cent[1] - self.height/2
        print("ex: {}, ey: {}".format(ex, ey))
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        print("pitch: {}, roll: {}, yaw: {}, vcy: {}".format(pitch, roll, yaw, vcy))

        wcx = -self.k2*ey - self.k3*(pitch-self.theta_d)
        print("-self.k2*ey: {}, self.theta_d: {}, -self.k3*(pitch-self.theta_d): {}".format(-self.k2*ey, self.theta_d, -self.k3*(pitch-self.theta_d)))
        wcy = yawrate_saturation(self.k5 * ex)
        wcz = self.k6 * (roll-self.phi_d)
        print("wcx: {},wcy: {},wcz: {}".format(wcx, wcy, wcz))
        wbx, wby, wbz = R_cb.dot(np.array([wcx, wcy, wcz]))
        throttle = self.hover * (self.k4/self.g*(vcy-self.k1*ey)+1) / np.cos(pitch)
        return wbx, wby, wbz, throttle

if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    width, height = 640, 480
    low = np.array([0, 170, 100])
    high = np.array([17, 256, 256])
    servo = IBVS([width, height], [low, high])
    while not servo.is_finished:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        if response is None:
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            img1d = np.fromstring(response.image_data_uint8,
                                dtype=np.uint8)
            image_bgr = img1d.reshape(height, width, 3)

        cent = servo.calc_centroid(image_bgr)
        print(cent)
        kinematics = client.simGetGroundTruthKinematics()
        q = kinematics.orientation
        vcy = kinematics.linear_velocity.z_val

        # cmd = servo.yawrateVzController(cent, q)
        # print(cmd)
        # client.moveByVelocityAsync(cmd[0], cmd[1], cmd[2], 1, 
        #     drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, 
        #     yaw_mode = airsim.YawMode(True, cmd[3]))

        cmd = servo.angluarVelocityThrottleController(cent, q, vcy)
        print(cmd)
        client.moveByAngleRatesThrottleAsync(cmd[0], -cmd[1], -cmd[2], cmd[3], 1)

        # client.moveByAngleRatesThrottleAsync(0.01, 0, 0.1, 0.65, 1)
        cv2.imshow("img", image_bgr)
        cv2.waitKey(1)

    airsim.wait_key('Press any key to reset')
    client.reset()