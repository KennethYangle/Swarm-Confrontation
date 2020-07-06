import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import airsim
import time
from KF import KF

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
        # 云台相关
        self.k_camyaw = 0.02
        self.k_campitch = -0.02
        self.interval = 0.01
        # 开始结束标志
        self.is_finished = False
        self.min_prop = 0.00001
        self.max_prop = 0.3
        # yawrateVzController
        self.kx = 0.1
        self.kz = 0.01
        self.velocity = 10
        # angluarVelocityThrottleController
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = 0.02, 0.001, 2, 10, 0.01, -2
        self.theta_d, self.phi_d = -np.pi/6, 0
        self.hover = 0.594
        self.g = 9.8
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
            - cmd: [roll_rate, pitch_rate, yaw_rate, throttle], 三轴角速度和油门, FRD
        """
        ex, ey = cent[0] - self.x0, cent[1] - self.y0
        print("ex: {}, ey: {}".format(ex, ey))
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        print("pitch: {}, roll: {}, yaw: {}, vcy: {}".format(pitch, roll, yaw, vcy))

        wcx = -self.k2*ey - self.k3*(pitch-self.theta_d)
        print("-self.k2*ey: {}, self.theta_d: {}, -self.k3*(pitch-self.theta_d): {}".format(-self.k2*ey, self.theta_d, -self.k3*(pitch-self.theta_d)))
        wcy = saturation(self.k5*ex, 1)
        wcz = self.k6 * (roll-self.phi_d)
        print("wcx: {}, wcy: {}, wcz: {}".format(wcx, wcy, wcz))
        wbx, wby, wbz = R_cb.dot(np.array([wcx, wcy, wcz]))
        throttle = self.hover * (self.k4/self.g*(vcy-self.k1*ey)+1) / np.cos(pitch)
        return wbx, wby, wbz, throttle

    def angluarVelocityThrottleWithGimbalLock(self, cnet, q, vcy):
        """
        - function: 2526中角速率控制方法，同时云台运动
        - params: 
            - cnet: 目标在图像上的坐标
            - q: 飞机在世界坐标系的四元数
            - vcy: 飞机竖直方向上速度
        - return:
            - cmd: [roll_rate, pitch_rate, yaw_rate, throttle, cam_rollrate, cam_pitchrate, cam_yawrate], 三轴角速度和油门(FRD)，期望相机姿态
        """
        R_cc0 = to_RotationMatrix(-self.cam_yaw, self.cam_pitch, self.cam_roll)
        R_cb = self.R_c0b.dot(R_cc0)
        r = R_cb.T.dot(np.array([1,0,0]))
        t = -R_cb.T.dot(self.T_cb)     # T_bc
        self.x0 = r[0]/r[2]*(self.f-t[2]) + t[0] + self.u0
        self.y0 = r[1]/r[2]*(self.f-t[2]) + t[1] + self.v0
        print("x0: {}, y0: {}".format(servo.x0, servo.y0))

        ex, ey = cent[0] - self.x0, cent[1] - self.y0
        ecx, ecy = cnet[0] - self.width/2, cent[1] - self.height/2
        print("ex: {}, ey: {}".format(ex, ey))
        print("ecx: {}, ecy: {}".format(ecx, ecy))
        pitch, roll, yaw = airsim.to_eularian_angles(q)
        print("pitch: {}, roll: {}, yaw: {}, vcy: {}".format(pitch, roll, yaw, vcy))

        wcx = -self.k2*ey - self.k3*(pitch-self.theta_d)
        wcy = saturation(self.k5*ex, 1)
        wcz = self.k6 * (roll-self.phi_d)
        print("wcx: {}, wcy: {}, wcz: {}".format(wcx, wcy, wcz))
        wbx, wby, wbz = R_cb.dot(np.array([wcx, wcy, wcz]))
        throttle = self.hover * (self.k4/self.g*(vcy-self.k1*ey)+1) / np.cos(pitch)

        cam_rollrate = 0
        cam_pitchrate = self.k_campitch * ecy
        cam_yawrate = self.k_camyaw * ecx
        return wbx, wby, wbz, throttle, cam_rollrate, cam_pitchrate, cam_yawrate

    def simGimbalRatate(self, cam_rollrate, cam_pitchrate, cam_yawrate):
        """
        - function: 云台按角速率运动
        - params: 
            - cam_rollrate: 滚转角速率
            - cam_pitchrate: 俯仰角速率
            - cam_yawrate: 偏航角速率
        """
        self.cam_roll += cam_rollrate * self.interval
        self.cam_pitch += cam_pitchrate * self.interval
        self.cam_yaw += cam_yawrate * self.interval
        print("cam_roll: {}, cam_pitch: {}, cam_yaw: {}".format(self.cam_roll, self.cam_pitch, self.cam_yaw))


def to_RotationMatrix(pitch, roll, yaw):
    """
    - function: 欧拉角转旋转矩阵
    - params: 
        - pitch: 绕y轴
        - roll: 绕x轴
        - yaw: 绕z轴
    - return:
        -R: 旋转矩阵
    """
    q = airsim.to_quaternion(pitch, roll, yaw)
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                    [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                    [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

def saturation(x, s):
    if x > s: return s
    elif x < -s: return -s
    else: return x


if __name__ == "__main__":
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    width, height = 640, 480
    low = np.array([0, 170, 100])
    high = np.array([17, 256, 256])
    servo = IBVS([width, height], [low, high], cam_pitch=np.pi/12, is_filter=True)
    client.simSetCameraOrientation("0", airsim.to_quaternion(servo.cam_pitch, servo.cam_roll, servo.cam_yaw))

    if servo.is_filter:
        # image filter initialize
        dt = 0.03
        F = np.array([[1,0,dt,0,0.5*dt*dt,0],[0,1,0,dt,0,0.5*dt*dt],[0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        H = np.hstack((np.identity(2), np.zeros((2,4))))
        Q = np.identity(6) * 10
        R = np.identity(2) * 10
        flt = KF(F, H, dt, Q, R)

    starttime = time.time()
    cnt = 1
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
        print("cent: {}".format(cent))
        kinematics = client.simGetGroundTruthKinematics()
        q = kinematics.orientation
        vcy = kinematics.linear_velocity.z_val

        # 滤波更新
        if servo.is_filter:
            tmp = flt.update(cent)
            cent_flt = tuple(int(a) for a in tmp)[:2]
            print("cent_flt: {}".format(cent_flt))
            cv2.circle(image_bgr, tuple(cent), 1, (0,255,0), 4)
            cv2.circle(image_bgr, cent_flt, 1, (255,0,0), 4)

        # 速度控制
        cmd = servo.yawrateVzController(cent_flt, q) if servo.is_filter else servo.yawrateVzController(cent, q)
        print(cmd)
        client.moveByVelocityAsync(cmd[0], cmd[1], cmd[2], 1, 
            drivetrain = airsim.DrivetrainType.MaxDegreeOfFreedom, 
            yaw_mode = airsim.YawMode(True, cmd[3]))

        # # 相机安装或云台旋转
        # client.simSetCameraOrientation("0", airsim.to_quaternion(servo.cam_pitch, servo.cam_roll, servo.cam_yaw))
        # R_cc0 = to_RotationMatrix(-servo.cam_yaw, servo.cam_pitch, servo.cam_roll)
        # R_cb = servo.R_c0b.dot(R_cc0)
        # r = R_cb.T.dot(np.array([1,0,0]))
        # print("R_cb: {}".format(R_cb))
        # print("r: {}".format(r))
        # t = -R_cb.T.dot(servo.T_cb)     # T_bc
        # servo.x0 = r[0]/r[2]*(servo.f-t[2]) + t[0] + servo.u0
        # servo.y0 = r[1]/r[2]*(servo.f-t[2]) + t[1] + servo.v0
        # print("x0: {}, y0: {}".format(servo.x0, servo.y0))

        # # 角速度控制
        # cmd = servo.angluarVelocityThrottleController(cent, q, vcy, R_cb)
        # print(cmd)
        # client.moveByAngleRatesThrottleAsync(cmd[0], -cmd[1], -cmd[2], cmd[3], 1)

        # # 角速度控制+云台跟踪
        # cmd = servo.angluarVelocityThrottleWithGimbalLock(cent, q, vcy)
        # print(cmd)
        # client.moveByAngleRatesThrottleAsync(cmd[0], -cmd[1], -cmd[2], cmd[3], 1)   # 添加负号从FRD转为FLU
        # interval = (time.time() - starttime) / cnt
        # servo.simGimbalRatate(cmd[4], cmd[5], cmd[6])
        # client.simSetCameraOrientation("0", airsim.to_quaternion(servo.cam_pitch, servo.cam_roll, servo.cam_yaw))
        # print("interval: {}".format(interval))

        cv2.imshow("img", image_bgr)
        cv2.waitKey(1)
        cnt += 1

    airsim.wait_key('Press any key to reset')
    client.reset()
    client.simSetCameraOrientation("0", airsim.to_quaternion(0, 0, 0))
