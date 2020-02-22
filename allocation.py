import json
import numpy as np
import time
import airsim
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from recursive_hungarian import RHA2


class Allocation:
    def __init__(self):
        self.nums = 1
        self.width = 640
        self.height = 480
        self.home = []
        self.low = np.array([[0, 100, 65], [80, 50, 130]])
        self.high = np.array([[10, 200, 130], [100, 160, 255]])
        self.targets = len(self.low)


    def calc_centroid(self, image_bgr, i):
        min_prop = 0.00001

        image_hue = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        cent = []
        for t in range(self.targets):
            th = cv2.inRange(image_hue, self.low[t], self.high[t])
            dilated = cv2.dilate(th,
                                cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (3, 3)),
                                iterations=1)
            # if t == 1:
            #     cv2.imshow("Dilated{}".format(i), dilated)
            
            M = cv2.moments(dilated, binaryImage=True)
            if M["m00"] >= min_prop * self.height * self.width:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cent.append([cx, cy]) 
            else:
                cent.append([-1, -1])
        return cent


    def reconstruction(self, feature, pose, angle) -> np.array:
        def quaternion2rotation(q):
            w, x, y, z = q[0], q[1], q[2], q[3]
            return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                             [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                             [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])

        f, u0, v0 = self.width/2, self.width/2, self.height/2
        K = np.array([[f,0,u0], [0,f,v0], [0,0,1]])
        R_bc = np.array([[0,1,0], [0,0,1], [1,0,0]])

        MM = []
        for i in range(self.nums):
            vehicle_name = "Drone{}".format(i)
            R_be = quaternion2rotation(angle[i])
            R_ec = np.dot(R_bc, R_be.T)
            T_ce = np.array(pose[i]).reshape((-1,1))
            MM.append( K.dot(R_ec).dot(np.hstack((np.identity(3),-T_ce))) )
            print("[{}]: R_ec: {}, T_ce: {}".format(vehicle_name, R_ec, T_ce))

        target_pos = np.zeros((1,3))
        for t in range(self.targets):
            srcA, srcb = [], []
            for i in range(self.nums):
                if feature[i][t] == [-1, -1]:
                    continue
                xi, yi = feature[i][t][0], feature[i][t][1]
                M = MM[i]
                srcA.append([M[2,0]*xi-M[0,0], M[2,1]*xi-M[0,1], M[2,2]*xi-M[0,2]])
                srcA.append([M[2,0]*yi-M[1,0], M[2,1]*yi-M[1,1], M[2,2]*yi-M[1,2]])
                srcb.append([M[0,3]-M[2,3]*xi])
                srcb.append([M[1,3]-M[2,3]*yi])

            if len(srcA) >= 4:
                ret, dstP = cv2.solve(np.array(srcA), np.array(srcb), flags=cv2.DECOMP_SVD)
                if not ret:
                    print("Solve Failed!!!")
                target_pos = np.vstack((target_pos, dstP.reshape((1,-1))))
            else:
                target_pos = np.vstack((target_pos, np.array([-1,-1,-1])))
        return target_pos[1:]


    def draw_reticle(self, image, feature):
        x, y = feature[0], feature[1]
        s = 10
        cv2.rectangle(image, (x-s, y-s), (x+s, y+s), (0, 0, 255), 2)
        cv2.line(image, (x, y-2*s), (x, y+2*s), (0, 0, 255), 2)
        cv2.line(image, (x-2*s, y), (x+2*s, y), (0, 0, 255), 2)


    def main(self):
        settings_file = open("/home/zhenglong/Documents/AirSim/settings.json")
        settings = json.load(settings_file)
        self.nums = len(settings["Vehicles"])
        self.height = settings["CameraDefaults"]["CaptureSettings"][0]["Height"]
        self.width = settings["CameraDefaults"]["CaptureSettings"][0]["Width"]
        print("The num of drones: {}".format(self.nums))
        for i in range(self.nums):
            vehicle_name = "Drone{}".format(i)
            self.home.append([settings["Vehicles"][vehicle_name]["X"],
                              settings["Vehicles"][vehicle_name]["Y"],
                              settings["Vehicles"][vehicle_name]["Z"]])

        client = airsim.MultirotorClient()
        client.confirmConnection()

        for i in range(self.nums):
            vehicle_name = "Drone{}".format(i)
            client.enableApiControl(True, vehicle_name=vehicle_name)
            client.armDisarm(True, vehicle_name=vehicle_name)
            client.takeoffAsync(vehicle_name=vehicle_name)

        while True:
            stash_pose = []
            stash_angle = []
            stash_feature = []
            image_bgr = []

            for i in range(self.nums):
                vehicle_name = "Drone{}".format(i)
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                ],
                                                vehicle_name=vehicle_name)
                response = responses[0]
                if response is None:
                    print(
                        "Camera is not returning image, please check airsim for error messages"
                    )
                    sys.exit(0)
                else:
                    img1d = np.fromstring(response.image_data_uint8,
                                        dtype=np.uint8)
                    image_rgba = img1d.reshape(self.height, self.width, 4)
                    image_bgr.append( cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR) )

                cent = self.calc_centroid(image_bgr[i], i)
                stash_feature.append(cent)
                print("[{}]: feature: {}".format(vehicle_name, stash_feature[i]))

                kinematics = client.simGetGroundTruthKinematics(
                    vehicle_name=vehicle_name)
                stash_pose.append([
                    kinematics.position.x_val + self.home[i][0], 
                    kinematics.position.y_val + self.home[i][1],
                    kinematics.position.z_val + self.home[i][2]
                ])
                stash_angle.append([
                    kinematics.orientation.w_val, kinematics.orientation.x_val,
                    kinematics.orientation.y_val, kinematics.orientation.z_val
                ])
                print("[{}]: pose: {}, angle: {}".format(vehicle_name,
                                                        stash_pose[i],
                                                        stash_angle[i]))

                client.moveByVelocityAsync(
                    vx=-5,
                    vy=0,
                    vz=0,
                    duration=1,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    vehicle_name=vehicle_name)

            stash_pose = np.array(stash_pose)
            stash_angle = np.array(stash_angle)
            target_pos = self.reconstruction(stash_feature, stash_pose, stash_angle)
            print("target pose: {}".format(target_pos))

            r = RHA2(stash_pose, target_pos)
            task = r.deal()
            print("task: {}".format(task[1]))
            for i in range(self.nums):
                target_pos = np.array(target_pos).tolist()
                idx = target_pos.index(task[1][i].tolist())
                self.draw_reticle(image_bgr[i], stash_feature[i][idx])
                cv2.imshow("Image{}".format(i), image_bgr[i])

            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q') or key == ord('x')):
                break

        client.reset()


if __name__ == "__main__":
    a = Allocation()
    a.main()
