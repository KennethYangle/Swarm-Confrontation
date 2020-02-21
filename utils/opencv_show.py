# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import airsim
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time
import numpy as np


def printUsage():
    print("Usage: python camera.py [depth|segmentation|scene]")

def calc_centroid(dilated):
    min_prop = 0.001

    M = cv2.moments(dilated, binaryImage = True)
    if M["m00"] >= 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    else:
        return -1, -1


cameraType = "scene"

for arg in sys.argv[1:]:
    cameraType = arg.lower()

cameraTypeMap = {
    "depth": airsim.ImageType.DepthVis,
    "segmentation": airsim.ImageType.Segmentation,
    "seg": airsim.ImageType.Segmentation,
    "scene": airsim.ImageType.Scene,
    "disparity": airsim.ImageType.DisparityNormalized,
    "normals": airsim.ImageType.SurfaceNormals
}

if (not cameraType in cameraTypeMap):
    printUsage()
    sys.exit(0)

print(cameraTypeMap[cameraType])

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
startTime = time.clock()
fps = 0

while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    responses = client.simGetImages([airsim.ImageRequest("0", cameraTypeMap[cameraType], False, False)])
    response = responses[0]
    if response is None:
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        image_rgba = img1d.reshape(response.height, response.width, 4)
        image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)
        cv2.putText(image_bgr, 'FPS ' + str(fps), textOrg, fontFace,
                    fontScale, (255, 0, 255), thickness)
        cv2.imshow("Image", image_bgr)

    image_hue = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low = np.array([0, 100, 65])
    high = np.array([10, 200, 130])
    th = cv2.inRange(image_hue, low, high)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    cv2.imshow("Dilated", dilated)
    cv2.waitKey(1)

    cx, cy = calc_centroid(dilated)
    print(cx, cy)


    endTime = time.clock()
    diff = endTime - startTime
    fps = 1 / diff
    startTime = endTime

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x')):
        break
